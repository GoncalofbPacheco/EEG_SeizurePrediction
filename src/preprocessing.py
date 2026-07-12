"""
preprocessing.py
================
Implements the full preprocessing pipeline (thesis §3.2):

  1. Band-pass filter  0.5 – 30 Hz, zero-phase Butterworth (order 4)
  2. Segment into 20-second windows with 50 % overlap
  3. Assign labels:
       1 = preictal   [onset − 30 min,  onset − 5 min]  (SPH excluded)
       0 = interictal  (with post-ictal 30-min exclusion)
       -1 = excluded   (ictal / transition / boundary overlap)

Windows labelled −1 are discarded before feature extraction.
Seizures for which the full 30-min preictal window cannot fit within the
recording are excluded from positive labelling (preictal windows not created).
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy.signal import butter, filtfilt

from config import (
    FS,
    LOWCUT,
    HIGHCUT,
    FILTER_ORDER,
    WINDOW_SAMP,
    STEP_SAMP,
    WINDOW_SEC,
    STEP_SEC,
    PREICTAL_SEC,
    SPH_SEC,
    POSTICTAL_EXCL_SEC,
)

Label = int  # 0 = interictal, 1 = preictal, -1 = excluded


# ── 1. Band-pass filter ───────────────────────────────────────────────────────


def bandpass_filter(data: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Apply a zero-phase band-pass Butterworth filter to multichannel EEG.

    Parameters
    ----------
    data : np.ndarray  shape (n_channels, n_samples)
    fs   : int         sampling frequency

    Returns
    -------
    filtered : np.ndarray  same shape as data
    """
    nyq = 0.5 * fs
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, data, axis=1)


# ── 2. Segmentation ───────────────────────────────────────────────────────────


def segment_signal(data: np.ndarray) -> np.ndarray:
    """
    Slice a multichannel signal into overlapping windows.

    Parameters
    ----------
    data : np.ndarray  shape (n_channels, n_samples)

    Returns
    -------
    windows : np.ndarray  shape (n_windows, n_channels, WINDOW_SAMP)
    """
    n_channels, n_samples = data.shape
    starts = range(0, n_samples - WINDOW_SAMP + 1, STEP_SAMP)
    windows = np.stack([data[:, s : s + WINDOW_SAMP] for s in starts], axis=0)
    return windows  # (n_windows, n_channels, n_samples_per_window)


def window_start_times(n_samples: int) -> np.ndarray:
    """Return the start time (seconds) of each window given total sample count."""
    starts_samp = np.arange(0, n_samples - WINDOW_SAMP + 1, STEP_SAMP)
    return starts_samp / FS


# ── 3. Labelling ──────────────────────────────────────────────────────────────


def _build_exclusion_mask(
    n_samples: int, seizures: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Create a sample-level exclusion mask for ictal and post-ictal regions.

    Returns a boolean array of length n_samples; True = sample is excluded.
    """
    mask = np.zeros(n_samples, dtype=bool)
    for onset, offset in seizures:
        s_on = int(onset * FS)
        s_off = int(offset * FS)
        s_post_end = min(s_off + int(POSTICTAL_EXCL_SEC * FS), n_samples)
        mask[s_on:s_post_end] = True
    return mask


def _build_preictal_mask(
    n_samples: int, seizures: List[Tuple[float, float]], recording_duration_sec: float
) -> np.ndarray:
    """
    Create a sample-level preictal mask.

    Preictal region for each seizure: [onset − PREICTAL_SEC, onset − SPH_SEC]
    Seizures where onset − PREICTAL_SEC < 0 are excluded (boundary condition).

    Returns a boolean array of length n_samples; True = sample is preictal.
    """
    mask = np.zeros(n_samples, dtype=bool)
    for onset, _ in seizures:
        preictal_start = onset - PREICTAL_SEC
        preictal_end = onset - SPH_SEC

        # Skip if the full 30-min window cannot be reconstructed
        if preictal_start < 0:
            print(
                f"    [LABEL] Seizure at {onset:.0f}s: preictal window out of bounds "
                f"(would start at {preictal_start:.0f}s) — skipping preictal label."
            )
            continue

        s_start = int(preictal_start * FS)
        s_end = int(preictal_end * FS)
        mask[s_start:s_end] = True

    return mask


def label_windows(
    n_samples: int, seizures: List[Tuple[float, float]], fs: int = FS
) -> np.ndarray:
    """
    Assign a label to each 20-second window extracted from a recording.

    Label rules (thesis §3.2)
    -------------------------
    •  A window is labelled only when **fully contained** within one state.
    •  Any window overlapping an exclusion boundary is labelled −1 and discarded.
    •  Priority: excluded (−1) > preictal (1) > interictal (0)

    Parameters
    ----------
    n_samples : int   Total samples in the recording.
    seizures  : list  [(onset_sec, offset_sec), ...] for this file.
    fs        : int   Sampling frequency.

    Returns
    -------
    labels : np.ndarray  shape (n_windows,)  dtype int8
    """
    excl_mask = _build_exclusion_mask(n_samples, seizures)
    preictal_mask = _build_preictal_mask(
        n_samples, seizures, recording_duration_sec=n_samples / fs
    )

    starts = np.arange(0, n_samples - WINDOW_SAMP + 1, STEP_SAMP)
    n_windows = len(starts)
    labels = np.zeros(n_windows, dtype=np.int8)

    for i, s in enumerate(starts):
        e = s + WINDOW_SAMP
        window_excl = excl_mask[s:e]
        window_preictal = preictal_mask[s:e]

        if window_excl.any():
            labels[i] = -1  # overlaps ictal / post-ictal → exclude
        elif window_preictal.all():
            labels[i] = 1  # fully within preictal zone
        elif window_preictal.any():
            labels[i] = -1  # straddles preictal boundary → exclude
        else:
            labels[i] = 0  # interictal

    return labels


# ── 4. Full per-file pipeline ─────────────────────────────────────────────────


def preprocess_file(
    data: np.ndarray, seizures: List[Tuple[float, float]], fs: int = FS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the complete preprocessing pipeline on a single EDF file.

    Steps:
      1. Band-pass filter
      2. Segment into overlapping windows
      3. Label each window
      4. Discard excluded windows (label == −1)

    Parameters
    ----------
    data     : np.ndarray  shape (n_channels, n_samples)  raw EEG
    seizures : list        [(onset_sec, offset_sec), ...]
    fs       : int         sampling frequency

    Returns
    -------
    windows    : np.ndarray  shape (n_valid, n_channels, WINDOW_SAMP)
    labels     : np.ndarray  shape (n_valid,)  {0, 1}
    start_times: np.ndarray  shape (n_valid,)  window start in seconds
    """
    # 1. Filter
    filtered = bandpass_filter(data, fs)

    # 2. Segment
    windows_all = segment_signal(filtered)  # (n_windows, n_ch, samp)
    starts_all = window_start_times(data.shape[1])  # (n_windows,)

    # 3. Label
    labels_all = label_windows(data.shape[1], seizures, fs)

    # 4. Keep only valid (non-excluded) windows
    valid = labels_all != -1
    windows = windows_all[valid]
    labels = labels_all[valid]
    start_times = starts_all[valid]

    return windows, labels, start_times


def print_label_stats(labels: np.ndarray, patient_id: str = "", fname: str = ""):
    """Print a quick summary of preictal/interictal counts."""
    n_preictal = (labels == 1).sum()
    n_interictal = (labels == 0).sum()
    prefix = f"[{patient_id}/{fname}]" if fname else f"[{patient_id}]"
    print(
        f"{prefix}  preictal={n_preictal:4d}  interictal={n_interictal:5d}  "
        f"ratio=1:{n_interictal // max(n_preictal, 1)}"
    )
