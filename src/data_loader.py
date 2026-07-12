"""
data_loader.py
==============
Loads CHB-MIT EDF files using MNE and extracts the 18 canonical bipolar channels
defined in Table 3 of the thesis.

Channel name normalisation
--------------------------
CHB-MIT labels occasionally use hyphens, spaces, or case variants.
We normalise everything to uppercase with a single hyphen (e.g. 'FP1-F7').
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import mne

mne.set_log_level("WARNING")   # suppress MNE info spam

from config import CANONICAL_CHANNELS, FS, N_CHANNELS


# ── Channel name normalisation ────────────────────────────────────────────────

def _normalise_ch_name(name: str) -> str:
    """Uppercase, strip whitespace, collapse multiple hyphens / spaces."""
    name = name.strip().upper()
    name = name.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    name = name.replace("  ", " ")
    return name


def _build_ch_index(raw_ch_names: List[str],
                    target_channels: List[str]) -> List[int]:
    """
    Return indices into raw_ch_names for each channel in target_channels.

    Handles the CHB-MIT quirk where T8-P8 appears twice in many EDFs.
    MNE renames duplicates to 'T8-P8-0' and 'T8-P8-1'; we take the first
    occurrence (-0) whenever an exact match is not found.

    Raises ValueError if any target channel is missing.
    """
    normalised = {_normalise_ch_name(ch): i for i, ch in enumerate(raw_ch_names)}
    indices = []
    missing = []
    for ch in target_channels:
        key = _normalise_ch_name(ch)
        if key in normalised:
            indices.append(normalised[key])
        elif key + "-0" in normalised:
            # MNE duplicate suffix fallback — take first occurrence
            indices.append(normalised[key + "-0"])
        else:
            missing.append(ch)
    if missing:
        raise ValueError(
            f"Missing canonical channels in EDF: {missing}\n"
            f"Available channels: {list(normalised.keys())}"
        )
    return indices


# ── EDF loader ────────────────────────────────────────────────────────────────

def load_edf(edf_path: str,
             target_channels: List[str] = None,
             expected_fs: int = FS) -> Tuple[np.ndarray, int]:
    """
    Load a single CHB-MIT EDF file and return the 18-channel signal array.

    Parameters
    ----------
    edf_path : str
        Full path to the .edf file.
    target_channels : list[str], optional
        Channel labels to extract (default: CANONICAL_CHANNELS from config).
    expected_fs : int
        Expected sampling frequency; raises if mismatch (default: 256 Hz).

    Returns
    -------
    data : np.ndarray  shape (N_CHANNELS, n_samples)
        Raw EEG signal in Volts (MNE default unit).
    fs   : int
        Actual sampling frequency.
    """
    if target_channels is None:
        target_channels = CANONICAL_CHANNELS

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Verify sampling rate
    fs = int(raw.info["sfreq"])
    if fs != expected_fs:
        raise ValueError(
            f"Unexpected sampling rate in {edf_path}: got {fs} Hz, expected {expected_fs} Hz"
        )

    # Map raw channel names → canonical target channels
    try:
        indices = _build_ch_index(raw.ch_names, target_channels)
    except ValueError as e:
        raise ValueError(f"[{edf_path}] {e}") from e

    # Extract and reorder channels  →  shape (N_CHANNELS, n_samples)
    data = raw.get_data()[indices]   # already in Volts

    return data, fs


def load_patient_files(patient_dir: str,
                       edf_filenames: List[str],
                       target_channels: List[str] = None,
                       verbose: bool = True) -> dict:
    """
    Load multiple EDF files for a single patient.

    Parameters
    ----------
    patient_dir   : str   Path to patient folder.
    edf_filenames : list  EDF file names to load (e.g. ['chb01_03.edf', ...]).
    target_channels : list  Channels to extract.
    verbose       : bool  Print progress.

    Returns
    -------
    loaded : dict  {edf_filename: (data_array, fs)}
                   Failed files are skipped (warning printed).
    """
    if target_channels is None:
        target_channels = CANONICAL_CHANNELS

    patient_dir = Path(patient_dir)
    loaded = {}

    for fname in edf_filenames:
        fpath = patient_dir / fname
        if not fpath.exists():
            print(f"[WARNING] EDF not found: {fpath} — skipping.")
            continue
        try:
            data, fs = load_edf(str(fpath), target_channels)
            loaded[fname] = (data, fs)
            if verbose:
                duration_min = data.shape[1] / fs / 60
                print(f"  Loaded {fname}  ({data.shape[1]} samples, {duration_min:.1f} min)")
        except ValueError as e:
            print(f"[WARNING] Could not load {fname}: {e}")

    return loaded


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from config import DATA_ROOT
    from summary_parser import parse_summary

    test_patient = "chb01"
    pdir = Path(DATA_ROOT) / test_patient
    smap = parse_summary(str(pdir))
    first_file = list(smap.keys())[0]

    print(f"Testing load of {first_file} ...")
    data, fs = load_edf(str(pdir / first_file))
    print(f"Shape: {data.shape}  |  fs: {fs} Hz")
    print(f"Channels: {CANONICAL_CHANNELS}")
    print("OK.")
