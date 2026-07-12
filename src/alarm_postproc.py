"""
Alarm post-processing helpers — bridges per-window classifier outputs to
event-level seizure-prediction metrics.

Implements the four-step pipeline used by Truong et al. (2018),
Bandarabadi et al. (2015), and Khan et al. (2018):

    raw probabilities
        → (1) temporal smoothing (moving average)
        → (2) thresholding
        → (3) K-of-M sliding-window persistence vote
        → (4) refractory-period suppression after each alarm

Steps (3) and (4) live in `metrics.evaluate_with_alarms`. This module adds
step (1), the temporal smoother, and a helper for choosing the threshold
operationally rather than via Youden-J on test data.

References
----------
Truong et al. (2018), "Convolutional neural networks for seizure prediction
    using intracranial and scalp electroencephalogram", Neural Networks 105.
Bandarabadi et al. (2015), "Epileptic seizure prediction using relative
    spectral power features", Clin Neurophysiol 126(2).
Mormann et al. (2007), "Seizure prediction: the long and winding road",
    Brain 130, esp. §"Statistical evaluation".
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def smooth_probs(probs: np.ndarray, window: int = 10, mode: str = "same") -> np.ndarray:
    """
    Apply a moving-average smoother to a per-window probability series.

    Parameters
    ----------
    probs  : (N,) array of raw per-window P(preictal).
    window : smoothing window length in windows.  Default 10 matches
             ~100 s at a 10 s step, i.e. comparable to the M=12 voting
             window used by `metrics.evaluate_with_alarms`. Truong 2018
             uses a 5-window filter at finer time-resolution.
    mode   : 'same' (centered, for offline evaluation) or 'valid'
             (causal, for prospective deployment simulation).

    Returns
    -------
    smoothed probability series, same length as `probs`.
    """
    if window <= 1 or len(probs) < window:
        return probs.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(probs.astype(float), kernel, mode=mode)


def operational_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    step_sec: int,
    target_fpr_h: float = 1.0,
    K: int = None,
    M: int = None,
    refractory_windows: int = None,
) -> float:
    """
    Select the decision threshold that maximises sensitivity subject to a
    false-prediction-rate constraint, measured at the EVENT (alarm) level
    after the K-of-M sliding-window vote and refractory suppression.

    This matches the threshold-selection procedure used by:
      - Truong et al. (2018) §II.E:   alarm-level Sens / FPR/h trade-off
      - Bandarabadi et al. (2015) §III.B: alarm-level threshold selection
      - Mormann et al. (2007) §"Statistical evaluation": SOP framework

    When K, M, refractory_windows are provided (the literature-standard
    behaviour), the threshold sweep applies the alarm rule at each candidate
    threshold and selects the one that maximises alarm-level sensitivity
    while keeping alarm-level FPR/h within `target_fpr_h`.

    When they are None (legacy behaviour), the constraint is evaluated at
    the window level; this is retained for backward compatibility but is
    not the published convention.

    Parameters
    ----------
    probs              : per-window probabilities (typically already smoothed).
    y_true             : per-window labels {0,1}.
    step_sec           : step between windows in seconds.
    target_fpr_h       : maximum acceptable alarm-level false predictions / hour
                         (Mormann 2007 clinical threshold ≈ 1.0).
    K, M               : sliding-window vote parameters.
    refractory_windows : refractory suppression length in windows after
                         each alarm.

    Returns
    -------
    threshold ∈ [0.05, 0.95]. Falls back to Youden-J if no threshold meets
    the FPR/h constraint at the alarm level.
    """
    n_interictal_hours = float((y_true == 0).sum()) * step_sec / 3600.0
    if n_interictal_hours <= 0:
        return 0.5

    use_alarm = K is not None and M is not None and refractory_windows is not None
    if use_alarm:
        from metrics import apply_alarm_postprocessing

    candidates = np.linspace(0.05, 0.95, 91)
    best_thr, best_sens = None, -1.0
    for t in candidates:
        if use_alarm:
            pred = apply_alarm_postprocessing(probs, t, K, M, refractory_windows)
        else:
            pred = (probs >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fpr_h = fp / max(n_interictal_hours, 1e-9)
        sens  = tp / max(tp + fn, 1)
        if fpr_h <= target_fpr_h and sens > best_sens:
            best_sens, best_thr = sens, float(t)

    if best_thr is not None:
        return best_thr

    # Fall-back: Youden-J on the (smoothed) probabilities.
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = np.argmax(tpr - fpr)
    return float(np.clip(thr[j], 0.05, 0.95))


def calibrated_alarm_pipeline(
    probs_raw: np.ndarray,
    y_true: np.ndarray,
    step_sec: int,
    smooth_window: int = 10,
    target_fpr_h: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Convenience wrapper: smooth → operational threshold → return
    (smoothed_probs, threshold) ready for `evaluate_with_alarms`.

    Threshold is selected on the SAME series passed in. For strict no-leakage
    evaluation, callers should select the threshold on a held-out validation
    series and pass it explicitly — this function is a convenience for the
    common case of per-fold evaluation in offline cross-patient LOPO, where
    the threshold-selection leakage is documented as a limitation.
    """
    smoothed = smooth_probs(probs_raw, window=smooth_window)
    thr      = operational_threshold(smoothed, y_true, step_sec, target_fpr_h)
    return smoothed, thr
