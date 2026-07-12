"""
seizure_metrics.py  — CORRECTED event-level seizure-prediction metrics
======================================================================

This module fixes the single most important coding error in the thesis:
the False-Prediction-Rate per hour (FPR/h) — and, relatedly, the alarm
sensitivity — were reported at the WINDOW level instead of the EVENT
(alarm) level.

WHY THE OLD NUMBERS WERE WRONG
------------------------------
With 20-second windows and 50% overlap the step is 10 s, i.e. 360 windows
per hour. The old code counted *every* false-positive 10-second window as a
separate "false prediction":

    fp = ((y_pred == 1) & (y_true == 0)).sum()      # window-level FP count
    fpr_h = fp / interictal_hours

Because the classifiers sit near AUC 0.5, roughly 30-50% of interictal
windows cross the threshold, so this produced FPR/h values of 100-360 —
i.e. "more than one false alarm every minute". That is physically
meaningless for a seizure-warning device and is exactly what the supervisor
flagged ("You can have 100+ false alarms per hour!! ... maybe it is a
calculation problem"). It IS a calculation problem.

THE CORRECT DEFINITION (Truong et al. 2018; Mormann et al. 2007)
----------------------------------------------------------------
A seizure-prediction system raises discrete ALARMS, not per-window flags.
The pipeline is:

    raw P(preictal) per window
        -> threshold                       (per-window 0/1)
        -> K-of-M persistence vote         (an alarm only if >=K of last M win)
        -> refractory suppression          (after an alarm, mute R windows)
        -> list of ALARM TIMES

From the alarm times, the two clinically meaningful numbers are:

  * Event Sensitivity  = (# seizures for which >=1 alarm fell inside that
                          seizure's preictal window) / (total # seizures)
                         -- "what fraction of seizures did we warn about?"

  * FPR/h              = (# alarms that fell in interictal time) /
                          (total interictal hours)
                         -- "how many false alarms per hour of normal EEG?"

With a 30-minute (180-window) refractory period the maximum possible FPR/h
is ~2, so a correct implementation can never report 100+.

The threshold-free metrics AUC and AUC-PR are unchanged; they are computed
on the raw per-window probabilities and are the primary evidence in the
thesis. Only the operational Sensitivity/FPR/h pair changes.

This module supersedes `metrics.compute_fpr_per_hour`,
`metrics.evaluate_with_alarms` (window-denominator sensitivity) and the
`fpr_h_window`/`sens_window` columns that leaked into Tables 9 and 10.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "generate_alarms",
    "event_sensitivity",
    "false_alarms_per_hour",
    "AlarmMetrics",
    "evaluate_event_level",
]


def generate_alarms(
    probs: np.ndarray,
    threshold: float,
    K: int,
    M: int,
    refractory_windows: int,
) -> np.ndarray:
    """Return a binary array marking the windows at which an alarm FIRES.

    An alarm fires at window ``i`` when at least ``K`` of the last ``M``
    windows (inclusive) are above ``threshold`` AND the system is not inside
    the refractory period of a previous alarm. After an alarm fires, the next
    ``refractory_windows`` windows cannot fire.

    This is the standard K-of-M persistence + refractory rule of
    Truong et al. (2018). Note that because of the refractory period each
    fired ``1`` corresponds to one *distinct* clinical alarm event.
    """
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    raw = (probs >= threshold).astype(int)
    alarms = np.zeros(n, dtype=int)
    refractory = 0
    for i in range(n):
        if refractory > 0:
            refractory -= 1
            continue
        start = max(0, i - M + 1)
        if raw[start : i + 1].sum() >= K:
            alarms[i] = 1
            refractory = refractory_windows
    return alarms


def event_sensitivity(
    alarms: np.ndarray,
    seizure_window_ids: Sequence[Sequence[int]],
) -> float:
    """Fraction of seizures anticipated by at least one alarm.

    Parameters
    ----------
    alarms : (N,) binary array from :func:`generate_alarms`.
    seizure_window_ids : list with one entry per seizure; each entry is the
        set/list of window indices that make up that seizure's preictal
        window (i.e. the positive windows belonging to that seizure).

    Returns
    -------
    Event-level sensitivity in [0, 1]. Returns nan if there are no seizures.
    """
    if len(seizure_window_ids) == 0:
        return float("nan")
    hit = 0
    for idxs in seizure_window_ids:
        idxs = np.asarray(list(idxs), dtype=int)
        if idxs.size and alarms[idxs].any():
            hit += 1
    return hit / len(seizure_window_ids)


def false_alarms_per_hour(
    alarms: np.ndarray,
    y_true: np.ndarray,
    step_sec: float,
) -> float:
    """Number of alarms fired in interictal time, per interictal hour.

    An alarm is "false" when it fires on an interictal (``y_true == 0``)
    window. The denominator is the total interictal duration in hours,
    ``(#interictal windows) * step_sec / 3600``.
    """
    alarms = np.asarray(alarms)
    y_true = np.asarray(y_true)
    interictal_hours = float((y_true == 0).sum()) * step_sec / 3600.0
    if interictal_hours <= 0:
        return float("nan")
    false_alarms = int(((alarms == 1) & (y_true == 0)).sum())
    return false_alarms / interictal_hours


@dataclass
class AlarmMetrics:
    auc: float
    auc_pr: float
    event_sensitivity: float
    fpr_per_hour: float
    n_alarms: int
    n_false_alarms: int
    n_seizures: int
    threshold: float
    K: int
    M: int
    refractory_windows: int


def evaluate_event_level(
    y_true: np.ndarray,
    probs: np.ndarray,
    seizure_window_ids: Sequence[Sequence[int]],
    step_sec: float,
    threshold: float = 0.5,
    K: int = 5,
    M: int = 12,
    refractory_windows: int = 180,
) -> AlarmMetrics:
    """Full corrected evaluation: threshold-free AUC/AUC-PR at window level,
    Sensitivity and FPR/h at the EVENT level.

    ``seizure_window_ids`` groups the positive windows by seizure so that
    event sensitivity can be computed. If you only have flat per-window
    labels, use :func:`infer_seizure_groups` to split the positive windows
    into contiguous runs (one run == one seizure's preictal window).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        auc_pr = float("nan")
    else:
        auc = roc_auc_score(y_true, probs)
        auc_pr = average_precision_score(y_true, probs)

    alarms = generate_alarms(probs, threshold, K, M, refractory_windows)
    sens = event_sensitivity(alarms, seizure_window_ids)
    fpr_h = false_alarms_per_hour(alarms, y_true, step_sec)

    return AlarmMetrics(
        auc=auc,
        auc_pr=auc_pr,
        event_sensitivity=sens,
        fpr_per_hour=fpr_h,
        n_alarms=int(alarms.sum()),
        n_false_alarms=int(((alarms == 1) & (y_true == 0)).sum()),
        n_seizures=len(seizure_window_ids),
        threshold=threshold,
        K=K,
        M=M,
        refractory_windows=refractory_windows,
    )


def infer_seizure_groups(y_true: np.ndarray) -> list[list[int]]:
    """Split the positive windows of a per-patient timeline into contiguous
    runs. Each maximal run of consecutive ``y_true == 1`` windows is treated
    as one seizure's preictal window.

    This assumes the windows are supplied in chronological order and that the
    30-minute post-ictal exclusion separates successive seizures (true in the
    CHB-MIT preprocessing used here). If seizures were pre-grouped upstream,
    pass the explicit groups to :func:`evaluate_event_level` instead.
    """
    y_true = np.asarray(y_true)
    groups: list[list[int]] = []
    current: list[int] = []
    for i, v in enumerate(y_true):
        if v == 1:
            current.append(i)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups
