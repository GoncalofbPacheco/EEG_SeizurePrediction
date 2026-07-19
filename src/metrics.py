"""
metrics.py
==========
Window-level classification metrics and alarm-level post-processing for the
GC-CNN experiments.

Threshold-free metrics (AUC-ROC, AUC-PR) are computed on the raw per-window
probabilities. Operational metrics can be reported either per window
(``evaluate_predictions``) or after alarm post-processing
(``evaluate_with_alarms``), which applies a K-of-M sliding-window vote and a
refractory period so that a run of positive windows counts as a single clinical
alarm rather than one alarm per 10-second window.

For the event-level operational metrics used in the final tables (§Corrected
metrics in the README), see ``seizure_metrics.py``.
"""

from typing import Optional, Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)

from config import STEP_SEC, ALARM_K, ALARM_M, ALARM_REFRACTORY


# ─────────────────────────────────────────────────────────────────────────────
# Primitive metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _threshold_preds(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(int)


def compute_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN)"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TN / (TN + FP)"""
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP)"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prec = compute_precision(y_true, y_pred)
    sens = compute_sensitivity(y_true, y_pred)
    denom = prec + sens
    return 2 * prec * sens / denom if denom > 0 else 0.0


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (compute_sensitivity(y_true, y_pred) +
            compute_specificity(y_true, y_pred)) / 2.0


def compute_fpr_per_hour(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    total_hours: float,
) -> float:
    """FP / total_interictal_hours"""
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return float(fp) / total_hours if total_hours > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Window-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true:      np.ndarray,
    probs:       np.ndarray,
    threshold:   float = 0.5,
    total_hours: Optional[float] = None,
    patient_id:  str = "",
) -> Dict[str, float]:
    """
    Window-level metrics.
    Use evaluate_with_alarms() for the alarm-level post-processed version.
    """
    y_pred = _threshold_preds(probs, threshold)

    if len(np.unique(y_true)) < 2:
        auc    = float("nan")
        auc_pr = float("nan")
    else:
        auc    = roc_auc_score(y_true, probs)
        auc_pr = average_precision_score(y_true, probs)

    sens    = compute_sensitivity(y_true, y_pred)
    spec    = compute_specificity(y_true, y_pred)
    prec    = compute_precision(y_true, y_pred)
    f1      = compute_f1(y_true, y_pred)
    bal_acc = compute_balanced_accuracy(y_true, y_pred)

    if total_hours is None:
        n_interictal = (y_true == 0).sum()
        total_hours  = n_interictal * STEP_SEC / 3600.0

    fpr_h = compute_fpr_per_hour(y_true, y_pred, total_hours)

    return {
        "patient_id":        patient_id,
        "auc":               round(auc,      4),
        "auc_pr":            round(auc_pr,   4),
        "sensitivity":       round(sens,     4),
        "specificity":       round(spec,     4),
        "precision":         round(prec,     4),
        "f1":                round(f1,       4),
        "balanced_accuracy": round(bal_acc,  4),
        "fpr_per_hour":      round(fpr_h,    4),
        "threshold":         threshold,
        "n_preictal":        int((y_true == 1).sum()),
        "n_interictal":      int((y_true == 0).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Alarm-level post-processing
# ─────────────────────────────────────────────────────────────────────────────

def apply_alarm_postprocessing(
    probs:               np.ndarray,
    threshold:           float = 0.5,
    K:                   int   = ALARM_K,
    M:                   int   = ALARM_M,
    refractory_windows:  int   = ALARM_REFRACTORY,
) -> np.ndarray:
    """
    Convert window-level probabilities to alarm-level binary predictions.

    Algorithm
    ---------
    1. Threshold probs to get raw per-window predictions.
    2. For each window i, look at the last M windows [i-M+1 … i].
       If ≥ K are predicted positive, fire an alarm at i.
    3. After firing, suppress alarm evaluation for the next
       `refractory_windows` steps (refractory period).

    Parameters
    ----------
    probs               : np.ndarray  shape (N,)  model output P(preictal)
    threshold           : float       decision threshold (default 0.5)
    K                   : int         min positives in window to fire alarm
    M                   : int         sliding-window size (in windows)
    refractory_windows  : int         suppression length after alarm fires

    Returns
    -------
    alarm_pred : np.ndarray  shape (N,)  int  {0, 1}
    """
    N        = len(probs)
    raw_pred = (probs >= threshold).astype(int)
    alarm_pred = np.zeros(N, dtype=int)

    refractory_count = 0
    for i in range(N):
        if refractory_count > 0:
            refractory_count -= 1
            continue
        start = max(0, i - M + 1)
        if raw_pred[start : i + 1].sum() >= K:
            alarm_pred[i]    = 1
            refractory_count = refractory_windows  # suppress next 30 min

    return alarm_pred


def evaluate_with_alarms(
    y_true:              np.ndarray,
    probs:               np.ndarray,
    threshold:           float         = 0.5,
    K:                   int           = ALARM_K,
    M:                   int           = ALARM_M,
    refractory_windows:  int           = ALARM_REFRACTORY,
    total_hours:         Optional[float] = None,
    patient_id:          str           = "",
) -> Dict[str, float]:
    """
    Full evaluation with alarm-level post-processing.

    AUC-ROC and AUC-PR are computed at window level (threshold-independent).
    All other metrics (Sensitivity, Specificity, F1, FPR/h) use alarm_pred.

    Parameters
    ----------
    y_true             : ground truth labels {0, 1}
    probs              : P(preictal) from model
    threshold          : decision threshold
    K, M               : voting parameters  (see apply_alarm_postprocessing)
    refractory_windows : refractory period length in windows
    total_hours        : interictal evaluation time; estimated if None
    patient_id         : for logging

    Returns
    -------
    metrics : dict  — same keys as evaluate_predictions() plus alarm parameters
    """
    # Window-level AUC (not affected by alarm logic)
    if len(np.unique(y_true)) < 2:
        auc    = float("nan")
        auc_pr = float("nan")
    else:
        auc    = roc_auc_score(y_true, probs)
        auc_pr = average_precision_score(y_true, probs)

    alarm_pred = apply_alarm_postprocessing(
        probs, threshold, K, M, refractory_windows
    )

    sens    = compute_sensitivity(y_true, alarm_pred)
    spec    = compute_specificity(y_true, alarm_pred)
    prec    = compute_precision(y_true, alarm_pred)
    f1      = compute_f1(y_true, alarm_pred)
    bal_acc = compute_balanced_accuracy(y_true, alarm_pred)

    if total_hours is None:
        n_interictal = (y_true == 0).sum()
        total_hours  = n_interictal * STEP_SEC / 3600.0

    fpr_h = compute_fpr_per_hour(y_true, alarm_pred, total_hours)

    return {
        "patient_id":        patient_id,
        "auc":               round(auc,      4),   # window-level
        "auc_pr":            round(auc_pr,   4),   # window-level
        "sensitivity":       round(sens,     4),   # alarm-level
        "specificity":       round(spec,     4),   # alarm-level
        "precision":         round(prec,     4),   # alarm-level
        "f1":                round(f1,       4),   # alarm-level
        "balanced_accuracy": round(bal_acc,  4),   # alarm-level
        "fpr_per_hour":      round(fpr_h,    4),   # alarm-level
        "threshold":         threshold,
        "alarm_K":           K,
        "alarm_M":           M,
        "refractory_windows": refractory_windows,
        "n_preictal":        int((y_true == 1).sum()),
        "n_interictal":      int((y_true == 0).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true:    np.ndarray,
    probs:     np.ndarray,
    criterion: str = "youden",
) -> float:
    """
    Find the optimal decision threshold.
    criterion : 'youden' → max (Sens + Spec - 1)
                'f1'     → max F1
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, probs)
    if criterion == "youden":
        idx = np.argmax(tpr_arr - fpr_arr)
    elif criterion == "f1":
        f1s = [compute_f1(y_true, (probs >= t).astype(int)) for t in thresholds]
        idx = int(np.argmax(f1s))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    return float(thresholds[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, alarm_level: bool = False):
    """Pretty-print a metrics dictionary."""
    pid    = metrics.get("patient_id", "")
    mode   = "ALARM" if alarm_level else "WINDOW"
    header = f"── Metrics [{pid}] [{mode}] "
    print(f"\n{header}{'─'*(55 - len(header))}")
    print(f"  AUC-ROC          : {metrics['auc']:.4f}  (window-level)")
    print(f"  AUC-PR           : {metrics['auc_pr']:.4f}  (window-level)")
    print(f"  Sensitivity      : {metrics['sensitivity']:.4f}")
    print(f"  Specificity      : {metrics['specificity']:.4f}")
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  F1-score         : {metrics['f1']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  FPR / hour       : {metrics['fpr_per_hour']:.4f}")
    if alarm_level:
        print(f"  Alarm K={metrics.get('alarm_K')}  M={metrics.get('alarm_M')}  "
              f"Refr={metrics.get('refractory_windows')}")
    print(f"  n_preictal       : {metrics['n_preictal']}")
    print(f"  n_interictal     : {metrics['n_interictal']}")
