"""
utils_chronological_split.py
============================
Chronological (position-based) train/val split for LOPO training, in place of a
random split.

Why this matters
----------------
With 50 %-overlap 20-second windows, consecutive windows are correlated:

    window i   covers [t,      t+20s]
    window i+1 covers [t+10s,  t+30s]

A random split places windows i and i+1 on opposite sides of the train/val
boundary half the time, so the model sees nearly-identical samples in both sets.
This inflates validation performance — corrupting early stopping and
hyper-parameter selection — without affecting LOPO test performance, so a random
split can show a plausible validation loss even when LOPO AUC stays near chance.
As a time series, the validation subset must respect chronological order.

What this does
--------------
For each training-pool patient, split that patient's chronologically-ordered
window array into [train | val] by position, not randomly. The patient_data
arrays produced by the cache loader are already in file-load order
(`sorted(glob)`), which is approximately chronological within a patient
(CHB-MIT file numbers increase monotonically with recording session).

Usage
-----
    from utils_chronological_split import lopo_train_val_split

    X_tr, y_tr, X_va, y_va, train_patient_counts = lopo_train_val_split(
        patient_data,
        test_pid='chb01',
        val_fraction=0.15,
        stratify_within_patient=True,
    )

`stratify_within_patient=True` (default) splits preictal and interictal
windows separately within each patient so both train and val see both
classes, but still respects temporal order within each class.
"""

from typing import Dict, List, Tuple
import numpy as np


def _chronological_split_one_patient(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split one patient's (X, y) chronologically into (X_tr, y_tr, X_va, y_va).

    If stratify is True, splits preictal and interictal class-wise so both
    train and val contain both classes (still chronological within each class).
    """
    n = len(y)
    if n == 0:
        empty = np.empty((0, *X.shape[1:]), dtype=X.dtype)
        return empty, np.empty(0, dtype=y.dtype), empty, np.empty(0, dtype=y.dtype)

    if not stratify:
        cut = int(np.ceil(n * (1 - val_fraction)))
        return X[:cut], y[:cut], X[cut:], y[cut:]

    # Class-wise chronological split
    tr_idx_parts, va_idx_parts = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]  # already in chronological order
        if len(cls_idx) == 0:
            continue
        cut = int(np.ceil(len(cls_idx) * (1 - val_fraction)))
        tr_idx_parts.append(cls_idx[:cut])
        va_idx_parts.append(cls_idx[cut:])

    tr_idx = np.sort(np.concatenate(tr_idx_parts)) if tr_idx_parts else np.array([], dtype=int)
    va_idx = np.sort(np.concatenate(va_idx_parts)) if va_idx_parts else np.array([], dtype=int)

    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


def lopo_train_val_split(
    patient_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_pid: str,
    val_fraction: float = 0.15,
    stratify_within_patient: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Build a LOPO fold with a chronological within-patient validation split.

    Parameters
    ----------
    patient_data : dict   {pid: (X, y)} — X must be chronologically ordered
                          within each patient (the cache loader already does this)
    test_pid     : str    patient to hold out for testing
    val_fraction : float  fraction of each training patient to set aside for val
    stratify_within_patient : bool
                          split preictal and interictal class-wise so both
                          train and val contain both classes
    verbose      : bool   print fold composition

    Returns
    -------
    X_tr, y_tr   : training arrays (concatenated across training patients)
    X_va, y_va   : validation arrays (concatenated across training patients)
    train_patient_counts : list of training-set sizes per patient (in concat
                          order). Used by `make_balanced_sampler` so each
                          training patient contributes equally per epoch.
    """
    X_tr_parts, y_tr_parts = [], []
    X_va_parts, y_va_parts = [], []
    train_counts = []

    train_pids = [pid for pid in patient_data if pid != test_pid]

    for pid in train_pids:
        X_p, y_p = patient_data[pid]
        X_tr_p, y_tr_p, X_va_p, y_va_p = _chronological_split_one_patient(
            X_p, y_p, val_fraction, stratify_within_patient
        )
        X_tr_parts.append(X_tr_p)
        y_tr_parts.append(y_tr_p)
        X_va_parts.append(X_va_p)
        y_va_parts.append(y_va_p)
        train_counts.append(len(y_tr_p))

        if verbose:
            print(f"  {pid}: train={len(y_tr_p):5d} "
                  f"(pre={int((y_tr_p==1).sum())}, int={int((y_tr_p==0).sum())})  "
                  f"val={len(y_va_p):4d} "
                  f"(pre={int((y_va_p==1).sum())}, int={int((y_va_p==0).sum())})")

    X_tr = np.concatenate(X_tr_parts, axis=0)
    y_tr = np.concatenate(y_tr_parts, axis=0)
    X_va = np.concatenate(X_va_parts, axis=0)
    y_va = np.concatenate(y_va_parts, axis=0)

    if verbose:
        print(f"\n  Total train: {len(y_tr):,}  "
              f"(pre={int((y_tr==1).sum())}, int={int((y_tr==0).sum())})")
        print(f"  Total val  : {len(y_va):,}  "
              f"(pre={int((y_va==1).sum())}, int={int((y_va==0).sum())})")

    return X_tr, y_tr, X_va, y_va, train_counts


def patient_chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.15,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Patient-specific chronological split (for within-patient experiments only).
    Use this for the patient-specific configuration, not for LOPO.
    """
    return _chronological_split_one_patient(X, y, val_fraction, stratify)


# ── Sanity test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic data: 3 patients, 1000 windows each, 20% preictal
    rng = np.random.default_rng(0)
    pd = {}
    for i, pid in enumerate(["p0", "p1", "p2"]):
        n = 1000
        # Chronologically ordered: preictal in first half, interictal in second
        # (typical in CHB-MIT seizure files)
        y = np.zeros(n, dtype=np.int8)
        y[:200] = 1
        X = rng.standard_normal((n, 18, 18)).astype(np.float32) + i
        pd[pid] = (X, y)

    X_tr, y_tr, X_va, y_va, counts = lopo_train_val_split(
        pd, test_pid="p0", val_fraction=0.15, verbose=True
    )
    assert len(X_tr) == sum(counts)
    assert len(X_va) > 0
    print("\nChronological split sanity test passed.")
