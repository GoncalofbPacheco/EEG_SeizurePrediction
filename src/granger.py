"""
granger.py  (V3 — FIX 2: VAR(p) instead of VAR(1))
=====================================================
Original VAR(1) problem
------------------------
The original code estimated A = Σ₁ · Σ₀⁻¹ — a one-lag approximation.
At 256 Hz this captures only ~4 ms of history, which is shorter than a single
EEG oscillation cycle (e.g. alpha at 10 Hz has a 100 ms period).
The resulting 18×18 matrix carries almost no frequency-domain information
about preictal activity, explaining why AUC ≈ 0.5.

VAR(p) fix
----------
We now fit a proper order-p VAR model via OLS:

    X(t) = A₁·X(t-1) + A₂·X(t-2) + ... + Aₚ·X(t-p) + ε(t)

With GC_ORDER = 5 the model captures up to 5 × (1/256 s) ≈ 20 ms of
directed influence — enough to encode alpha/theta band coupling.

Output GC matrix
-----------------
We output the element-wise sum of absolute lag matrices:

    A_total = |A₁| + |A₂| + ... + |Aₚ|   (shape 18×18)

This preserves backward compatibility with the CNN (still one 18×18 channel)
while encoding the total directed influence across all lags.

Caching
-------
V3 uses GC_MATRICES_DIR_V3 = "cache_gc_var5" so old VAR(1) caches are
never accidentally loaded.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from config import N_CHANNELS, GC_MATRICES_DIR_V3, GC_ORDER


# ── Core VAR(p) estimation ────────────────────────────────────────────────────

def estimate_gc_matrix(
    window: np.ndarray,
    p:      int   = GC_ORDER,
    eps:    float = 1e-10,
) -> Tuple[np.ndarray, bool]:
    """
    Estimate a VAR(p) connectivity matrix for a single EEG window via OLS.

    Model:  X(t) = A₁·X(t-1) + ... + Aₚ·X(t-p) + ε(t)

    OLS solution:
        B = (Y · Zᵀ) · (Z · Zᵀ)⁻¹
    where
        Y = X[:, p:]          — targets,   shape (n_ch, T_eff)
        Z = stack of p lags   — regressors, shape (p·n_ch, T_eff)
        B = [A₁ | A₂ | … | Aₚ], shape (n_ch, p·n_ch)

    GC output:  A_total = Σᵢ |Aᵢ|   (18×18, float32)

    Parameters
    ----------
    window : np.ndarray  shape (n_channels, n_samples)
    p      : int         VAR model order (default GC_ORDER from config)
    eps    : float       diagonal regularisation added to Z·Zᵀ

    Returns
    -------
    A_total : np.ndarray  shape (n_channels, n_channels)
    valid   : bool        False when OLS cannot be solved (rank-deficient)
    """
    n_ch, T = window.shape

    # De-mean per channel
    X = window - window.mean(axis=1, keepdims=True)

    T_eff = T - p
    # Sanity: need more observations than parameters
    if T_eff < n_ch * p + 1:
        return np.zeros((n_ch, n_ch), dtype=np.float32), False

    # ── Build target matrix Y  (n_ch, T_eff) ─────────────────────────────────
    # Y[:, i] = X[:, p+i]   for i = 0 … T_eff-1
    Y = X[:, p:]  # (n_ch, T_eff)

    # ── Build regressor matrix Z  (p*n_ch, T_eff) ────────────────────────────
    # Lag l row-block: X[:, p-l : p-l+T_eff]  for l = 1 … p
    Z = np.vstack([X[:, p - lag : p - lag + T_eff] for lag in range(1, p + 1)])
    # shape: (p*n_ch, T_eff)

    # ── OLS: B = (Y·Zᵀ / T_eff) · inv(Z·Zᵀ / T_eff) ─────────────────────────
    ZZT = (Z @ Z.T) / T_eff                   # (p*n_ch, p*n_ch)
    ZZT += eps * np.eye(p * n_ch)             # regularise

    rank = np.linalg.matrix_rank(ZZT)
    if rank < p * n_ch:
        return np.zeros((n_ch, n_ch), dtype=np.float32), False

    YZT = (Y @ Z.T) / T_eff                   # (n_ch, p*n_ch)
    B   = YZT @ np.linalg.inv(ZZT)            # (n_ch, p*n_ch)

    # ── GC matrix: sum of |Aᵢ| across all lags ───────────────────────────────
    A_total = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(p):
        A_total += np.abs(B[:, i * n_ch : (i + 1) * n_ch])

    return A_total.astype(np.float32), True


def compute_gc_matrices(
    windows: np.ndarray,
    p:       int  = GC_ORDER,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute VAR(p) GC matrices for an array of windows.

    Parameters
    ----------
    windows : np.ndarray  shape (n_windows, n_channels, n_samples)
    p       : int         VAR order
    verbose : bool

    Returns
    -------
    gc_matrices : np.ndarray  shape (n_windows, n_channels, n_channels)
    valid_mask  : np.ndarray  shape (n_windows,)  bool
    """
    n_windows = windows.shape[0]
    gc_matrices = np.zeros((n_windows, N_CHANNELS, N_CHANNELS), dtype=np.float32)
    valid_mask  = np.ones(n_windows, dtype=bool)

    n_invalid = 0
    for i, win in enumerate(windows):
        A, valid = estimate_gc_matrix(win, p=p)
        gc_matrices[i] = A
        if not valid:
            valid_mask[i] = False
            n_invalid += 1

    if verbose and n_invalid > 0:
        print(f"  [GC] {n_invalid}/{n_windows} rank-deficient windows → zeroed")

    return gc_matrices, valid_mask


# ── Disk caching (V3 uses cache_gc_var5/) ─────────────────────────────────────

def _cache_path(
    patient_id: str,
    fname:      str,
    cache_dir:  str = GC_MATRICES_DIR_V3,
) -> Path:
    stem = fname.replace(".edf", "")
    return Path(cache_dir) / patient_id / f"{stem}_gc.npy"


def _labels_cache_path(
    patient_id: str,
    fname:      str,
    cache_dir:  str = GC_MATRICES_DIR_V3,
) -> Path:
    stem = fname.replace(".edf", "")
    return Path(cache_dir) / patient_id / f"{stem}_labels.npy"


def cache_exists(
    patient_id: str,
    fname:      str,
    cache_dir:  str = GC_MATRICES_DIR_V3,
) -> bool:
    return (
        _cache_path(patient_id, fname, cache_dir).exists()
        and _labels_cache_path(patient_id, fname, cache_dir).exists()
    )


def save_gc_cache(
    gc_matrices: np.ndarray,
    labels:      np.ndarray,
    patient_id:  str,
    fname:       str,
    cache_dir:   str = GC_MATRICES_DIR_V3,
) -> None:
    cp = _cache_path(patient_id, fname, cache_dir)
    lp = _labels_cache_path(patient_id, fname, cache_dir)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cp), gc_matrices)
    np.save(str(lp), labels)


def load_gc_cache(
    patient_id: str,
    fname:      str,
    cache_dir:  str = GC_MATRICES_DIR_V3,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    cp = _cache_path(patient_id, fname, cache_dir)
    lp = _labels_cache_path(patient_id, fname, cache_dir)
    if cp.exists() and lp.exists():
        return np.load(str(cp)), np.load(str(lp))
    return None


def process_file_to_gc(
    windows:   np.ndarray,
    labels:    np.ndarray,
    patient_id: str  = "",
    fname:      str  = "",
    use_cache:  bool = True,
    cache_dir:  str  = GC_MATRICES_DIR_V3,
    verbose:    bool = True,
    p:          int  = GC_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (or load from cache) GC matrices for one EDF file's windows.

    Rank-deficient windows are discarded.

    Returns
    -------
    gc_matrices : np.ndarray  shape (n_valid, 18, 18)
    labels      : np.ndarray  shape (n_valid,)
    """
    if use_cache and fname and cache_exists(patient_id, fname, cache_dir):
        if verbose:
            print(f"  [GC] Cache hit: {patient_id}/{fname}")
        return load_gc_cache(patient_id, fname, cache_dir)

    if verbose:
        print(f"  [GC] Computing {windows.shape[0]} VAR({p}) matrices  "
              f"{patient_id}/{fname} ...")

    gc_matrices, valid_mask = compute_gc_matrices(windows, p=p, verbose=verbose)

    # Discard rank-deficient
    gc_matrices = gc_matrices[valid_mask]
    labels      = labels[valid_mask]

    if use_cache and fname:
        save_gc_cache(gc_matrices, labels, patient_id, fname, cache_dir)

    return gc_matrices, labels
