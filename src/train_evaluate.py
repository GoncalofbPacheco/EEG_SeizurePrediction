"""
train_evaluate.py  (V3)
=======================
All original V1 functions are preserved unchanged.
New additions:
  run_lopo_v3()   — LOPO with all V2 improvements + alarm post-processing
  run_pooled_v3() — pooled baseline with alarm post-processing

The key addition in run_lopo_v3 is that after model inference we call
evaluate_with_alarms() instead of evaluate_predictions(), applying the
sliding-window vote (FIX 3) and refractory period (FIX 4).
"""

import random
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE,
    RANDOM_SEED, RESULTS_DIR, N_CHANNELS,
    ALARM_K, ALARM_M, ALARM_REFRACTORY,
)
from model   import GCPredictor, GCDataset
from metrics import evaluate_predictions, evaluate_with_alarms


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Loss: Binary Focal Loss (V2 improvement) ─────────────────────────────────

class BinaryFocalLoss(nn.Module):
    """FL(p_t) = −α·(1−p_t)^γ·log(p_t)   with class-frequency α."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, preds, targets):
        p   = preds.squeeze(1).clamp(1e-7, 1 - 1e-7)
        t   = targets.float()
        alpha = torch.where(
            t == 1,
            torch.full_like(t, self.pos_weight),
            torch.ones_like(t),
        )
        bce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
        pt  = torch.where(t == 1, p, 1 - p)
        return (alpha * (1 - pt) ** self.gamma * bce).mean()


# ── Balanced sampler (V2 improvement) ────────────────────────────────────────

def make_balanced_sampler(
    patient_counts:  List[int],
    max_per_patient: int,
) -> WeightedRandomSampler:
    """
    WeightedRandomSampler that caps each patient's contribution to
    `max_per_patient` windows per epoch.
    """
    weights = []
    for n in patient_counts:
        w = min(max_per_patient, n) / n
        weights.extend([w] * n)
    weights   = torch.tensor(weights, dtype=torch.float32)
    n_samples = len(patient_counts) * max_per_patient
    return WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)


# ── V3 training (focal loss + LR decay + balanced sampler) ───────────────────

def train_model_v3(
    train_matrices: np.ndarray,
    train_labels:   np.ndarray,
    val_matrices:   np.ndarray,
    val_labels:     np.ndarray,
    device:         torch.device,
    patient_counts: Optional[List[int]] = None,
    focal_gamma:    float = 2.0,
    lr_decay:       float = 0.90,
    max_win_per_pat: int  = 600,
    verbose:        bool  = True,
) -> GCPredictor:
    """
    Train GCPredictor with V2/V3 improvements:
      - Binary Focal Loss
      - ExponentialLR decay (×0.90 / epoch)
      - Optional per-patient balanced sampler

    Returns model at best validation loss checkpoint.
    """
    set_seed()

    n_pos = max(int((train_labels == 1).sum()), 1)
    n_neg = max(int((train_labels == 0).sum()), 1)
    pos_w = float(min(n_neg / n_pos, 50.0))

    criterion = BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_w).to(device)
    model     = GCPredictor(n_channels=N_CHANNELS).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay)

    train_ds = GCDataset(train_matrices, train_labels)
    val_ds   = GCDataset(val_matrices,   val_labels)

    if patient_counts is not None:
        sampler  = make_balanced_sampler(patient_counts, max_win_per_pat)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val  = float("inf")
    best_wts  = None
    p_ctr     = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(train_ds)

        # Validate
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb   = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item() * len(yb)
        va_loss /= len(val_ds)

        scheduler.step()

        if verbose and epoch % 5 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:3d}/{MAX_EPOCHS}  "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}  lr={lr_now:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_ctr    = 0
        else:
            p_ctr += 1
            if p_ctr >= PATIENCE:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_wts)
    return model


def predict(
    model:    GCPredictor,
    matrices: np.ndarray,
    device:   torch.device,
) -> np.ndarray:
    """Return P(preictal) for each window."""
    model.eval()
    ds     = GCDataset(matrices, np.zeros(len(matrices)))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    probs  = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.append(model(xb.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(probs)


# ── LOPO V3 (all fixes integrated) ───────────────────────────────────────────

def run_lopo_v3(
    patient_data:    Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_fraction:    float = 0.15,
    max_win_per_pat: int   = 600,
    alarm_K:         int   = ALARM_K,
    alarm_M:         int   = ALARM_M,
    alarm_refractory: int  = ALARM_REFRACTORY,
    verbose:         bool  = True,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Leave-One-Patient-Out cross-validation with all V3 fixes.

    Returns
    -------
    window_results : dict  {patient_id: metrics}  — window-level (for AUC)
    alarm_results  : dict  {patient_id: metrics}  — alarm-level (for FPR/h)
    """
    device  = get_device()
    pids    = list(patient_data.keys())
    window_results: Dict[str, dict] = {}
    alarm_results:  Dict[str, dict] = {}

    print(f"\n{'='*62}")
    print(f"LOPO V3  ({len(pids)} folds)  |  device: {device}")
    print(f"  FIX 1: all EDFs loaded   FIX 2: VAR({5}) GC matrices")
    print(f"  FIX 3: alarm voting K={alarm_K}/M={alarm_M}")
    print(f"  FIX 4: refractory {alarm_refractory} windows ({alarm_refractory*10//60} min)")
    print(f"{'='*62}")

    for fold_idx, test_pid in enumerate(pids, 1):
        print(f"\n[Fold {fold_idx}/{len(pids)}]  Test: {test_pid}")

        train_pids = [p for p in pids if p != test_pid]

        # Pool training data; track per-patient counts for balanced sampler
        X_parts, y_parts, raw_counts = [], [], []
        for p in train_pids:
            Xp, yp = patient_data[p]
            X_parts.append(Xp)
            y_parts.append(yp)
            raw_counts.append(len(yp))

        X_pool = np.concatenate(X_parts)
        y_pool = np.concatenate(y_parts)

        # Stratified train / val split (from training pool only)
        idx = np.arange(len(y_pool))
        tr_idx, va_idx = train_test_split(
            idx, test_size=val_fraction,
            random_state=RANDOM_SEED, stratify=y_pool,
        )
        X_tr, y_tr = X_pool[tr_idx], y_pool[tr_idx]
        X_va, y_va = X_pool[va_idx], y_pool[va_idx]
        X_te, y_te = patient_data[test_pid]

        # GC normalisation — fit on TRAIN only
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(X_tr.shape)
        X_va   = scaler.transform(X_va.reshape(len(X_va), -1)).reshape(X_va.shape)
        X_te   = scaler.transform(X_te.reshape(len(X_te), -1)).reshape(X_te.shape)

        # Scale per-patient counts to the train fraction
        train_frac   = len(tr_idx) / len(y_pool)
        train_counts = [max(1, round(c * train_frac)) for c in raw_counts]

        print(f"  Train={len(y_tr)} "
              f"(pre={(y_tr==1).sum()}, int={(y_tr==0).sum()})  "
              f"Val={len(y_va)}  "
              f"Test={len(y_te)} (pre={(y_te==1).sum()})")

        model  = train_model_v3(
            X_tr, y_tr, X_va, y_va, device,
            patient_counts=train_counts,
            max_win_per_pat=max_win_per_pat,
            verbose=verbose,
        )
        probs  = predict(model, X_te, device)

        # Window-level metrics (keeps AUC for comparison)
        w_met = evaluate_predictions(y_te, probs, patient_id=test_pid)
        # Alarm-level metrics (FIX 3 & 4)
        a_met = evaluate_with_alarms(
            y_te, probs,
            K=alarm_K, M=alarm_M, refractory_windows=alarm_refractory,
            patient_id=test_pid,
        )
        window_results[test_pid] = w_met
        alarm_results[test_pid]  = a_met

        print(f"  -> [window] AUC={w_met['auc']:.3f}  "
              f"Sens={w_met['sensitivity']:.3f}  "
              f"Spec={w_met['specificity']:.3f}  "
              f"FPR/h={w_met['fpr_per_hour']:.1f}")
        print(f"     [alarm]  AUC={a_met['auc']:.3f}  "
              f"Sens={a_met['sensitivity']:.3f}  "
              f"Spec={a_met['specificity']:.3f}  "
              f"FPR/h={a_met['fpr_per_hour']:.1f}")

    return window_results, alarm_results


# ── Original V1 functions (preserved unchanged) ───────────────────────────────

def train_model(
    train_matrices: np.ndarray,
    train_labels:   np.ndarray,
    val_matrices:   np.ndarray,
    val_labels:     np.ndarray,
    device:         torch.device,
    verbose:        bool = True,
) -> GCPredictor:
    """Original V1 trainer (weighted BCE, no decay, no sampler)."""
    from model import build_weighted_bce
    set_seed()

    train_ds = GCDataset(train_matrices, train_labels)
    val_ds   = GCDataset(val_matrices,   val_labels)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = GCPredictor(n_channels=N_CHANNELS).to(device)
    criterion = build_weighted_bce(torch.from_numpy(train_labels), device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val  = float("inf")
    best_wts  = copy.deepcopy(model.state_dict())
    p_ctr     = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb   = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item() * len(yb)
        va_loss /= len(val_ds)

        if verbose and epoch % 5 == 0:
            print(f"    Epoch {epoch:3d}/{MAX_EPOCHS}  "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_wts = copy.deepcopy(model.state_dict())
            p_ctr    = 0
        else:
            p_ctr += 1
            if p_ctr >= PATIENCE:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_wts)
    return model


def run_lopo(
    patient_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_fraction: float = 0.15,
    verbose:      bool  = True,
) -> Dict[str, dict]:
    """Original V1 LOPO (window-level evaluation, weighted BCE)."""
    device = get_device()
    pids   = list(patient_data.keys())
    results: Dict[str, dict] = {}

    print(f"\n{'='*60}")
    print(f"LOPO V1  ({len(pids)} folds)  |  device: {device}")
    print(f"{'='*60}")

    for fold_idx, test_pid in enumerate(pids, 1):
        print(f"\n[Fold {fold_idx}/{len(pids)}]  Test: {test_pid}")
        train_pids = [p for p in pids if p != test_pid]
        train_X = np.concatenate([patient_data[p][0] for p in train_pids])
        train_y = np.concatenate([patient_data[p][1] for p in train_pids])
        test_X, test_y = patient_data[test_pid]

        idx = np.arange(len(train_y))
        set_seed()
        np.random.shuffle(idx)
        n_val    = max(1, int(len(train_y) * val_fraction))
        val_idx  = idx[:n_val]
        train_idx = idx[n_val:]

        model = train_model(
            train_X[train_idx], train_y[train_idx],
            train_X[val_idx],   train_y[val_idx],
            device, verbose=verbose,
        )
        probs   = predict(model, test_X, device)
        metrics = evaluate_predictions(test_y, probs, patient_id=test_pid)
        results[test_pid] = metrics
        print(f"  → AUC={metrics['auc']:.3f}  "
              f"Sens={metrics['sensitivity']:.3f}  "
              f"Spec={metrics['specificity']:.3f}  "
              f"F1={metrics['f1']:.3f}")

    return results


def run_pooled(
    patient_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_fraction: float = 0.20,
    val_fraction:  float = 0.15,
    verbose:       bool  = True,
) -> dict:
    """Original V1 pooled evaluation."""
    device = get_device()
    all_X  = np.concatenate([v[0] for v in patient_data.values()])
    all_y  = np.concatenate([v[1] for v in patient_data.values()])
    set_seed()
    idx    = np.arange(len(all_y))
    np.random.shuffle(idx)
    n_test = int(len(idx) * test_fraction)
    n_val  = int((len(idx) - n_test) * val_fraction)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    print(f"\n{'='*60}")
    print(f"Pooled V1  |  device: {device}")
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"{'='*60}")

    model   = train_model(all_X[train_idx], all_y[train_idx],
                          all_X[val_idx],   all_y[val_idx], device, verbose=verbose)
    probs   = predict(model, all_X[test_idx], device)
    metrics = evaluate_predictions(all_y[test_idx], probs, patient_id="pooled")
    print(f"\n  Pooled → AUC={metrics['auc']:.3f}  "
          f"Sens={metrics['sensitivity']:.3f}  "
          f"Spec={metrics['specificity']:.3f}")
    return metrics


def save_results(
    lopo_results:   Dict[str, dict],
    pooled_results: dict,
    results_dir:    str = RESULTS_DIR,
):
    """Save LOPO and pooled metrics to CSV."""
    import pandas as pd
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    lopo_df   = pd.DataFrame(lopo_results).T
    pooled_df = pd.DataFrame([pooled_results])
    lopo_df.to_csv(Path(results_dir) / "lopo_results.csv")
    pooled_df.to_csv(Path(results_dir) / "pooled_results.csv", index=False)
    print(f"\nResults saved → {results_dir}/")
    print(lopo_df[["auc", "sensitivity", "specificity", "f1", "fpr_per_hour"]].to_string())
    print(f"\n  Mean AUC : {lopo_df['auc'].mean():.3f} ± {lopo_df['auc'].std():.3f}")
    return lopo_df, pooled_df
