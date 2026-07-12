import os as _os, sys as _sys
from pathlib import Path as _Path
_FC_REPO = str(_Path(__file__).resolve().parent.parent)  # repo root (scripts/ lives under it)
_sys.path.insert(0, _os.path.join(_FC_REPO, "src"))

"""
generate_v7_v8.py
-----------------
Generates Main_V7_PreictalWindow.ipynb and Main_V8_WithinPatient.ipynb.
Run once: python3 generate_v7_v8.py
"""
import json, os

CODE_DIR = _FC_REPO

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def md(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

def code(cid, src):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src}

# ═══════════════════════════════════════════════════════════════════
# V7 — Preictal Window Sensitivity
# ═══════════════════════════════════════════════════════════════════

v7_cells = [

md("v7_m0", """\
# Main V7 — Preictal Window Sensitivity

## Motivation

V3–V6 use a **30-minute preictal window** (minus 5-min SPH = 25 min effective).
Some literature papers argue shorter windows (5–15 min) better capture the
preictal state and explain performance gaps. V7 tests whether preictal window
length drives the cross-patient LOPO results.

**Experimental design**: broadband GC features (same as V4, VAR(5) + 67 graph
descriptors) with PREICTAL_SEC ∈ {10, 15, 30} min. SPH fixed at 5 min throughout.
The 30-min result replicates V4 on the same feature set for a clean comparison.

**Key question**: Does a shorter preictal window improve cross-patient skill score?
"""),

code("v7_c0", """\
# Cell 0 — Imports & config
import os, sys, json, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

CODE_DIR = _FC_REPO
sys.path.insert(0, CODE_DIR)

from config import (
    DATA_ROOT, CANONICAL_CHANNELS, N_CHANNELS, FS,
    WINDOW_SEC, STEP_SEC, GC_ORDER,
    EXCLUDED_PATIENTS, RESULTS_DIR,
    INTERICTAL_MULTIPLIER, MAX_INTERICTAL_ABS, RANDOM_SEED,
    SPH_SEC,
)
from summary_parser import parse_all_summaries
from data_loader import load_edf
import preprocessing as _pp  # monkey-patched per setting

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

np.random.seed(RANDOM_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Preictal window settings to test (minutes).
# SPH stays fixed at 5 min → effective window = PREICTAL - 5.
PREICTAL_SETTINGS = [10, 15, 30]   # 30 min = V4 reference

V7_CACHE_DIR = os.path.join(CODE_DIR, 'cache_gc_features_v7')
os.makedirs(V7_CACHE_DIR, exist_ok=True)

print(f'SPH (fixed) : {SPH_SEC//60} min')
print(f'Preictal settings : {PREICTAL_SETTINGS} min')
print(f'Effective windows : {[p - SPH_SEC//60 for p in PREICTAL_SETTINGS]} min')
print(f'GC order : {GC_ORDER}  (broadband VAR(5))')
print(f'V7 cache : {V7_CACHE_DIR}')
"""),

md("v7_m1", "## 1 · Broadband GC + graph features"),

code("v7_c1", """\
# Cell 1 — Broadband GC (VAR(5), A_total = Σ|A_k|) + 67-feature graph descriptor

def estimate_var_broadband(window, p=GC_ORDER, eps=1e-10):
    n_ch, T = window.shape
    X = window - window.mean(axis=1, keepdims=True)
    T_eff = T - p
    if T_eff < n_ch * p + 1:
        return np.zeros((n_ch, n_ch), dtype=np.float32), False
    Y = X[:, p:]
    Z = np.vstack([X[:, p - lag : p - lag + T_eff] for lag in range(1, p + 1)])
    ZZT = (Z @ Z.T) / T_eff + eps * np.eye(p * n_ch)
    YZT = (Y @ Z.T) / T_eff
    B   = YZT @ np.linalg.inv(ZZT)
    A_total = sum(np.abs(B[:, k * n_ch:(k + 1) * n_ch]) for k in range(p))
    return A_total.astype(np.float32), True


def extract_graph_features(A):
    frob = np.linalg.norm(A, 'fro')
    if frob > 1e-10:
        A = A / frob
    n    = A.shape[0]
    off  = ~np.eye(n, dtype=bool)
    a_off = A[off]
    in_deg   = A.sum(axis=1) - np.diag(A)
    out_deg  = A.sum(axis=0) - np.diag(A)
    net_flow = out_deg - in_deg
    mean_off = a_off.mean(); std_off  = a_off.std()
    max_off  = a_off.max();  min_off  = a_off.min()
    asym     = np.abs(A - A.T)
    asym_off = asym[np.triu_indices(n, k=1)]
    mean_asym = asym_off.mean(); std_asym = asym_off.std()
    thr      = 0.5 * max(float(max_off), 1e-12)
    density  = float((a_off > thr).mean())
    spec_rad = float(np.max(np.abs(np.linalg.eigvals(A))))
    sv = np.linalg.svd(A, compute_uv=False)
    sv5 = sv[:5] if len(sv) >= 5 else np.pad(sv, (0, 5 - len(sv)))
    return np.concatenate([in_deg, out_deg, net_flow,
        [mean_off, std_off, max_off, min_off, mean_asym, std_asym, density, spec_rad],
        sv5]).astype(np.float32)

FEATS_PER_WIN = 67
print('GC + graph-feature functions defined.')
print(f'Features per window : {FEATS_PER_WIN}')
"""),

md("v7_m2", "## 2 · Compute / cache GC features per patient per preictal setting"),

code("v7_c2", """\
# Cell 2 — Per-patient, per-setting feature extraction with preictal monkey-patch

def _cache_paths(preictal_min, pid):
    d = Path(V7_CACHE_DIR) / f'{preictal_min}min' / pid
    return d / 'features.npy', d / 'labels.npy'


def load_gc_features(pid, seizure_map, preictal_min, force=False):
    feat_p, lab_p = _cache_paths(preictal_min, pid)
    if not force and feat_p.exists() and lab_p.exists():
        X = np.load(str(feat_p)); y = np.load(str(lab_p))
        print(f'  [cache] {pid} ({preictal_min}min)  {X.shape}')
        return X, y

    feat_p.parent.mkdir(parents=True, exist_ok=True)

    # Override preictal window before preprocessing
    _pp.PREICTAL_SEC = preictal_min * 60

    pdir      = Path(DATA_ROOT) / pid
    all_feats, all_labs = [], []

    for fname, seizures in sorted(seizure_map.items()):
        edf_path = pdir / fname
        if not edf_path.exists():
            continue
        try:
            raw, fs = load_edf(str(edf_path))
        except Exception:
            continue
        windows, labels, _ = _pp.preprocess_file(raw, seizures, fs)
        if len(windows) == 0:
            continue
        file_feats = []
        for win in windows:
            A, ok = estimate_var_broadband(win)
            file_feats.append(extract_graph_features(A) if ok
                               else np.zeros(FEATS_PER_WIN, dtype=np.float32))
        F = np.stack(file_feats)
        valid = ~np.all(F == 0, axis=1)
        all_feats.append(F[valid]); all_labs.append(labels[valid])

    _pp.PREICTAL_SEC = 30 * 60  # restore default

    if not all_feats:
        return None, None
    X = np.concatenate(all_feats).astype(np.float32)
    y = np.concatenate(all_labs).astype(np.int8)
    np.save(str(feat_p), X); np.save(str(lab_p), y)
    print(f'  [computed] {pid} ({preictal_min}min)  {X.shape}')
    return X, y


all_seizures  = parse_all_summaries(DATA_ROOT)
patients_all  = sorted([
    p for p in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, p))
    and p.startswith('chb') and p not in EXCLUDED_PATIENTS
])

gc_raw = {}   # {preictal_min: {pid: (X, y)}}
for pm in PREICTAL_SETTINGS:
    print(f'\\n=== Preictal {pm} min ===')
    gc_raw[pm] = {}
    t0 = time.time()
    for pid in patients_all:
        if pid not in all_seizures:
            continue
        X, y = load_gc_features(pid, all_seizures[pid], pm)
        if X is None or (y == 1).sum() == 0:
            print(f'  skip {pid}: no preictal windows')
            continue
        gc_raw[pm][pid] = (X, y)
    print(f'  {len(gc_raw[pm])} patients  |  {(time.time()-t0)/60:.1f} min')
"""),

md("v7_m3", "## 3 · Interictal cap + LOPO pipeline"),

code("v7_c3", """\
# Cell 3 — Apply interictal cap (same as V4/V6)

gc_data = {}
for pm in PREICTAL_SETTINGS:
    gc_data[pm] = {}
    for pid, (X, y) in gc_raw[pm].items():
        n_pre = int((y == 1).sum())
        cap   = min(int((y == 0).sum()), INTERICTAL_MULTIPLIER * n_pre, MAX_INTERICTAL_ABS)
        if int((y == 0).sum()) > cap:
            rng  = np.random.default_rng(RANDOM_SEED + hash(pid) % 10_000)
            keep = np.sort(np.concatenate([
                np.where(y == 1)[0],
                rng.choice(np.where(y == 0)[0], size=cap, replace=False)
            ]))
            X, y = X[keep], y[keep]
        gc_data[pm][pid] = (X, y)
    tot_pre = sum(int((yy == 1).sum()) for _, (_, yy) in gc_data[pm].items())
    tot_int = sum(int((yy == 0).sum()) for _, (_, yy) in gc_data[pm].items())
    print(f'{pm:2d}min: {len(gc_data[pm])} patients  pre={tot_pre:,}  int={tot_int:,}')
"""),

code("v7_c4", """\
# Cell 4 — LOPO machinery (LR only — fast, sufficient for sensitivity test)

METRIC_KEYS = ['auc','auc_pr','sensitivity','specificity',
               'precision','f1','balanced_accuracy','fpr_per_hour']


def evaluate_fold(probs, y_te, n_int_hours):
    if len(np.unique(y_te)) < 2:
        return None
    auc    = roc_auc_score(y_te, probs)
    auc_pr = average_precision_score(y_te, probs)
    fpr, tpr, thr = roc_curve(y_te, probs)
    t      = float(thr[np.argmax(tpr - fpr)])
    pred   = (probs >= t).astype(int)
    tp = int(((pred==1)&(y_te==1)).sum()); fp = int(((pred==1)&(y_te==0)).sum())
    tn = int(((pred==0)&(y_te==0)).sum()); fn = int(((pred==0)&(y_te==1)).sum())
    sens = tp/max(tp+fn,1); spec = tn/max(tn+fp,1); prec = tp/max(tp+fp,1)
    return dict(auc=auc, auc_pr=auc_pr, sensitivity=sens, specificity=spec,
                precision=prec, f1=2*prec*sens/max(prec+sens,1e-9),
                balanced_accuracy=0.5*(sens+spec),
                fpr_per_hour=fp/max(n_int_hours,1e-9))


def run_lopo_lr(feat_data, pids, label=''):
    pipe = Pipeline([('scl', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=400, solver='lbfgs',
                         class_weight='balanced', random_state=RANDOM_SEED))])
    grid = {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    rows = []; t0 = time.time()
    print(f'\\n══ LR LOPO [{label}] — {len(pids)} folds ══')
    for i, test_pid in enumerate(pids, 1):
        Xtr = np.concatenate([feat_data[p][0] for p in pids if p != test_pid])
        ytr = np.concatenate([feat_data[p][1] for p in pids if p != test_pid])
        grp = np.concatenate([np.full(len(feat_data[p][1]), j)
                               for j, p in enumerate(pids) if p != test_pid])
        Xte, yte = feat_data[test_pid]
        cv = GroupKFold(n_splits=min(3, len(np.unique(grp))))
        gs = GridSearchCV(pipe, grid, cv=cv, scoring='average_precision',
                          n_jobs=-1, refit=True, verbose=0)
        gs.fit(Xtr, ytr, groups=grp)
        probs = gs.best_estimator_.predict_proba(Xte)[:, 1]
        m = evaluate_fold(probs, yte, (yte==0).sum() * STEP_SEC / 3600.)
        if m:
            m['patient'] = test_pid; rows.append(m)
            print(f'  [{i:2d}] {test_pid}  AUC={m["auc"]:.3f}  PR={m["auc_pr"]:.3f}')
    print(f'  done in {(time.time()-t0)/60:.1f} min')
    return pd.DataFrame(rows)

print('LOPO machinery ready.')
"""),

md("v7_m4", "## 4 · Run LOPO for all preictal settings"),

code("v7_c5", """\
# Cell 5 — Run LR LOPO for each preictal setting

lopo_v7 = {}

for pm in PREICTAL_SETTINGS:
    pids = sorted(gc_data[pm].keys())
    df   = run_lopo_lr(gc_data[pm], pids, label=f'{pm}min')
    lopo_v7[pm] = df

    # Save CSV with MEAN/STD rows
    df_out  = df.copy()
    mean_row = {'patient': 'MEAN', **{k: round(df[k].mean(), 4) for k in METRIC_KEYS}}
    std_row  = {'patient': 'STD',  **{k: round(df[k].std(),  4) for k in METRIC_KEYS}}
    df_out   = pd.concat([df_out, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    path     = os.path.join(RESULTS_DIR, f'lopo_v7_LR_{pm}min.csv')
    df_out.to_csv(path, index=False)
    print(f'Saved {path}')
"""),

md("v7_m5", "## 5 · Results — Sensitivity analysis + Skill Score"),

code("v7_c6", """\
# Cell 6 — Comparison table with prevalence-adjusted Skill Score

def skill_scores(df, feat_data):
    prev = {pid: float((y==1).sum())/len(y) for pid,(_, y) in feat_data.items()}
    vals = [(row['auc_pr'] - prev[row['patient']]) / max(1 - prev[row['patient']], 1e-9)
            for _, row in df.iterrows() if row['patient'] in prev]
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(list(prev.values())))


summary_rows = []
for pm, df in lopo_v7.items():
    sk_mean, sk_std, mean_prev = skill_scores(df, gc_data[pm])
    summary_rows.append({
        'preictal_min':   pm,
        'eff_window_min': pm - SPH_SEC // 60,
        'n_patients':     len(df),
        'mean_prev':      round(mean_prev, 4),
        'auc':            round(df['auc'].mean(), 4),
        'auc_pr':         round(df['auc_pr'].mean(), 4),
        'skill':          round(sk_mean, 4),
        'skill_std':      round(sk_std, 4),
        'sensitivity':    round(df['sensitivity'].mean(), 4),
        'fpr_per_hour':   round(df['fpr_per_hour'].mean(), 1),
    })

summ = pd.DataFrame(summary_rows)
summ.to_csv(os.path.join(RESULTS_DIR, 'lopo_v7_summary.csv'), index=False)

print('══ V7 — Preictal window sensitivity (LR LOPO, broadband GC) ══')
print(f'{"Preictal":>10}  {"Eff.win":>8}  {"Prev":>6}  {"AUC":>7}  '
      f'{"AUC-PR":>7}  {"Skill":>7}  {"±":>6}')
print('─' * 68)
for _, r in summ.iterrows():
    marker = ' ← current (V4)' if r['preictal_min'] == 30 else ''
    print(f'{r["preictal_min"]:>8}min  {r["eff_window_min"]:>6}min  '
          f'{r["mean_prev"]:>6.3f}  {r["auc"]:>7.4f}  {r["auc_pr"]:>7.4f}  '
          f'{r["skill"]:>7.4f}  ±{r["skill_std"]:>5.4f}{marker}')

print('\\nSaved results/lopo_v7_summary.csv')
"""),

code("v7_c7", """\
# Cell 7 — Visualisation: Skill Score and AUC-PR across preictal settings

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels = [f'{r["preictal_min"]}min\\n(eff {r["eff_window_min"]}min)' for _, r in summ.iterrows()]

for ax, metric, ylabel in zip(axes,
                               ['auc_pr', 'skill'],
                               ['AUC-PR', 'Skill Score (prevalence-adjusted)']):
    vals = summ[metric].values
    errs = summ['skill_std'].values if metric == 'skill' else None
    bars = ax.bar(labels, vals, yerr=errs, capsize=6,
                  color=['#e87722' if m == 10 else '#3a7bc8' if m == 15 else '#888'
                         for m in summ['preictal_min']],
                  edgecolor='white', alpha=0.9)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Chance (Skill=0)')
    ax.set_title(f'{ylabel} — V7 preictal window sensitivity')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'v7_preictal_sensitivity.png'), dpi=130)
plt.show()
print('Saved results/v7_preictal_sensitivity.png')
"""),

]  # end v7_cells


# ═══════════════════════════════════════════════════════════════════
# V8 — Within-Patient vs Cross-Patient Comparison
# ═══════════════════════════════════════════════════════════════════

v8_cells = [

md("v8_m0", """\
# Main V8 — Within-Patient vs Cross-Patient Evaluation

## Motivation

Most published papers reporting AUC > 0.85 on CHB-MIT use **within-patient
(personalized) models**: they train and test on the **same patient** using a
temporal split, not a cross-patient leave-one-patient-out (LOPO) protocol.

V8 quantifies this methodological gap directly:

| Protocol | Training data | Expected AUC |
|---|---|---|
| **Cross-patient LOPO** (V3–V6) | All OTHER patients | ~0.54–0.57 |
| **Within-patient** (this notebook) | Same patient (first 70%) | ~ literature |

Both use the **same V6 PDC features** (reused from `cache_pdc_var20/` cache).
No feature recomputation — within-patient just changes the training/test split.

## Within-patient split strategy

For each patient with windows in temporal order (features extracted from sequential
EDF files):
- Find the index of the first preictal window in the **last 30%** of preictal windows
- Use all windows before that index as **training**
- Use all windows from that index onward as **test**

This guarantees: (1) temporal ordering preserved, (2) test always has preictal windows.
"""),

code("v8_c0", """\
# Cell 0 — Imports & load V6 PDC features

import os, sys, json, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

CODE_DIR = _FC_REPO
sys.path.insert(0, CODE_DIR)

from config import (
    DATA_ROOT, CANONICAL_CHANNELS, N_CHANNELS, FS,
    WINDOW_SEC, STEP_SEC, EXCLUDED_PATIENTS, RESULTS_DIR,
    INTERICTAL_MULTIPLIER, MAX_INTERICTAL_ABS, RANDOM_SEED,
)
from summary_parser import parse_all_summaries

from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import SVC
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (roc_auc_score, average_precision_score,
                                       roc_curve)

np.random.seed(RANDOM_SEED)

PDC_CACHE_DIR = os.path.join(CODE_DIR, 'cache_pdc_var20')
BAND_NAMES    = ['delta', 'theta', 'alpha', 'beta']
FEATS_PER_BAND = 67
N_FEATS       = len(BAND_NAMES) * FEATS_PER_BAND   # 268


# ── Load V6 PDC features (no recomputation) ───────────────────────────────
pdc_raw = {}

patients_all = sorted([
    p for p in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, p))
    and p.startswith('chb') and p not in EXCLUDED_PATIENTS
])

for pid in patients_all:
    feat_p = Path(PDC_CACHE_DIR) / pid / 'features.npy'
    lab_p  = Path(PDC_CACHE_DIR) / pid / 'labels.npy'
    if not feat_p.exists():
        continue
    X = np.load(str(feat_p)); y = np.load(str(lab_p))
    if (y == 1).sum() == 0:
        continue
    pdc_raw[pid] = (X, y)
    print(f'  {pid}: {X.shape}  pre={(y==1).sum()}  int={(y==0).sum()}')

print(f'\\nLoaded {len(pdc_raw)} patients  |  features shape: 268 (4 bands × 67)')
"""),

md("v8_m1", "## 1 · Apply interictal cap"),

code("v8_c1", """\
# Cell 1 — Apply interictal cap (same parameters as V6)

pdc_data = {}

for pid, (X, y) in pdc_raw.items():
    n_pre = int((y == 1).sum())
    cap   = min(int((y == 0).sum()), INTERICTAL_MULTIPLIER * n_pre, MAX_INTERICTAL_ABS)
    if int((y == 0).sum()) > cap:
        rng  = np.random.default_rng(RANDOM_SEED + hash(pid) % 10_000)
        keep = np.sort(np.concatenate([
            np.where(y == 1)[0],
            rng.choice(np.where(y == 0)[0], size=cap, replace=False)
        ]))
        X, y = X[keep], y[keep]
    pdc_data[pid] = (X, y)

patient_ids = sorted(pdc_data.keys())
print(f'{len(patient_ids)} patients ready.')
print(f'Total preictal   : {sum(int((y==1).sum()) for _,(_, y) in pdc_data.items()):,}')
print(f'Total interictal : {sum(int((y==0).sum()) for _,(_, y) in pdc_data.items()):,}')
"""),

md("v8_m2", "## 2 · Within-patient evaluation"),

code("v8_c2", """\
# Cell 2 — Within-patient temporal split and model fitting

def within_patient_split(X, y, test_preictal_frac=0.30, min_test_pre=2):
    \"\"\"
    Temporally split (X, y) so that the last `test_preictal_frac` fraction
    of preictal windows go to the test set, along with all subsequent windows.

    Returns (X_train, y_train, X_test, y_test) or (None,)*4 if infeasible.
    \"\"\"
    pre_idx = np.where(y == 1)[0]
    if len(pre_idx) < min_test_pre + 1:
        return None, None, None, None

    n_test_pre = max(min_test_pre, int(len(pre_idx) * test_preictal_frac))
    n_test_pre = min(n_test_pre, len(pre_idx) - 1)   # keep ≥1 in train

    split_at = int(pre_idx[-n_test_pre])   # temporal split point

    Xtr, ytr = X[:split_at], y[:split_at]
    Xte, yte = X[split_at:], y[split_at:]

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None, None, None, None

    return Xtr, ytr, Xte, yte


def evaluate_fold(probs, y_te, n_int_hours):
    if len(np.unique(y_te)) < 2:
        return None
    auc    = roc_auc_score(y_te, probs)
    auc_pr = average_precision_score(y_te, probs)
    fpr, tpr, thr = roc_curve(y_te, probs)
    t    = float(thr[np.argmax(tpr - fpr)])
    pred = (probs >= t).astype(int)
    tp = int(((pred==1)&(y_te==1)).sum()); fp = int(((pred==1)&(y_te==0)).sum())
    tn = int(((pred==0)&(y_te==0)).sum()); fn = int(((pred==0)&(y_te==1)).sum())
    sens = tp/max(tp+fn,1); spec = tn/max(tn+fp,1); prec = tp/max(tp+fp,1)
    return dict(auc=auc, auc_pr=auc_pr, sensitivity=sens, specificity=spec,
                precision=prec, f1=2*prec*sens/max(prec+sens,1e-9),
                balanced_accuracy=0.5*(sens+spec),
                fpr_per_hour=fp/max(n_int_hours,1e-9))


METRIC_KEYS = ['auc','auc_pr','sensitivity','specificity',
               'precision','f1','balanced_accuracy','fpr_per_hour']

print('Within-patient split and evaluation functions defined.')
"""),

code("v8_c3", """\
# Cell 3 — Run within-patient models (LR + SVM) for all patients

models = {
    'LR': Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(max_iter=400, solver='lbfgs',
                        class_weight='balanced', C=1.0,
                        random_state=RANDOM_SEED))]),
    'SVM': Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(kernel='rbf', class_weight='balanced',
                         C=1.0, probability=True, random_state=RANDOM_SEED))]),
}

within_results = {name: [] for name in models}
skipped = []

print(f'Within-patient evaluation  ({len(patient_ids)} patients)')
print(f'Split: last 30% of preictal windows → test, rest → train\\n')

for pid in patient_ids:
    X, y = pdc_data[pid]
    Xtr, ytr, Xte, yte = within_patient_split(X, y)

    if Xtr is None:
        skipped.append(pid)
        print(f'  SKIP {pid}: insufficient preictal windows for split')
        continue

    n_int_hours = (yte == 0).sum() * STEP_SEC / 3600.

    for mname, pipe in models.items():
        pipe.fit(Xtr, ytr)
        probs = pipe.predict_proba(Xte)[:, 1]
        m = evaluate_fold(probs, yte, n_int_hours)
        if m:
            m['patient'] = pid
            within_results[mname].append(m)

    lr_row  = next((r for r in within_results['LR']  if r['patient'] == pid), None)
    svm_row = next((r for r in within_results['SVM'] if r['patient'] == pid), None)
    lr_pr   = f'{lr_row["auc_pr"]:.3f}'  if lr_row  else '-'
    svm_pr  = f'{svm_row["auc_pr"]:.3f}' if svm_row else '-'
    print(f'  {pid}  train={len(ytr)} test={len(yte)}'
          f'  pre_test={(yte==1).sum()}  '
          f'LR AUC-PR={lr_pr}  SVM AUC-PR={svm_pr}')

within_dfs = {name: pd.DataFrame(rows) for name, rows in within_results.items()}

print(f'\\nSkipped: {skipped if skipped else "none"}')
print('\\n══ Within-patient summary ══')
for name, df in within_dfs.items():
    print(f'  {name}: AUC={df["auc"].mean():.4f}  AUC-PR={df["auc_pr"].mean():.4f}  '
          f'(n={len(df)} patients)')
"""),

md("v8_m3", "## 3 · Cross-patient vs within-patient comparison"),

code("v8_c4", """\
# Cell 4 — Load cross-patient LOPO results from V6 + build comparison table

def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    pid_col = [c for c in df.columns if 'patient' in c.lower()][0]
    df = df.rename(columns={pid_col: 'patient'})
    return df[~df['patient'].isin(['MEAN', 'STD'])].reset_index(drop=True)


# Per-patient prevalence (same for both protocols — same EDF files)
patient_prevalence = {
    pid: float((y == 1).sum()) / len(y)
    for pid, (_, y) in pdc_data.items()
}
mean_prev = float(np.mean(list(patient_prevalence.values())))


def summarise(df, label, feat_data=None):
    \"\"\"Compute mean metrics + skill score for a result DataFrame.\"\"\"
    row = {'protocol': label}
    for k in METRIC_KEYS:
        if k in df.columns:
            row[k] = round(df[k].astype(float).mean(), 4)

    # Skill score (needs per-patient prevalence)
    if feat_data is not None:
        skills = []
        for _, r in df.iterrows():
            pid = str(r['patient'])
            if pid in patient_prevalence:
                prev = patient_prevalence[pid]
                skills.append((float(r['auc_pr']) - prev) / max(1 - prev, 1e-9))
        row['skill_mean'] = round(float(np.mean(skills)), 4)
        row['skill_std']  = round(float(np.std(skills)),  4)
    return row


comp_rows = []

# Within-patient results
for name, df in within_dfs.items():
    comp_rows.append(summarise(df, f'Within-patient {name}  ← this notebook',
                               pdc_data))

# Cross-patient LOPO from V6
cross_patient_files = {
    'Cross-patient LOPO SVM (V6)':  'lopo_v6_SVM_all_bands.csv',
    'Cross-patient LOPO LR  (V6)':  'lopo_v6_LR_all_bands.csv',
    'Cross-patient LOPO V3 GC-CNN': 'lopo_v3_window.csv',
    'Cross-patient LOPO V4 LR':     'lopo_v4_LR.csv',
}
for label, fname in cross_patient_files.items():
    df = load_csv(os.path.join(RESULTS_DIR, fname))
    if df is not None:
        comp_rows.append(summarise(df, label, pdc_data))

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(RESULTS_DIR, 'lopo_v8_comparison.csv'), index=False)

print('══ Within-patient vs Cross-patient — Full comparison ══')
print(f'  Prevalence baseline (random AUC-PR) : {mean_prev:.3f}\\n')
print(f'{"Protocol":<45}  {"AUC":>7}  {"AUC-PR":>7}  {"Skill":>7}  {"±":>6}')
print('─' * 75)
for _, r in comp_df.iterrows():
    sk   = f'{r["skill_mean"]:>7.4f}' if 'skill_mean' in r and pd.notna(r.get('skill_mean')) else '      -'
    sk_s = f'±{r["skill_std"]:>5.4f}' if 'skill_std'  in r and pd.notna(r.get('skill_std'))  else '      '
    print(f'{r["protocol"]:<45}  {r["auc"]:>7.4f}  {r["auc_pr"]:>7.4f}  {sk}  {sk_s}')

print(f'\\nSaved results/lopo_v8_comparison.csv')
"""),

code("v8_c5", """\
# Cell 5 — Per-patient AUC scatter: within-patient vs cross-patient LOPO

xp_df = load_csv(os.path.join(RESULTS_DIR, 'lopo_v6_SVM_all_bands.csv'))
wp_df = within_dfs.get('SVM', within_dfs.get('LR'))

if xp_df is not None and wp_df is not None:
    shared = sorted(set(xp_df['patient']) & set(wp_df['patient']))
    xp_auc = xp_df.set_index('patient').loc[shared, 'auc'].astype(float)
    wp_auc = wp_df.set_index('patient').loc[shared, 'auc'].astype(float)
    xp_pr  = xp_df.set_index('patient').loc[shared, 'auc_pr'].astype(float)
    wp_pr  = wp_df.set_index('patient').loc[shared, 'auc_pr'].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xvals, yvals, xlab, ylab, title in [
        (axes[0], xp_auc, wp_auc,
         'Cross-patient LOPO AUC', 'Within-patient AUC', 'AUC-ROC'),
        (axes[1], xp_pr,  wp_pr,
         'Cross-patient LOPO AUC-PR', 'Within-patient AUC-PR', 'AUC-PR'),
    ]:
        ax.scatter(xvals, yvals, color='#3a7bc8', alpha=0.8, s=60, zorder=3)
        for pid in shared:
            ax.annotate(pid.replace('chb',''), (xvals[pid], yvals[pid]),
                        fontsize=6, ha='left', va='bottom')
        lims = [min(xvals.min(), yvals.min()) - 0.05,
                max(xvals.max(), yvals.max()) + 0.05]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='y = x (no difference)')
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f'{title}: within vs cross-patient')
        ax.legend(fontsize=8)

    plt.suptitle('V8: Within-patient (SVM) vs Cross-patient LOPO (V6 SVM)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'v8_within_vs_lopo_scatter.png'), dpi=130)
    plt.show()
    print('Saved results/v8_within_vs_lopo_scatter.png')

    # Wilcoxon test
    try:
        _, p_auc = wilcoxon(wp_auc.values - xp_auc.values,
                            alternative='greater', zero_method='zsplit')
        _, p_pr  = wilcoxon(wp_pr.values  - xp_pr.values,
                            alternative='greater', zero_method='zsplit')
        print(f'\\nWilcoxon (within > cross-patient):')
        print(f'  AUC:   p={p_auc:.4f}  {"significant" if p_auc < 0.05 else "not significant"} at α=0.05')
        print(f'  AUC-PR: p={p_pr:.4f}  {"significant" if p_pr  < 0.05 else "not significant"} at α=0.05')
    except ValueError:
        pass
"""),

code("v8_c6", """\
# Cell 6 — Bar chart summary

protocols = []
auc_vals  = []
pr_vals   = []
colors    = []

for _, r in comp_df.iterrows():
    protocols.append(r['protocol'].split('←')[0].strip())
    auc_vals.append(float(r['auc']))
    pr_vals.append(float(r['auc_pr']))
    colors.append('#e87722' if 'Within' in r['protocol'] else '#888')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, vals, title in zip(axes, [auc_vals, pr_vals], ['AUC-ROC', 'AUC-PR']):
    y_pos = range(len(protocols))
    ax.barh(y_pos, vals, color=colors, edgecolor='white', alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(protocols, fontsize=8)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.4, label='Chance AUC-ROC')
    ax.set_xlabel(title)
    ax.set_title(f'{title} — Within vs Cross-patient')
    ax.legend(fontsize=8)

plt.suptitle('V8: Methodological gap — within-patient vs cross-patient LOPO', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'v8_protocol_comparison.png'), dpi=130)
plt.show()
print('Saved results/v8_protocol_comparison.png')
"""),

md("v8_m4", """\
## 4 · Thesis interpretation

**Expected result**: within-patient AUC substantially higher than cross-patient LOPO.

This directly explains the gap between this thesis and papers reporting AUC > 0.85:
they train and test on the same patient. Our cross-patient LOPO is the harder,
clinically relevant evaluation (a seizure prediction system trained on population
data must generalise to a new patient without any of their recordings).

The within-patient result also quantifies how much of the seizure signal is
**patient-specific** — if within-patient AUC is 0.80 but cross-patient is 0.57,
the majority of the predictive signal is idiosyncratic to each patient,
not a generalizable preictal biomarker.
"""),

]  # end v8_cells

# ── Write notebooks ──────────────────────────────────────────────────────────

v7_path = os.path.join(CODE_DIR, 'Main_V7_PreictalWindow.ipynb')
v8_path = os.path.join(CODE_DIR, 'Main_V8_WithinPatient.ipynb')

with open(v7_path, 'w') as f:
    json.dump(nb(v7_cells), f, indent=1, ensure_ascii=False)

with open(v8_path, 'w') as f:
    json.dump(nb(v8_cells), f, indent=1, ensure_ascii=False)

print(f'Created: {v7_path}')
print(f'Created: {v8_path}')
