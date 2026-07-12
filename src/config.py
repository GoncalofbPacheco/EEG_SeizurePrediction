"""
config.py  (V3.1 — imbalance bug fixed)
=========================================
Changes vs original:
  FIX 2  — GC_ORDER raised from 1 → 5  (captures multi-lag EEG dynamics)
  FIX 1  — INTERICTAL_MULTIPLIER: per-patient proportional cap (5× preictal)
            replaces the old fixed MAX_INTERICTAL_PER_PATIENT = 15 000.
            A fixed cap created 1:44 ratios; a multiplier guarantees 1:5 for
            every patient regardless of how many preictal windows they have.
  FIX 3/4 — ALARM_K, ALARM_M, ALARM_REFRACTORY (sliding-window alarm logic)
  NEW    — GC_MATRICES_DIR_V3 separate cache so old VAR(1) cache is never reused

Paths are resolved relative to the repository root (the parent of this src/
folder) so the project is portable across machines. To point at the CHB-MIT
data without moving it, set the CHBMIT_DATA environment variable, e.g.
    export CHBMIT_DATA=/path/to/physionet
Otherwise the data is expected in <repo>/data/physionet (see data/README.md).
"""
import os
from pathlib import Path

# Repository root = parent of this src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Dataset ───────────────────────────────────────────────────────────────────
# CHB-MIT raw recordings (not redistributed — download from PhysioNet).
DATA_ROOT = os.environ.get("CHBMIT_DATA", str(PROJECT_ROOT / "data" / "physionet"))

# Patients excluded from all experiments (thesis §3.2)
EXCLUDED_PATIENTS = ["chb12", "chb21", "chb11"]

# 18 canonical bipolar channels shared across all retained patients
CANONICAL_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]
N_CHANNELS = len(CANONICAL_CHANNELS)  # 18

# ── Signal processing ─────────────────────────────────────────────────────────
FS = 256          # native sampling frequency (Hz)
LOWCUT = 0.5      # band-pass lower edge (Hz)
HIGHCUT = 30.0    # band-pass upper edge (Hz)
FILTER_ORDER = 4  # Butterworth filter order (zero-phase via filtfilt)

# ── Segmentation ──────────────────────────────────────────────────────────────
WINDOW_SEC = 20                              # window duration (s)
OVERLAP    = 0.50                            # 50 % overlap
STEP_SEC   = int(WINDOW_SEC * (1 - OVERLAP)) # 10 s step

WINDOW_SAMP = WINDOW_SEC * FS   # 5 120 samples per window
STEP_SAMP   = STEP_SEC   * FS   # 2 560 samples per step

# ── Labelling ─────────────────────────────────────────────────────────────────
PREICTAL_SEC       = 30 * 60   # 1 800 s — preictal window length
SPH_SEC            =  5 * 60   #   300 s — seizure prediction horizon
POSTICTAL_EXCL_SEC = 30 * 60   # 1 800 s — post-ictal exclusion buffer

# ── Granger causality ─────────────────────────────────────────────────────────
# FIX 2: Raised from 1 → 5.
# VAR(1) only captures 1-step (≈4 ms) autocorrelation — almost no seizure signal.
# VAR(5) captures dynamics up to 5 * (1/256) ≈ 20 ms, which covers the
# theta/alpha band oscillatory cycles most relevant to preictal activity.
GC_ORDER = 5

# ── Model ─────────────────────────────────────────────────────────────────────
DROPOUT_RATE  = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64
MAX_EPOCHS    = 50
PATIENCE      = 10      # early stopping patience (validation loss)
RANDOM_SEED   = 42

# ── Paths (absolute, under the repository root, so they work from any cwd) ────
# Feature caches live at the repository root as cache_<name>/ and are git-ignored
# (regenerated from the raw data — see the top-level README). Notebooks build the
# same paths as Path(CODE_DIR) / "cache_<name>".
GC_MATRICES_DIR    = str(PROJECT_ROOT / "cache_gc_var1")   # obsolete VAR(1) cache — do not reuse
GC_MATRICES_DIR_V3 = str(PROJECT_ROOT / "cache_gc_var5")   # VAR(5) cache used by V3/V4
RESULTS_DIR        = str(PROJECT_ROOT / "results")

# ── FIX 1: Per-patient proportional interictal cap ───────────────────────────
#
# BUG in V3.0: MAX_INTERICTAL_PER_PATIENT = 15 000 (fixed number) created
# a 1:44 ratio for patients with only 296 preictal windows, and only 1:5 for
# chb24 which had 939. The fixed cap interacted badly with the balanced sampler:
#
#   training batches → balanced (sampler oversamples preictal)  → loss ≈ 0.017
#   validation set   → 1:19 true distribution                   → loss ≈ 0.268
#   early stopping   → fires on inflated val loss → model undertrained
#
# FIX: cap interictal at INTERICTAL_MULTIPLIER × n_preictal per patient.
# With multiplier = 5, every patient has a 1:5 ratio regardless of seizure count.
# This makes train loss and val loss computed on comparable distributions,
# so early stopping behaves correctly.
#
# Result from data:  total interictal drops from 225 243 → ~54 540 (1:5 ratio)
INTERICTAL_MULTIPLIER = 5       # interictal = min(available, 5 × preictal)
MAX_INTERICTAL_ABS    = 5_000   # hard ceiling per patient (prevents chb06 from dominating)

# ── FIX 3 & 4: Alarm-level post-processing ────────────────────────────────────
# Instead of treating each 10-second window as an independent alarm:
#   1. Sliding-window vote: fire alarm only if ≥ ALARM_K of the last ALARM_M
#      windows are predicted positive.
#   2. Refractory period: after an alarm fires, suppress the next
#      ALARM_REFRACTORY windows (30 min = 180 × 10-second steps).
#
# Default parameters (can be tuned):
#   ALARM_M = 12  → 2-minute voting window
#   ALARM_K =  5  → 5/12 threshold (≈ 42 %) — lowered from 8 because at
#                   AUC ≈ 0.53 the original K=8 almost never fires (sensitivity ≈ 0).
#                   K=5 gives the alarm a chance to fire while still suppressing
#                   isolated single-window false positives.
#   ALARM_REFRACTORY = 180 → 30-minute refractory period
ALARM_K           = 5    # minimum positives in window to fire alarm
ALARM_M           = 12   # sliding-window size (windows)
ALARM_REFRACTORY  = 180  # refractory period (windows)
