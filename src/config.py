"""
config.py
=========
Central configuration: dataset location, montage, signal-processing,
segmentation, labelling, model and alarm-post-processing constants.

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
# VAR model order. At 256 Hz, VAR(1) captures only ~4 ms of history — shorter
# than a single EEG oscillation cycle (alpha at 10 Hz has a 100 ms period), so
# it encodes almost no frequency-domain preictal information. Order 5 spans
# 5 × (1/256) ≈ 20 ms, covering the theta/alpha band coupling most relevant to
# preictal activity.
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
GC_MATRICES_DIR = str(PROJECT_ROOT / "cache_gc_var5")   # VAR(5) Granger-causality cache
RESULTS_DIR     = str(PROJECT_ROOT / "results")

# ── Class-imbalance handling ──────────────────────────────────────────────────
# Cap interictal windows at INTERICTAL_MULTIPLIER × n_preictal per patient, so
# every patient keeps the same preictal:interictal ratio regardless of seizure
# count. A fixed absolute cap gives patients with few preictal windows a far more
# skewed ratio than patients with many; with a proportional cap the training and
# validation sets share comparable class distributions, which keeps the balanced
# sampler and early stopping well behaved.
INTERICTAL_MULTIPLIER = 5       # interictal = min(available, 5 × preictal)
MAX_INTERICTAL_ABS    = 5_000   # hard ceiling per patient

# ── Alarm-level post-processing ───────────────────────────────────────────────
# Rather than treating each 10-second window as an independent alarm:
#   1. Sliding-window vote: fire an alarm only if ≥ ALARM_K of the last ALARM_M
#      windows are predicted positive.
#   2. Refractory period: after an alarm fires, suppress the next
#      ALARM_REFRACTORY windows (30 min = 180 × 10-second steps), so a seizure
#      does not trigger repeated alarms.
# ALARM_K = 5 of ALARM_M = 12 (≈42 %) fires often enough to catch seizures while
# still suppressing isolated single-window false positives.
ALARM_K           = 5    # minimum positives in window to fire alarm
ALARM_M           = 12   # sliding-window size (windows)
ALARM_REFRACTORY  = 180  # refractory period (windows)
