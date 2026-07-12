# Epileptic Seizure Prediction from EEG — Cross-Patient Directed Connectivity

Code for the MSc thesis *"Epileptic Seizure Prediction from EEG Signals:
Modelling Pre-Ictal Temporal Dynamics for Cross-Patient Generalization"*
(NOVA Information Management School).

The project asks whether **directed functional connectivity** (Granger
causality / Partial Directed Coherence) carries preictal information that
**generalises across patients** under strict Leave-One-Patient-Out (LOPO)
validation on the CHB-MIT scalp-EEG corpus — and finds, honestly, that it does
not translate into clinically useful cross-patient performance, while the
patient-calibrated regime remains the realistic deployment target.

---

## Repository layout

```
.
├── src/                     Python modules (importable; portable paths via config.py)
│   ├── config.py            all constants + paths (self-locating; CHBMIT_DATA env override)
│   ├── data_loader.py       read CHB-MIT .edf → 18-channel montage
│   ├── summary_parser.py    parse *-summary.txt (seizure times, absolute timeline)
│   ├── preprocessing.py     band-pass, 20 s / 50 % windows, preictal/interictal labelling
│   ├── granger.py           VAR(p) Granger-causality matrices
│   ├── model.py             GC-CNN and DANN (PyTorch)
│   ├── train_evaluate.py    LOPO training / evaluation helpers
│   ├── metrics.py           original metrics  ── see note below ──
│   ├── seizure_metrics.py   CORRECTED event-level metrics (use these)
│   ├── seeds.py             reproducibility (global seed)
│   └── alarm_postproc.py    smoothing + operational threshold selection
│
├── notebooks/
│   ├── 00_diagnostics/      VAR-order (AIC/BIC), Granger diagnostics, patient identifiability
│   ├── 01_experiments/      V1–V12: the main experimental progression
│   ├── 02_baselines/        band power, coherence, correlation, chance regimes
│   └── 03_corrections/      audited re-analysis: corrected metrics, DTF, extra
│                            experiments, statistical validation, all-files &
│                            lead-seizure analyses
│
├── scripts/                 batch feature generation + the all-files extractor
├── results/                 per-experiment result CSVs (the numbers behind the thesis)
├── outputs/                 outputs of the correction/validation notebooks
├── data/                    (git-ignored) put the CHB-MIT recordings here — see data/README.md
├── requirements.txt
└── .gitignore
```

Large feature caches (`cache_*/`) and the raw data are **git-ignored** and
regenerated locally (see below).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the CHB-MIT data into `data/physionet/` (or set `CHBMIT_DATA`) —
see [`data/README.md`](data/README.md).

Paths are portable: `src/config.py` locates the repo root from its own
position, and every notebook begins with a small bootstrap that adds `src/` to
the path and resolves the repo root, so notebooks run from anywhere in the tree
without editing any paths.

---

## How to run

The feature caches are the expensive step; build them once, then the notebooks
are fast.

1. **Generate features** (needs the raw data; writes `cache_*/` at the repo root):
   ```bash
   python scripts/generate_v7_v8.py     # broadband GC (VAR5) + PDC (VAR20) caches
   python scripts/generate_v9.py        # literature-baseline features
   python scripts/extract_allfiles.py   # realistic all-files PDC (see §Realistic evaluation)
   ```
2. **Diagnostics** — `notebooks/00_diagnostics/` (VAR-order selection, identifiability).
3. **Experiments** — `notebooks/01_experiments/` V1 → V12, in order.
4. **Baselines** — `notebooks/02_baselines/`.
5. **Audited re-analysis** — `notebooks/03_corrections/` (run these for the final,
   corrected numbers).

---

## ⚠️ Two metrics modules — read this

`src/metrics.py` is the **original** metrics code. Its false-prediction-rate and
alarm "sensitivity" were computed at the **window level** (every false-positive
10-second window counted as a separate alarm), which produced physically
impossible rates of ~100–300 alarms/hour and a near-zero alarm sensitivity.

`src/seizure_metrics.py` is the **corrected, event-level** implementation used in
`notebooks/03_corrections/`:

* **Event sensitivity** = fraction of seizures with ≥1 alarm in their preictal
  window, after the K-of-M persistence vote and 30-minute refractory.
* **FPR/h** = number of *distinct alarm events* in interictal time per interictal
  hour (≈0.2–1.4/h, not 100+).

**Use `seizure_metrics.py` for any operational metric.** `metrics.py` is kept
only so the original experiment notebooks reproduce as they were first run; the
`03_corrections/` notebooks recompute every operational number at the event level.

---

## Headline results (LOPO, 21 patients)

| Setting | AUC | AUC-PR | Skill | Notes |
|---|---|---|---|---|
| Broadband GC-CNN (VAR1) | 0.534 | 0.232 | −0.21 | near chance |
| Broadband GC graph features (VAR5) | 0.52 | 0.20 | −0.27 | no gain |
| **PDC VAR(20), SVM, all bands** | **0.566** | **0.402** | **+0.091** | best; delta/theta strongest |
| DTF (same VAR20) | 0.522 | 0.364 | +0.05 | PDC > DTF (p<0.05) |
| Within-patient temporal (80/20) | 0.552 | 0.515 | +0.258 | patient-calibrated regime |
| Leaky within-patient random split | 0.927 | 0.845 | +0.77 | shows the evaluation-protocol gap |

* Raw GC matrices identify individual patients at **99.1 % accuracy (20.8× chance)** —
  patient-specific structure dominates the broadband feature space.
* Permutation test (p = 0.001), patient bootstrap (Skill CI excludes 0) and an
  operating curve above the analytic random-predictor confirm the seizure-files
  result is genuinely above chance, though modest.

### Realistic (all-files) evaluation — the honest caveat

Under a **realistic interictal load** (interictal drawn from *all* recordings,
≥1 h from any seizure — `scripts/extract_allfiles.py` + `AllFiles_Evaluation.ipynb`),
cross-patient performance falls to **AUC ≈ 0.52 (not significantly above chance)**
and the skill becomes marginal and single-patient-dependent. The modest
above-chance signal in the standard (seizure-files) evaluation does **not**
survive realistic deployment conditions — reinforcing the thesis's core
conclusion that cross-patient scalp-EEG seizure prediction is not deployable and
the patient-calibrated configuration is the realistic target.

---

## Notes

* **Reproducibility:** a global seed (42) is set via `src/seeds.py`; stochastic
  models (RF, XGBoost, DANN, CNN) are seeded. Report stochastic models as
  mean ± std over seeds for full rigour.
* **Corrected metrics:** all operational FPR/h and sensitivity figures are
  computed at the event level in `src/seizure_metrics.py` and regenerated by the
  `notebooks/03_corrections/` notebooks (the numbers behind the final tables).
* **Data ethics:** CHB-MIT is de-identified pediatric EEG released on PhysioNet;
  it is not redistributed here.
