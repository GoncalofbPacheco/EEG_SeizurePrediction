#!/usr/bin/env python3
"""
extract_allfiles.py  -  All-files feature extraction, run from the TERMINAL.

Why a script and not a notebook cell: the all-files extraction runs for hours,
and VS Code / Jupyter drops long-running kernels ("notebook controller is
disposed"). A plain terminal process is immune to that. It also processes ONE
patient at a time and frees memory between patients, so it will not spike RAM.

It extracts PDC VAR(20) features from EVERY EDF of each retained patient
(seizure-containing AND seizure-free), labels seizure-free files as all-
interictal, and caches per patient to ../cache_pdc_var20_allfiles/.
It is RESUMABLE: patients already cached are skipped, so if it is interrupted
just run it again.

Run:
    cd scripts
    python extract_allfiles.py            # all patients
    python extract_allfiles.py chb01 chb02   # only these (for a quick test)

Requires: mne, numpy, scikit-learn (same environment that built cache_pdc_var20).
After it finishes, open notebooks/03_corrections/AllFiles_Evaluation.ipynb and run the cells;
the extraction cell will find the cache and load it instantly, then run LOPO.
"""

import os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                       # repo root (scripts/ lives under it)
sys.path.insert(0, str(REPO / "src"))

from config import DATA_ROOT, EXCLUDED_PATIENTS, FS, WINDOW_SEC
from summary_parser import parse_summary, parse_summary_with_times
from data_loader import load_edf
from preprocessing import preprocess_file

# Literature interictal definition: an interictal window must lie at least
# INTERICTAL_BUFFER_H hours from EVERY seizure, measured on the patient's full
# absolute recording timeline (across files), not just within the file it
# belongs to. This prevents pre-/post-ictal contamination of the interictal
# class (the convention behind Truong et al. 2018 and other prediction work).
#
# Buffer choice: 1 h is used here. The stricter 4 h separation (Truong) was
# tested but starves the seizure-dense patients chb08 (5 seizures) and chb15
# (20 seizures), whose recordings sit almost entirely within 4 h of a seizure,
# leaving fewer interictal than preictal windows. A 1 h separation keeps all 21
# patients viable while remaining a literature-supported choice; set to 2.0 or
# 4.0 to tighten it.
INTERICTAL_BUFFER_H = 1.0
BUFFER_S = INTERICTAL_BUFFER_H * 3600.0

PDC_ORDER = 20
BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
]
BAND_NAMES = [b[0] for b in BANDS]
FEATS = 67
ALL_CACHE = REPO / "cache_pdc_var20_allfiles"
ALL_CACHE.mkdir(exist_ok=True)


def estimate_var_matrix(window, p=PDC_ORDER, eps=1e-10):
    n_ch, T = window.shape
    X = window - window.mean(1, keepdims=True)
    Te = T - p
    if Te < n_ch * p + 1:
        return np.zeros((n_ch, p * n_ch)), False
    Y = X[:, p:]
    Z = np.vstack([X[:, p - l : p - l + Te] for l in range(1, p + 1)])
    ZZT = (Z @ Z.T) / Te + eps * np.eye(p * n_ch)
    return ((Y @ Z.T) / Te) @ np.linalg.inv(ZZT), True


def compute_pdc_bands(B, bands=BANDS, fs=FS, p=PDC_ORDER, nfh=4.0):
    n = B.shape[0]
    A = [B[:, k * n : (k + 1) * n] for k in range(p)]
    out = {}
    for nm, lo, hi in bands:
        nf = max(2, int((hi - lo) * nfh))
        fr = np.linspace(lo, hi, nf)
        acc = np.zeros((n, n))
        for f in fr:
            Af = np.eye(n, dtype=complex)
            for k in range(p):
                Af -= A[k] * np.exp(-2j * np.pi * f * (k + 1) / fs)
            num = np.abs(Af)
            den = np.sqrt((np.abs(Af) ** 2).sum(0, keepdims=True))  # column norm = PDC
            acc += (num / np.maximum(den, 1e-12)) ** 2
        out[nm] = acc / len(fr)
    return out


def extract_graph_features(A):
    frob = np.linalg.norm(A, "fro")
    if frob > 1e-10:
        A = A / frob
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    a = A[off]
    ind = A.sum(1) - np.diag(A)
    outd = A.sum(0) - np.diag(A)
    net = outd - ind
    asym = np.abs(A - A.T)
    ao = asym[np.triu_indices(n, 1)]
    thr = 0.5 * max(float(a.max()), 1e-12)
    dens = float((a > thr).mean())
    spec = float(np.max(np.abs(np.linalg.eigvals(A))))
    sv = np.linalg.svd(A, compute_uv=False)
    sv5 = sv[:5] if len(sv) >= 5 else np.pad(sv, (0, 5 - len(sv)))
    return np.concatenate(
        [
            ind,
            outd,
            net,
            [a.mean(), a.std(), a.max(), a.min(), ao.mean(), ao.std(), dens, spec],
            sv5,
        ]
    ).astype(np.float32)


def all_edf_files(pid):
    d = os.path.join(DATA_ROOT, pid)
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".edf"))


def _far_from_all_seizures(abs_t, global_seizures):
    """True if a window at absolute time abs_t (its span [abs_t, abs_t+WINDOW_SEC])
    is at least BUFFER_S from every seizure interval."""
    w0, w1 = abs_t, abs_t + WINDOW_SEC
    for on, off in global_seizures:
        if not (w1 <= on - BUFFER_S or w0 >= off + BUFFER_S):
            return False
    return True


def extract_patient(pid):
    fp = ALL_CACHE / pid / "features.npy"
    lp = ALL_CACHE / pid / "labels.npy"
    if fp.exists() and lp.exists():
        y = np.load(lp)
        print(
            f"  [skip] {pid} already cached  (pre={int((y == 1).sum())} int={int((y == 0).sum())})"
        )
        return
    (ALL_CACHE / pid).mkdir(parents=True, exist_ok=True)
    pdir = os.path.join(DATA_ROOT, pid)
    smap = parse_summary(pdir)  # {seizure_edf: [(onset,offset),...]} local times
    tl = parse_summary_with_times(pdir)  # {edf: {abs_start, abs_end, seizures_abs}}
    global_seizures = sorted(
        s for info in tl.values() for s in info.get("seizures_abs", [])
    )
    feats, labs = [], []
    n_pre = n_int = n_dropped = 0
    edfs = all_edf_files(pid)
    for j, edf in enumerate(edfs, 1):
        path = os.path.join(pdir, edf)
        seizures = smap.get(
            edf, []
        )  # local seizure times for in-file preictal labelling
        abs_start = tl.get(edf, {}).get("abs_start")
        try:
            raw, fs = load_edf(path)
            wins, labels, starts = preprocess_file(raw, seizures, fs)
        except Exception as e:
            print(f"    ! {edf}: {e}")
            continue
        for win, lab, st in zip(wins, labels, starts):
            if lab == 0:
                # interictal: keep only if >= BUFFER_S from every seizure (Truong rule),
                # measured on the absolute timeline. Patients whose summary lacks file
                # start-times (e.g. chb24) have no absolute timeline; for them we fall
                # back to the per-file interictal already labelled by preprocess_file
                # (a documented exception, noted in the methodology).
                if abs_start is not None and not _far_from_all_seizures(
                    abs_start + st, global_seizures
                ):
                    n_dropped += 1
                    continue
            B, ok = estimate_var_matrix(win)
            if not ok:
                feats.append(np.zeros(FEATS * 4, dtype=np.float32))
                labs.append(lab)
            else:
                bands = compute_pdc_bands(B)
                feats.append(
                    np.concatenate(
                        [extract_graph_features(bands[b]) for b in BAND_NAMES]
                    )
                )
                labs.append(lab)
            if lab == 1:
                n_pre += 1
            else:
                n_int += 1
        print(
            f"    {edf} ({j}/{len(edfs)})  kept pre={n_pre} int={n_int} (dropped {n_dropped} near-seizure interictal)",
            flush=True,
        )
    X = np.array(feats, dtype=np.float32)
    y = np.array(labs, dtype=np.int8)
    np.save(fp, X)
    np.save(lp, y)
    print(
        f"  [done] {pid}: {X.shape}  pre={int((y == 1).sum())} int={int((y == 0).sum())}  "
        f"(interictal >= {INTERICTAL_BUFFER_H} h from any seizure)"
    )


def main(argv):
    patients = sorted(
        p
        for p in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, p))
        and p.startswith("chb")
        and p not in EXCLUDED_PATIENTS
    )
    if len(argv) > 1:
        patients = [p for p in patients if p in set(argv[1:])]
    print(f"Extracting all-files PDC for {len(patients)} patient(s) -> {ALL_CACHE}")
    t0 = time.time()
    for pid in patients:
        try:
            extract_patient(pid)
        except Exception:
            print(f"  !! FAILED on {pid}:")
            traceback.print_exc()
        print(f"  elapsed {time.time() - t0:.0f}s\n", flush=True)
    print("All requested patients processed.")


if __name__ == "__main__":
    main(sys.argv)
