"""
patient_correlation.py
======================
Supervisor comment 167: "Does cross-patient AUC-PR correlate with any
recorded patient property (seizure count, age, average seizure duration in
Table 2)? Could be a nice extra result."

Also supports comment 141/156 ("investigate which cases perform poorly").

Correlates the per-patient LOPO AUC-PR of the best configuration
(PDC VAR(20), SVM, all bands = lopo_v6_SVM_all_bands.csv) against the
patient metadata in thesis Table 2.

Run:
    python patient_correlation.py
Output:
    ../outputs/patient_correlation.csv
    ../outputs/patient_correlation.md
"""
from __future__ import annotations

import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))
os.makedirs(OUT, exist_ok=True)

# Thesis Table 2 (age in years, seizure count, avg seizure duration in s).
# chb24 age/gender are Not Disclosed -> None. chb11/chb12/chb21 excluded.
TABLE2 = {
    "chb01": {"age": 11.0, "n_seiz": 7, "dur": 63.14},
    "chb02": {"age": 11.0, "n_seiz": 3, "dur": 57.33},
    "chb03": {"age": 14.0, "n_seiz": 7, "dur": 57.43},
    "chb04": {"age": 22.0, "n_seiz": 4, "dur": 94.50},
    "chb05": {"age": 7.0, "n_seiz": 5, "dur": 111.60},
    "chb06": {"age": 1.5, "n_seiz": 10, "dur": 15.30},
    "chb07": {"age": 14.5, "n_seiz": 3, "dur": 108.33},
    "chb08": {"age": 3.5, "n_seiz": 5, "dur": 183.80},
    "chb09": {"age": 10.0, "n_seiz": 4, "dur": 69.00},
    "chb10": {"age": 3.0, "n_seiz": 7, "dur": 53.86},
    "chb13": {"age": 3.0, "n_seiz": 10, "dur": 41.30},
    "chb14": {"age": 9.0, "n_seiz": 8, "dur": 21.12},
    "chb15": {"age": 16.0, "n_seiz": 20, "dur": 99.60},
    "chb16": {"age": 7.0, "n_seiz": 10, "dur": 8.40},
    "chb17": {"age": 12.0, "n_seiz": 3, "dur": 97.67},
    "chb18": {"age": 18.0, "n_seiz": 6, "dur": 52.83},
    "chb19": {"age": 19.0, "n_seiz": 3, "dur": 78.67},
    "chb20": {"age": 6.0, "n_seiz": 8, "dur": 36.75},
    "chb22": {"age": 9.0, "n_seiz": 3, "dur": 68.00},
    "chb23": {"age": 6.0, "n_seiz": 7, "dur": 60.57},
    "chb24": {"age": None, "n_seiz": 16, "dur": 31.94},
}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def load_aucpr():
    path = os.path.join(RESULTS, "lopo_v6_SVM_all_bands.csv")
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            pid = r["patient"]
            if pid in ("MEAN", "STD"):
                continue
            out[pid] = float(r["auc_pr"])
    return out


aucpr = load_aucpr()
rows = []
for pid, m in TABLE2.items():
    if pid not in aucpr:
        continue
    rows.append({"patient": pid, "auc_pr": aucpr[pid], **m})

# Correlations (drop patients with missing age for the age correlation)
props = ["n_seiz", "age", "dur"]
labels = {
    "n_seiz": "seizure count",
    "age": "age (years)",
    "dur": "avg seizure duration (s)",
}
corr = {}
for p in props:
    xs, ys = [], []
    for r in rows:
        if r[p] is None:
            continue
        xs.append(float(r[p]))
        ys.append(float(r["auc_pr"]))
    corr[p] = (pearson(xs, ys), len(xs))

# Poorly performing cases (comment 141/156)
poor = sorted(rows, key=lambda r: r["auc_pr"])[:5]

csv_path = os.path.join(OUT, "patient_correlation.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["patient", "auc_pr", "n_seiz", "age", "dur"])
    w.writeheader()
    for r in sorted(rows, key=lambda r: r["patient"]):
        w.writerow({k: r[k] for k in ["patient", "auc_pr", "n_seiz", "age", "dur"]})

md_path = os.path.join(OUT, "patient_correlation.md")
with open(md_path, "w") as fh:
    fh.write("# Per-patient AUC-PR vs patient properties (PDC VAR(20) SVM, LOPO)\n\n")
    fh.write("Pearson correlation between cross-patient LOPO AUC-PR and each property:\n\n")
    fh.write("| Property | Pearson r | n |\n|---|---|---|\n")
    for p in props:
        r, n = corr[p]
        fh.write(f"| {labels[p]} | {r:+.3f} | {n} |\n")
    fh.write("\n## Five worst-performing patients (lowest AUC-PR)\n\n")
    fh.write("| Patient | AUC-PR | seizures | age | avg dur (s) |\n|---|---|---|---|---|\n")
    for r in poor:
        fh.write(
            f"| {r['patient']} | {r['auc_pr']:.3f} | {r['n_seiz']} | "
            f"{r['age']} | {r['dur']} |\n"
        )

print("Wrote:", csv_path)
print("Wrote:", md_path)
print("\nPearson correlation of AUC-PR with:")
for p in props:
    r, n = corr[p]
    print(f"  {labels[p]:26s} r = {r:+.3f}  (n={n})")
print("\nWorst 5 patients:", ", ".join(f"{r['patient']}({r['auc_pr']:.2f})" for r in poor))
