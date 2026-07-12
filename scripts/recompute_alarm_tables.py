"""
recompute_alarm_tables.py
=========================
Regenerate the corrected FPR/h numbers for thesis Tables 9 and 10 from the
per-patient result CSVs that the original pipeline already produced.

The original pipeline (V6b, V8b) DID compute the correct event-level
`fpr_h_alarm` column, but Tables 9 and 10 quoted the WRONG `fpr_h_window`
column (window-level FP count), yielding the impossible 106-166 alarms/hour.

This script reads the existing `results/lopo_v6b_compare_*.csv` and
`results/lopo_v8b_compare_*.csv` files, contrasts the window-level and
event-level columns, and writes a corrected comparison table.

Run:
    python recompute_alarm_tables.py
Output:
    ../outputs/corrected_alarm_fpr.csv
    ../outputs/corrected_alarm_fpr.md
"""
from __future__ import annotations

import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.abspath(os.path.join(HERE, "..", "outputs"))
os.makedirs(OUT, exist_ok=True)


def load(name):
    path = os.path.join(RESULTS, name)
    with open(path) as fh:
        return list(csv.DictReader(fh))


def mean(rows, col):
    vals = []
    for r in rows:
        v = r.get(col)
        if v not in (None, "", "nan"):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return st.mean(vals) if vals else float("nan")


# (label, file, protocol) — the four rows of thesis Table 9
SPECS = [
    ("Within-patient temporal — SVM", "lopo_v8b_compare_SVM.csv", "Within-patient temporal"),
    ("Within-patient temporal — LR", "lopo_v8b_compare_LR.csv", "Within-patient temporal"),
    ("Cross-patient LOPO — SVM", "lopo_v6b_compare_SVM.csv", "Cross-patient LOPO"),
    ("Cross-patient LOPO — LR", "lopo_v6b_compare_LR.csv", "Cross-patient LOPO"),
]

rows_out = []
for label, fname, proto in SPECS:
    rows = load(fname)
    fpr_window = mean(rows, "fpr_h_window")
    fpr_alarm = mean(rows, "fpr_h_alarm")
    rows_out.append(
        {
            "config": label,
            "protocol": proto,
            "FPR/h (WRONG, window-level, in thesis)": round(fpr_window, 1),
            "FPR/h (CORRECT, event-level)": round(fpr_alarm, 3),
        }
    )

# Write CSV
csv_path = os.path.join(OUT, "corrected_alarm_fpr.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

# Write markdown
md_path = os.path.join(OUT, "corrected_alarm_fpr.md")
with open(md_path, "w") as fh:
    fh.write("# Corrected FPR/h for thesis Tables 9 & 10\n\n")
    fh.write(
        "The thesis quoted the window-level false-positive count as FPR/h. "
        "Below, the last column is the correct event-level alarm rate "
        "(K=5 of M=12, 30-min refractory), which the pipeline already "
        "computed in the `fpr_h_alarm` column but which never reached the "
        "tables.\n\n"
    )
    hdr = list(rows_out[0].keys())
    fh.write("| " + " | ".join(hdr) + " |\n")
    fh.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
    for r in rows_out:
        fh.write("| " + " | ".join(str(r[h]) for h in hdr) + " |\n")

print("Wrote:", csv_path)
print("Wrote:", md_path)
print()
for r in rows_out:
    print(
        f"{r['config']:34s}  thesis(wrong)={r['FPR/h (WRONG, window-level, in thesis)']:>7}"
        f"   corrected={r['FPR/h (CORRECT, event-level)']}"
    )
