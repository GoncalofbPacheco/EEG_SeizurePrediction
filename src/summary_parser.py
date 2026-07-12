"""
summary_parser.py  (V3 — added timeline reconstruction + clean interictal filter)
==================================================================================
Parses CHB-MIT per-patient summary files (*-summary.txt).

Original functions (unchanged):
  parse_summary()          — seizure map for one patient
  parse_all_summaries()    — seizure maps for all patients
  get_seizure_containing_files()

New functions (V3 Option B):
  parse_summary_with_times()    — full file timeline with absolute timestamps
  get_clean_interictal_files()  — files that are ≥ N hours from any seizure

Why this matters
----------------
CHB-MIT summary files include "File Start Time" and "File End Time" for every
EDF. By reconstructing the absolute timeline we can apply a temporal distance
criterion for interictal selection — the standard in the prediction literature
(e.g. ≥4h before and ≥4h after any seizure) instead of using all non-seizure
files blindly, which would include post-ictal and pre-ictal recordings.

CHB-MIT time quirks handled:
  • Files spanning midnight use "24:xx:xx" instead of "00:xx:xx" in some patients.
  • Gaps between files (hardware gaps) are typically <10s but can be hours.
    Day rollovers are detected by checking whether a file's start time is more
    than 1 hour before the previous file's end time.
  • Missing file numbers (e.g. chb20 jumps from _08 to _11) — handled because
    we process files in chronological order from the summary, not by filename.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# ── Type aliases ──────────────────────────────────────────────────────────────
# {edf_filename: [(onset_sec, offset_sec), ...]}  — times within the file
SeizureMap = Dict[str, List[Tuple[float, float]]]

# {edf_filename: {"abs_start": float, "abs_end": float,
#                 "seizures_abs": [(abs_onset, abs_offset), ...]}}
FileTimeline = Dict[str, Dict]

SECS_PER_DAY = 86_400


def parse_summary(patient_dir: str) -> SeizureMap:
    """
    Parse a single patient's summary file.

    Parameters
    ----------
    patient_dir : str
        Path to the patient folder (e.g. '.../chb01/').

    Returns
    -------
    seizure_map : SeizureMap
        Keys   — EDF file name (e.g. 'chb01_03.edf')
        Values — list of (onset_sec, offset_sec) tuples (times within the file)
                 Only files with ≥1 seizure are included.
    """
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name  # e.g. 'chb01'
    summary_path = patient_dir / f"{patient_id}-summary.txt"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    text = summary_path.read_text(encoding="utf-8", errors="replace")

    seizure_map: SeizureMap = {}

    # Split into per-file blocks at every "File Name:" header
    blocks = re.split(r"(?=File Name:)", text, flags=re.IGNORECASE)

    for block in blocks:
        # ── File name ─────────────────────────────────────────────────────────
        fn_match = re.search(r"File Name:\s*(\S+\.edf)", block, re.IGNORECASE)
        if not fn_match:
            continue
        edf_name = fn_match.group(1).strip()

        # ── Number of seizures ────────────────────────────────────────────────
        n_match = re.search(
            r"Number of Seizures in File:\s*(\d+)", block, re.IGNORECASE
        )
        if not n_match:
            continue
        n_seizures = int(n_match.group(1))
        if n_seizures == 0:
            continue

        # ── Seizure times ─────────────────────────────────────────────────────
        # Handles both "Seizure Start Time:" and "Seizure N Start Time:" formats
        onsets = [
            float(m)
            for m in re.findall(
                r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s+seconds",
                block,
                re.IGNORECASE,
            )
        ]
        offsets = [
            float(m)
            for m in re.findall(
                r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s+seconds",
                block,
                re.IGNORECASE,
            )
        ]

        if len(onsets) != n_seizures or len(offsets) != n_seizures:
            print(
                f"[WARNING] Seizure count mismatch in {edf_name} "
                f"(expected {n_seizures}, found {len(onsets)} onsets / {len(offsets)} offsets). "
                f"Skipping file."
            )
            continue

        seizure_map[edf_name] = list(zip(onsets, offsets))

    return seizure_map


def parse_all_summaries(
    data_root: str, excluded_patients: List[str] = None
) -> Dict[str, SeizureMap]:
    """
    Parse summary files for every patient in the CHB-MIT root directory.

    Parameters
    ----------
    data_root : str
        Root directory containing per-patient subdirectories (chb01, chb02, …).
    excluded_patients : list[str], optional
        Patient IDs to skip (e.g. ['chb12']).

    Returns
    -------
    all_seizures : dict
        {patient_id: SeizureMap}
    """
    excluded_patients = set(excluded_patients or [])
    data_root = Path(data_root)

    # Find patient directories (folders matching chbXX pattern)
    patient_dirs = sorted(
        [
            d
            for d in data_root.iterdir()
            if d.is_dir() and re.match(r"chb\d{2}$", d.name)
        ]
    )

    all_seizures: Dict[str, SeizureMap] = {}

    for pdir in patient_dirs:
        pid = pdir.name
        if pid in excluded_patients:
            print(f"[INFO] Skipping excluded patient: {pid}")
            continue
        try:
            smap = parse_summary(str(pdir))
            n_files = len(smap)
            n_seizures = sum(len(v) for v in smap.values())
            print(
                f"[INFO] {pid}: {n_files} seizure-containing files, {n_seizures} total seizures"
            )
            all_seizures[pid] = smap
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")

    return all_seizures


def get_seizure_containing_files(patient_seizure_map: SeizureMap) -> List[str]:
    """Return list of EDF file names that contain at least one seizure."""
    return list(patient_seizure_map.keys())


# ── V3: Timeline reconstruction ───────────────────────────────────────────────

def _parse_hhmmss(t: str) -> float:
    """
    Parse "HH:MM:SS" to seconds.
    Handles CHB-MIT "24:xx:xx" convention (some patients write 24: instead of 00:
    for the first hour after midnight).
    """
    parts = t.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return float(h * 3600 + m * 60 + s)


def parse_summary_with_times(patient_dir: str) -> FileTimeline:
    """
    Parse the summary file and return absolute timestamps for every EDF file.

    The summary lists File Start Time / File End Time for every file (not just
    seizure files). We reconstruct absolute wall-clock seconds from the first
    file, handling:
      • "24:xx:xx" end times (CHB-MIT convention for the first post-midnight hour)
      • True day rollovers detected by comparing consecutive start times

    Parameters
    ----------
    patient_dir : str

    Returns
    -------
    timeline : FileTimeline
        {fname: {
            "abs_start":     float   — seconds from start of first file
            "abs_end":       float
            "seizures_rel":  [(onset_sec, offset_sec)]  — within-file times
            "seizures_abs":  [(onset_abs, offset_abs)]  — absolute times
            "has_seizure":   bool
        }}
        Files are in summary order (chronological).
    """
    patient_dir   = Path(patient_dir)
    patient_id    = patient_dir.name
    summary_path  = patient_dir / f"{patient_id}-summary.txt"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    text   = summary_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?=File Name:)", text, flags=re.IGNORECASE)

    # ── First pass: collect raw per-file data in summary order ────────────────
    raw = []   # list of (fname, t_start_str, t_end_str, seizures_rel)
    for block in blocks:
        fn_m = re.search(r"File Name:\s*(\S+\.edf)", block, re.IGNORECASE)
        if not fn_m:
            continue
        fname = fn_m.group(1).strip()

        ts_m = re.search(r"File Start Time:\s*([\d:]+)", block, re.IGNORECASE)
        te_m = re.search(r"File End Time:\s*([\d:]+)",   block, re.IGNORECASE)
        if not ts_m or not te_m:
            continue

        n_m = re.search(r"Number of Seizures in File:\s*(\d+)", block, re.IGNORECASE)
        n_seizures = int(n_m.group(1)) if n_m else 0

        onsets  = [float(x) for x in re.findall(
            r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s+seconds", block, re.IGNORECASE)]
        offsets = [float(x) for x in re.findall(
            r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s+seconds",   block, re.IGNORECASE)]

        seizures_rel = []
        if n_seizures > 0 and len(onsets) == n_seizures == len(offsets):
            seizures_rel = list(zip(onsets, offsets))

        raw.append((fname, ts_m.group(1), te_m.group(1), seizures_rel))

    # ── Second pass: reconstruct absolute timeline ─────────────────────────────
    timeline: FileTimeline = {}
    day_offset = 0.0
    prev_abs_end = None

    for fname, t_start_str, t_end_str, seizures_rel in raw:
        raw_start = _parse_hhmmss(t_start_str)
        raw_end   = _parse_hhmmss(t_end_str)

        abs_start = raw_start + day_offset
        abs_end   = raw_end   + day_offset

        # "24:xx:xx" → end is already > 86400, so abs_end > abs_start — correct.
        # True midnight rollover: raw_end < raw_start (e.g. start=23:50, end=00:05)
        if abs_end < abs_start:
            abs_end += SECS_PER_DAY

        # Day rollover between files: new file's start appears > 1h before
        # previous file's end (clock rolled over midnight)
        if prev_abs_end is not None and abs_start < prev_abs_end - 3600:
            day_offset += SECS_PER_DAY
            abs_start  += SECS_PER_DAY
            abs_end    += SECS_PER_DAY

        prev_abs_end = abs_end

        seizures_abs = [
            (abs_start + on, abs_start + off)
            for on, off in seizures_rel
        ]

        timeline[fname] = {
            "abs_start":    abs_start,
            "abs_end":      abs_end,
            "seizures_rel": seizures_rel,
            "seizures_abs": seizures_abs,
            "has_seizure":  len(seizures_rel) > 0,
        }

    return timeline


# ── V3: Clean interictal file selector ───────────────────────────────────────

def get_clean_interictal_files(
    timeline:         FileTimeline,
    pre_seizure_gap:  float = 4 * 3600,   # seconds before seizure onset
    post_seizure_gap: float = 4 * 3600,   # seconds after seizure offset
    verbose:          bool  = True,
) -> Tuple[List[str], List[str]]:
    """
    Return EDF files that are temporally distant from all seizures.

    A file is CLEAN if:
      • It contains no seizures (not a seizure file), AND
      • Its end time is ≥ pre_seizure_gap before every seizure onset, AND
      • Its start time is ≥ post_seizure_gap after every seizure offset.

    This implements the standard interictal selection criterion from the
    seizure prediction literature (e.g. ≥4h buffer on both sides).

    Parameters
    ----------
    timeline          : FileTimeline from parse_summary_with_times()
    pre_seizure_gap   : seconds of buffer before any seizure onset  (default 4h)
    post_seizure_gap  : seconds of buffer after  any seizure offset (default 4h)
    verbose           : print exclusion reasons

    Returns
    -------
    clean_files    : list of filenames safe to use as interictal
    excluded_files : list of filenames that were excluded (with reasons if verbose)
    """
    # Collect all absolute seizure (onset, offset) pairs across the entire patient
    all_seizures_abs: List[Tuple[float, float]] = []
    for info in timeline.values():
        all_seizures_abs.extend(info["seizures_abs"])

    if not all_seizures_abs:
        # No seizures at all — nothing to exclude
        return list(timeline.keys()), []

    clean_files:    List[str] = []
    excluded_files: List[str] = []

    for fname, info in timeline.items():
        # Always exclude seizure files themselves
        if info["has_seizure"]:
            excluded_files.append(fname)
            continue

        abs_start = info["abs_start"]
        abs_end   = info["abs_end"]
        too_close = False
        reason    = ""

        for onset, offset in all_seizures_abs:
            # File overlaps with pre-seizure buffer
            if abs_end > onset - pre_seizure_gap and abs_start < onset:
                gap_h = (onset - abs_end) / 3600
                too_close = True
                reason = (f"ends only {gap_h:.2f}h before seizure onset "
                          f"(need ≥{pre_seizure_gap/3600:.0f}h)")
                break
            # File overlaps with post-ictal buffer
            if abs_start < offset + post_seizure_gap and abs_end > offset:
                gap_h = (abs_start - offset) / 3600
                too_close = True
                reason = (f"starts only {gap_h:.2f}h after seizure offset "
                          f"(need ≥{post_seizure_gap/3600:.0f}h)")
                break

        if too_close:
            excluded_files.append(fname)
            if verbose:
                print(f"    [EXCL] {fname}: {reason}")
        else:
            clean_files.append(fname)

    return clean_files, excluded_files


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import DATA_ROOT, EXCLUDED_PATIENTS

    all_sz = parse_all_summaries(DATA_ROOT, EXCLUDED_PATIENTS)
    total_files    = sum(len(v) for v in all_sz.values())
    total_seizures = sum(len(sz) for v in all_sz.values() for sz in v.values())
    print(f"\nTotal patients  : {len(all_sz)}")
    print(f"Total EDF files : {total_files}")
    print(f"Total seizures  : {total_seizures}")

    # Demo: show clean interictal file counts for each patient
    print("\n── Clean interictal files (4h buffer) ──────────────────────────────")
    for pid in sorted(all_sz.keys()):
        pdir    = Path(DATA_ROOT) / pid
        try:
            tl      = parse_summary_with_times(str(pdir))
            clean, excl = get_clean_interictal_files(tl, verbose=False)
            n_total = len(tl)
            print(f"  {pid}: {len(clean):2d}/{n_total:2d} clean  "
                  f"({len(excl)} excluded)")
        except Exception as e:
            print(f"  {pid}: ERROR — {e}")
