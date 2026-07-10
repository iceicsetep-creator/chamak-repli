#!/usr/bin/env python3
"""
Stage 1A: OASIS-1 dataset filesystem audit (Kaggle edition).

Self-contained audit script. It reports EXACTLY what is present on the
Kaggle filesystem for the OASIS-1 cross-sectional dataset. It makes no
assumptions, assigns no labels, loads no volume data, and creates no
output directories.

Paper reference figures (for the comparison table):
    - 416 subjects in the OASIS-1 cross-sectional dataset
    - 235 subjects with complete clinical data (used for analysis)
    - 12 disc directories

The script runs as a Kaggle notebook cell or as a standalone script.
Output is printed to stdout and written identically to
`stage1A_report.md` in `/kaggle/working/`.

Standard library only (os, glob, pathlib, datetime, plus sys / platform /
importlib.metadata for version reporting -- all part of the stdlib).
"""

import os
import glob
import datetime
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed paths for the Kaggle datasets (per the stage specification).
# ---------------------------------------------------------------------------
CLINICAL_ROOT = "/kaggle/input/datasets/mdjulkarnainsiamaaub/cross-sectional-info1/"
OASIS_ROOT = "/kaggle/input/datasets/mdjulkarnainsiamaaub/oasis1original/"
WORKING_DIR = "/kaggle/working/"
REPORT_NAME = "stage1A_report.md"

# Paper figures.
PAPER_TOTAL_SUBJECTS = 416
PAPER_COMPLETE_DATA = 235
PAPER_DISCS = 12

# STOP thresholds (per the stage specification).
MIN_SUBJECT_DIRS = 200          # < 200 -> STOP (severe dataset problem)
MAX_ZERO_VOLUME_SUBJECTS = 10   # > 10 subjects with 0 volumes -> STOP

VOLUME_GLOB = "*_mpr_n*_anon_111_t88_masked_gfc.img"


# ---------------------------------------------------------------------------
# Helpers (paths are passed in so the audit logic is testable; main() calls
# them with the fixed Kaggle constants above).
# ---------------------------------------------------------------------------
def numpy_version():
    """Report the installed NumPy version WITHOUT importing NumPy."""
    try:
        return importlib_metadata.version("numpy")
    except Exception:
        return "NOT INSTALLED"


def find_clinical_files(clinical_root):
    """Return sorted [(full_path, size_bytes), ...] for every .xlsx/.csv
    file found anywhere under clinical_root."""
    found = []
    if os.path.isdir(clinical_root):
        for dirpath, _dirnames, filenames in os.walk(clinical_root):
            for name in filenames:
                lower = name.lower()
                if lower.endswith(".xlsx") or lower.endswith(".csv"):
                    full = os.path.join(dirpath, name)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        size = -1
                    found.append((full, size))
    found.sort()
    return found


def find_disc_directories(oasis_root):
    """Return sorted full paths of every directory whose name starts with
    'disc' (case-insensitive), anywhere under oasis_root."""
    discs = []
    if os.path.isdir(oasis_root):
        for dirpath, dirnames, _filenames in os.walk(oasis_root):
            for name in dirnames:
                if name.lower().startswith("disc"):
                    discs.append(os.path.join(dirpath, name))
    return sorted(discs)


def _is_subject_name(name):
    """True if name matches OAS1_XXXX_MR1 with exactly 4 digits."""
    parts = name.split("_")
    return (
        len(parts) == 3
        and parts[0] == "OAS1"
        and len(parts[1]) == 4
        and parts[1].isdigit()
        and parts[2] == "MR1"
    )


def find_subjects_in_disc(disc_path):
    """Return sorted [(subject_id, full_path), ...] for every OAS1_XXXX_MR1
    directory found (recursively) under a single disc directory."""
    subjects = []
    for dirpath, dirnames, _filenames in os.walk(disc_path):
        for name in dirnames:
            if _is_subject_name(name):
                subjects.append((name, os.path.join(dirpath, name)))
    subjects.sort()
    return subjects


def find_volume_files(subject_path):
    """Return sorted list of full paths of volume files under a subject."""
    return sorted(str(p) for p in Path(subject_path).rglob(VOLUME_GLOB))


def suffix_of(filename):
    """Extract the preprocessing suffix (n3/n4/n5/n6/...) from a volume
    filename, or None if it cannot be determined."""
    base = os.path.basename(filename)
    marker = "_mpr_"
    idx = base.find(marker)
    if idx == -1:
        return None
    rest = base[idx + len(marker):]           # e.g. "n4_anon_111_..."
    token = rest.split("_", 1)[0]             # e.g. "n4"
    if len(token) >= 2 and token[0] == "n" and token[1:].isdigit():
        return token
    return None


# ---------------------------------------------------------------------------
# Report builder.
# ---------------------------------------------------------------------------
def run_audit(clinical_root, oasis_root):
    """Execute the audit and return (report_text, result_line).

    result_line is either "PROCEED ..." or "STOP — ...". If the clinical
    file is missing this returns an early STOP with a minimal report.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    on_kaggle = os.path.isdir("/kaggle")

    lines = []
    add = lines.append

    add("=== STAGE 1A: DATASET AUDIT ===")
    add(f"Executed: {now}")
    add("")
    add("--- Environment ---")
    add(f"Python: {platform.python_version()}")
    add(f"NumPy: {numpy_version()}")
    add(f"OS: {platform.platform()}")
    add(f"Kaggle: {on_kaggle}")
    add("")
    add("--- Paths ---")
    add(f"Clinical file root: {clinical_root}")
    add(f"OASIS root:        {oasis_root}")
    add("")

    # -- Step 1: clinical file ---------------------------------------------
    clinical_files = find_clinical_files(clinical_root)
    add("--- Clinical File ---")
    add(f"Files found: {len(clinical_files)}")
    for full, size in clinical_files:
        add(f"  {full}  ({size} bytes)")
    add("")

    if len(clinical_files) == 0:
        # Early STOP required by the spec.
        add("--- RESULT ---")
        add("[STOP: Clinical file not found]")
        return "\n".join(lines), "STOP — Clinical file not found"

    # -- Step 2: disc directories ------------------------------------------
    disc_dirs = find_disc_directories(oasis_root)
    add("--- Disc Directories Found ---")
    for disc in disc_dirs:
        add(disc)
    add(f"Total disc directories: {len(disc_dirs)}")
    add("")

    # -- Step 3: subject directories ---------------------------------------
    # Map subject_id -> disc path, and detect empty discs.
    subject_to_disc = {}
    subject_to_path = {}
    disc_subject_counts = {}
    empty_discs = []
    for disc in disc_dirs:
        subs = find_subjects_in_disc(disc)
        disc_subject_counts[disc] = len(subs)
        # "Empty" disc = truly empty on disk OR contains no subject dirs.
        try:
            truly_empty = len(os.listdir(disc)) == 0
        except OSError:
            truly_empty = True
        if truly_empty or len(subs) == 0:
            empty_discs.append(disc)
        for sid, spath in subs:
            # First disc wins if a subject id somehow appears twice.
            if sid not in subject_to_disc:
                subject_to_disc[sid] = disc
                subject_to_path[sid] = spath

    all_subject_ids = sorted(subject_to_disc.keys())
    add("--- Subject Directory Audit ---")
    add(f"Total unique subject IDs found: {len(all_subject_ids)}")
    for sid in all_subject_ids:
        add(f"  {sid}  [{subject_to_disc[sid]}]")
    add("")

    # -- Step 4: volume files ----------------------------------------------
    exactly_one = []
    more_than_one = []   # (sid, [filenames])
    zero = []            # sid
    suffix_subjects = {}  # suffix -> set(subject_ids)

    for sid in all_subject_ids:
        vols = find_volume_files(subject_to_path[sid])
        filenames = [os.path.basename(v) for v in vols]
        if len(vols) == 1:
            exactly_one.append(sid)
        elif len(vols) > 1:
            more_than_one.append((sid, filenames))
        else:
            zero.append(sid)
        for fn in filenames:
            suf = suffix_of(fn)
            if suf is not None:
                suffix_subjects.setdefault(suf, set()).add(sid)

    add("--- Volume File Audit ---")
    add(f"Subjects with exactly 1 volume file:   {len(exactly_one)}")
    add(f"Subjects with more than 1 volume file: {len(more_than_one)}")
    for sid, filenames in more_than_one:
        add(f"  {sid}: {', '.join(filenames)}")
    add(f"Subjects with 0 volume files:          {len(zero)}")
    for sid in zero:
        add(f"  {sid}")
    add("")

    # -- Preprocessing suffix distribution ---------------------------------
    add("--- Preprocessing Suffix Distribution ---")
    for suf in ("n3", "n4", "n5", "n6"):
        count = len(suffix_subjects.get(suf, set()))
        add(f"{suf} files found: {count} subjects")
    # Report any other suffixes that turned up, so nothing is hidden.
    other = sorted(s for s in suffix_subjects if s not in ("n3", "n4", "n5", "n6"))
    for suf in other:
        add(f"{suf} files found: {len(suffix_subjects[suf])} subjects")
    add("[Note: one subject may appear in multiple rows if it has multiple variants]")
    add("")

    # -- Paper comparison table --------------------------------------------
    total_dirs = len(all_subject_ids)
    subjects_with_volume = len(exactly_one) + len(more_than_one)
    total_match = "YES" if total_dirs == PAPER_TOTAL_SUBJECTS else "NO"
    vol_match = "YES" if subjects_with_volume == PAPER_COMPLETE_DATA else "NO"
    disc_match = "YES" if len(disc_dirs) == PAPER_DISCS else "NO"

    add("--- PAPER COMPARISON TABLE ---")
    add("| Metric                            | Paper     | Observed | Match  |")
    add("|-----------------------------------|-----------|----------|--------|")
    add(f"| Total subjects in dataset         | 416       | {total_dirs:<8} | {total_match:<6} |")
    add(f"| Subjects with complete data       | 235       | {'*':<8} | {'*':<6} |")
    add(f"| Subject dirs with volume file     | 235       | {subjects_with_volume:<8} | {vol_match:<6} |")
    add(f"| Disc directories                  | 12        | {len(disc_dirs):<8} | {disc_match:<6} |")
    add("")
    add("* Cannot verify until Stage 1B loads clinical data. Report total subject dirs found.")
    add("")

    # -- Result / STOP conditions ------------------------------------------
    stop_reasons = []
    if total_dirs < MIN_SUBJECT_DIRS:
        stop_reasons.append(
            f"Total subject directories found ({total_dirs}) < {MIN_SUBJECT_DIRS}")
    if len(zero) > MAX_ZERO_VOLUME_SUBJECTS:
        stop_reasons.append(
            f"Subjects with 0 volume files ({len(zero)}) > {MAX_ZERO_VOLUME_SUBJECTS}")
    if empty_discs:
        stop_reasons.append(
            f"{len(empty_discs)} disc directory(ies) appear empty: "
            + ", ".join(empty_discs))

    warnings = []
    if more_than_one:
        warnings.append(
            f"{len(more_than_one)} subject(s) have multiple volume variants")
    if len(disc_dirs) != PAPER_DISCS:
        warnings.append(
            f"Disc directory count is {len(disc_dirs)}, paper reports {PAPER_DISCS}")
    if zero:
        warnings.append(
            f"{len(zero)} subject(s) have 0 volume files (below STOP threshold)")

    add("--- RESULT ---")
    if stop_reasons:
        reason = "; ".join(stop_reasons)
        add(f"[STOP: {reason}]")
        result_line = f"STOP — {reason}"
    else:
        add("[PROCEED]")
        if warnings:
            add("Warnings:")
            for w in warnings:
                add(f"  - {w}")
        result_line = "PROCEED"

    return "\n".join(lines), result_line


def write_report(report_text):
    """Write the report to /kaggle/working/ if available, else next to the
    script. Never creates a directory."""
    if os.path.isdir(WORKING_DIR):
        report_path = os.path.join(WORKING_DIR, REPORT_NAME)
    else:
        report_path = str(Path(__file__).resolve().parent / REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text + "\n")
    return report_path


def main():
    report_text, result_line = run_audit(CLINICAL_ROOT, OASIS_ROOT)

    # Print the full report to the console (notebook output).
    print(report_text)

    # Write the identical report to disk.
    report_path = write_report(report_text)
    print(f"\n[stage1A_audit] Report written to: {report_path}")

    # Final mandatory result line.
    if result_line == "PROCEED":
        print("\nSTAGE 1A RESULT: PROCEED")
        return 0
    else:
        print(f"\nSTAGE 1A RESULT: STOP — {result_line[len('STOP — '):]}"
              if result_line.startswith("STOP — ")
              else f"\nSTAGE 1A RESULT: {result_line}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
