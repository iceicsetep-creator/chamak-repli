#!/usr/bin/env python3
"""
Stage 1A: OASIS-1 dataset filesystem audit.

This script reports EXACTLY what is present on disk for the OASIS-1
cross-sectional dataset. It makes no assumptions, assigns no labels,
loads no volume data, and creates no output directories other than
writing the single report file next to this script.

It walks the OASIS-1 root, finds every `OAS1_XXXX_MR1` subject
directory, checks each subject for the atlas-registered brain-masked
volume file (`*_mpr_n*_anon_111_t88_masked_gfc.img`), confirms the
clinical spreadsheet exists, and compares the observed counts against
the numbers reported in the source paper (Section 3.1: 235 subjects
with complete clinical data).

Reference (paper, Section 3.1):
    "From the initial set of 416 subjects, we focused on individuals
     with complete clinical data, excluding 181 subjects lacking this
     information ... leaving 235 subjects for analysis."

Usage:
    python3 stage1A_audit.py \
        --oasis-root /path/to/OASIS-1 \
        --clinical-file /path/to/oasis_cross-sectional.csv
"""

import argparse
import datetime
import os
import platform
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants fixed by the stage specification. Do not change these here; they
# encode the expectations this audit is checking the filesystem against.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42                      # Seed that later stages will use.
PAPER_SUBJECT_COUNT = 235             # Paper Section 3.1: subjects analysed.
SUBJECT_DIR_PATTERN = re.compile(r"^OAS1_\d{4}_MR1$")
DISC_DIR_PATTERN = re.compile(r"^disc_?\d+$", re.IGNORECASE)
VOLUME_GLOB = "*_mpr_n*_anon_111_t88_masked_gfc.img"
REPORT_NAME = "stage1A_report.md"


def get_package_version(distribution_name):
    """Return an installed package version WITHOUT importing the package.

    Reads the installed distribution metadata via importlib.metadata, which
    does not execute the package's code. The spec requires that nibabel is
    only *checked*, never imported; this keeps that guarantee for every
    package we report on.
    """
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:  # pragma: no cover - Python < 3.8
        import importlib_metadata  # type: ignore
    try:
        return importlib_metadata.version(distribution_name)
    except Exception:
        return "NOT INSTALLED"


def scan_disc_directories(oasis_root):
    """Return the sorted list of disc directories directly discoverable
    under the OASIS root (e.g. disc1 ... disc12 or disc_01 ...)."""
    discs = []
    for dirpath, dirnames, _filenames in os.walk(oasis_root):
        for name in dirnames:
            if DISC_DIR_PATTERN.match(name):
                discs.append(os.path.join(dirpath, name))
    return sorted(discs)


def scan_subject_directories(oasis_root):
    """Walk the OASIS root and return every directory whose *name* matches
    OAS1_XXXX_MR1. Returns a list of (subject_id, full_path) sorted by id."""
    subjects = []
    for dirpath, dirnames, _filenames in os.walk(oasis_root):
        for name in dirnames:
            if SUBJECT_DIR_PATTERN.match(name):
                subjects.append((name, os.path.join(dirpath, name)))
    subjects.sort(key=lambda pair: pair[0])
    return subjects


def scan_volume_files(subject_path):
    """Recursively find volume files under a subject directory.

    Returns (list_of_full_paths, list_of_filenames), both sorted.
    """
    matches = sorted(str(p) for p in Path(subject_path).rglob(VOLUME_GLOB))
    filenames = [os.path.basename(p) for p in matches]
    return matches, filenames


def audit_clinical_file(clinical_path):
    """Confirm the clinical spreadsheet exists and is readable.

    The file is opened in binary mode and a single byte is read to prove
    readability; it is NOT parsed. Returns a dict describing what was found.
    """
    result = {
        "found": False,
        "filename": os.path.basename(clinical_path) if clinical_path else "",
        "size_bytes": None,
        "readable": False,
        "error": None,
    }
    if not clinical_path or not os.path.isfile(clinical_path):
        return result
    result["found"] = True
    try:
        result["size_bytes"] = os.path.getsize(clinical_path)
    except OSError as exc:
        result["error"] = f"size check failed: {exc}"
    try:
        with open(clinical_path, "rb") as handle:
            handle.read(1)  # touch the file without parsing it
        result["readable"] = True
    except OSError as exc:
        result["readable"] = False
        result["error"] = f"open failed: {exc}"
    return result


def build_report(oasis_root, clinical_path):
    """Run the full audit and return (report_text, stop_triggered)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    add = lines.append

    # -- Environment block (Step 5) -----------------------------------------
    add("# Stage 1A Audit Report")
    add("")
    add("## Environment")
    add("")
    add(f"- Python version: {platform.python_version()}")
    add(f"- nibabel version: {get_package_version('nibabel')}")
    add(f"- numpy version: {get_package_version('numpy')}")
    add(f"- pillow version: {get_package_version('pillow')}")
    add(f"- Operating system: {platform.platform()}")
    add(f"- Execution date and time: {now}")
    add(f"- OASIS root path: {oasis_root}")
    add(f"- Clinical file path: {clinical_path}")
    add(f"- Random seed that will be used: {RANDOM_SEED}")
    add("")
    add("```")

    # -- Header (Step 4) -----------------------------------------------------
    add("=== STAGE 1A: DATASET AUDIT ===")
    add(f"Date: {now}")
    add(f"OASIS Root: {oasis_root}")
    add(f"Clinical File: {clinical_path}")
    add("")

    root_exists = os.path.isdir(oasis_root)

    # -- Disc directories ----------------------------------------------------
    add("--- Disc Directories Found ---")
    disc_dirs = scan_disc_directories(oasis_root) if root_exists else []
    for disc in disc_dirs:
        add(disc)
    add(f"Total disc directories: {len(disc_dirs)}")
    add("")

    # -- Subject directories -------------------------------------------------
    subjects = scan_subject_directories(oasis_root) if root_exists else []
    add("--- Subject Directories ---")
    add(f"Total subject directories found: {len(subjects)}")
    add("")

    # -- Volume file audit ---------------------------------------------------
    exactly_one = []
    more_than_one = []   # list of (subject_id, [filenames])
    zero = []            # list of subject_id
    for subject_id, subject_path in subjects:
        matches, filenames = scan_volume_files(subject_path)
        if len(matches) == 1:
            exactly_one.append(subject_id)
        elif len(matches) > 1:
            more_than_one.append((subject_id, filenames))
        else:
            zero.append(subject_id)

    add("--- Volume File Audit ---")
    add(f"Subjects with exactly 1 volume file:  {len(exactly_one)}")
    add(f"Subjects with more than 1 volume file: {len(more_than_one)}"
        + ("  [see below]" if more_than_one else ""))
    for subject_id, filenames in more_than_one:
        add(f"    {subject_id}: {', '.join(filenames)}")
    add(f"Subjects with 0 volume files:          {len(zero)}"
        + ("  [see below]" if zero else ""))
    for subject_id in zero:
        add(f"    {subject_id}")
    add("")

    # -- Clinical file -------------------------------------------------------
    clinical = audit_clinical_file(clinical_path)
    add("--- Clinical File ---")
    add(f"File found: {'YES' if clinical['found'] else 'NO'}")
    add(f"Filename: {clinical['filename']}")
    add(f"Size: {clinical['size_bytes'] if clinical['size_bytes'] is not None else 'N/A'}")
    add(f"Readable: {'YES' if clinical['readable'] else 'NO'}")
    if clinical["error"]:
        add(f"Note: {clinical['error']}")
    add("")

    # -- Comparison to paper -------------------------------------------------
    subjects_with_volume = len(exactly_one) + len(more_than_one)
    dir_match = "YES" if len(subjects) == PAPER_SUBJECT_COUNT else "NO"
    vol_match = "YES" if subjects_with_volume == PAPER_SUBJECT_COUNT else "NO"
    add("--- COMPARISON TO PAPER ---")
    add("| Metric                          | Paper (§3.1) | Observed | Match |")
    add("|---------------------------------|--------------|----------|-------|")
    add(f"| Total subjects with directories | 235          | {len(subjects):<8} | {dir_match:<5} |")
    add(f"| Subjects with volume file found | 235          | {subjects_with_volume:<8} | {vol_match:<5} |")
    add("")

    # -- Stop conditions -----------------------------------------------------
    add("--- STOP CONDITIONS ---")
    stop_reasons = []
    if not root_exists:
        stop_reasons.append(
            f"OASIS root does not exist or is not a directory: {oasis_root}")
    if len(subjects) != PAPER_SUBJECT_COUNT:
        stop_reasons.append(
            f"Subject directory count is {len(subjects)}, expected "
            f"{PAPER_SUBJECT_COUNT}.")
    if subjects_with_volume != PAPER_SUBJECT_COUNT:
        stop_reasons.append(
            f"Subjects with a volume file is {subjects_with_volume}, "
            f"expected {PAPER_SUBJECT_COUNT}.")
    if zero:
        stop_reasons.append(
            f"{len(zero)} subject(s) have 0 volume files: "
            f"{', '.join(zero)}.")
    if not clinical["found"]:
        stop_reasons.append(
            f"Clinical file not found at: {clinical_path}")

    if stop_reasons:
        add("STOP")
        if len(subjects) != PAPER_SUBJECT_COUNT:
            add("STOP: Subject count mismatch")
            add(f"Exact count found: {len(subjects)}")
        for reason in stop_reasons:
            add(f"  - {reason}")
    else:
        add("PROCEED")

    add("```")

    return "\n".join(lines), bool(stop_reasons)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 1A: OASIS-1 dataset filesystem audit.")
    parser.add_argument(
        "--oasis-root", required=True,
        help="Path to the OASIS-1 root directory.")
    parser.add_argument(
        "--clinical-file", required=True,
        help="Path to the clinical spreadsheet file.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    oasis_root = os.path.abspath(os.path.expanduser(args.oasis_root))
    clinical_path = os.path.abspath(os.path.expanduser(args.clinical_file))

    report_text, stop_triggered = build_report(oasis_root, clinical_path)

    # Print to console.
    print(report_text)

    # Write the report next to this script (do not create new directories).
    report_path = Path(__file__).resolve().parent / REPORT_NAME
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n[stage1A_audit] Report written to: {report_path}")

    # Non-zero exit code communicates a STOP condition to any orchestrator.
    return 1 if stop_triggered else 0


if __name__ == "__main__":
    sys.exit(main())
