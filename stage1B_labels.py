#!/usr/bin/env python3
"""
Stage 1B: OASIS-1 clinical labels.

Self-contained script. Loads the OASIS-1 cross-sectional clinical
spreadsheet, assigns binary dementia labels from each subject's CDR
score, verifies the resulting counts against the paper (Fig. 1 / §3.1),
and saves a labeled subject list.

It performs no train/test/val splitting, loads no volume files, creates
no output directories, and makes no include/exclude decision beyond the
CDR rule specified below.

Binary labeling rule (Research Supervisor decision resolving paper §3.1:
"combining all dementia stages into a single category labeled 'Dementia'
while retaining 'No Dementia' as the other class"):
    CDR == 0.0  -> "No_Dementia"
    CDR >  0.0  -> "Dementia"

Permitted imports: pandas, os, sys, datetime, pathlib.
"""

import os
import sys
import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Fixed constants (per the stage specification).
# ---------------------------------------------------------------------------
CLINICAL_FILE = (
    "/kaggle/input/datasets/mdjulkarnainsiamaaub/cross-sectional-info1/"
    "oasis_cross-sectional-5708aa0a98d82080.xlsx"
)
WORKING_DIR = "/kaggle/working/"
CSV_NAME = "subject_labels.csv"
REPORT_NAME = "stage1B_report.md"

ID_COL = "ID"
CDR_COL = "CDR"

# Paper reference figures (Fig. 1 / §3.1).
EXPECT_TOTAL = 235
EXPECT_NO_DEMENTIA = 135
EXPECT_DEMENTIA = 100
EXPECT_VERY_MILD = 70   # CDR == 0.5
EXPECT_MILD = 28        # CDR == 1.0
EXPECT_MODERATE = 2     # CDR == 2.0


def out_path(name):
    """Path under /kaggle/working/ if it exists, else next to this script.
    Never creates a directory."""
    if os.path.isdir(WORKING_DIR):
        return os.path.join(WORKING_DIR, name)
    return str(Path(__file__).resolve().parent / name)


def write_report(lines):
    path = out_path(REPORT_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    add = lines.append

    add("=== STAGE 1B: CLINICAL LABELS ===")
    add(f"Executed: {now}")
    add("")
    add("--- Environment ---")
    add(f"Python: {sys.version.split()[0]}")
    add(f"Pandas: {pd.__version__}")
    add("")

    # -- Step 1: load the spreadsheet --------------------------------------
    print(f"Loading clinical spreadsheet: {CLINICAL_FILE}")
    df = pd.read_excel(CLINICAL_FILE)
    columns = list(df.columns)
    rows_loaded = len(df)
    print("Column names (exact):")
    for c in columns:
        print(f"  {repr(c)}")
    print(f"Total rows loaded: {rows_loaded}")

    add("--- Input ---")
    add(f"File: {os.path.basename(CLINICAL_FILE)}")
    add(f"Rows loaded: {rows_loaded}")
    add(f"Columns: {columns}")
    add("")

    # -- Step 2: identify required columns ---------------------------------
    if ID_COL not in columns or CDR_COL not in columns:
        print("Column names present in file:")
        for c in columns:
            print(f"  {repr(c)}")
        print("STOP: Required column not found")
        add("--- RESULT ---")
        add("[STOP: Required column not found]")
        add(f"Columns present: {columns}")
        report_path = write_report(lines)
        print(f"\n[stage1B] Report written to: {report_path}")
        print("\nSTAGE 1B RESULT: STOP — Required column not found")
        return 1

    # -- Step 3: exclude subjects without CDR ------------------------------
    cdr_numeric = pd.to_numeric(df[CDR_COL], errors="coerce")
    missing_mask = df[CDR_COL].isna() | cdr_numeric.isna()
    rows_removed = int(missing_mask.sum())
    kept = df.loc[~missing_mask].copy()
    kept[CDR_COL] = pd.to_numeric(kept[CDR_COL], errors="coerce")
    rows_remaining = len(kept)
    print(f"Rows removed (CDR missing): {rows_removed}")
    print(f"Rows remaining: {rows_remaining}")

    add("--- CDR Exclusion ---")
    add(f"Rows removed (CDR missing): {rows_removed}")
    add(f"Rows remaining: {rows_remaining}")
    add("")

    # -- Step 4: validate CDR values, then assign labels -------------------
    # Any negative or non-numeric CDR value remaining is unexpected.
    bad_mask = kept[CDR_COL].isna() | (kept[CDR_COL] < 0.0)
    if bool(bad_mask.any()):
        offending = kept.loc[bad_mask, [ID_COL, CDR_COL]]
        print("STOP: Unexpected CDR value")
        print(offending.to_string(index=False))
        add("--- RESULT ---")
        add("[STOP: Unexpected CDR value]")
        add("Offending rows:")
        for _, r in offending.iterrows():
            add(f"  {r[ID_COL]}: CDR={r[CDR_COL]}")
        report_path = write_report(lines)
        print(f"\n[stage1B] Report written to: {report_path}")
        print("\nSTAGE 1B RESULT: STOP — Unexpected CDR value")
        return 1

    labels = kept[CDR_COL].apply(
        lambda v: "No_Dementia" if v == 0.0 else "Dementia")

    # -- Step 5: labeled DataFrame (exact columns/order) -------------------
    labeled = pd.DataFrame({
        "subject_id": kept[ID_COL].astype(str).values,
        "CDR": kept[CDR_COL].values,
        "label": labels.values,
    })

    add("--- Binary Label Assignment ---")
    add("Rule: CDR == 0.0 → No_Dementia; CDR > 0.0 → Dementia")
    add("(Research Supervisor decision resolving paper §3.1 ambiguity)")
    add("")

    # -- CDR distribution --------------------------------------------------
    n_cdr0 = int((labeled["CDR"] == 0.0).sum())
    n_cdr05 = int((labeled["CDR"] == 0.5).sum())
    n_cdr1 = int((labeled["CDR"] == 1.0).sum())
    n_cdr2 = int((labeled["CDR"] == 2.0).sum())

    add("--- CDR Distribution ---")
    add(f"CDR = 0.0 (No_Dementia):  {n_cdr0} subjects")
    add(f"CDR = 0.5 (Very Mild):    {n_cdr05} subjects")
    add(f"CDR = 1.0 (Mild):         {n_cdr1} subjects")
    add(f"CDR = 2.0 (Moderate):     {n_cdr2} subjects")
    add("")

    # -- Step 6: verification checks ---------------------------------------
    total = len(labeled)
    n_no_dementia = int((labeled["label"] == "No_Dementia").sum())
    n_dementia = int((labeled["label"] == "Dementia").sum())
    dup_ids = labeled["subject_id"][labeled["subject_id"].duplicated(keep=False)]
    dup_list = sorted(set(dup_ids.tolist()))

    check1 = total == EXPECT_TOTAL
    check2 = n_no_dementia == EXPECT_NO_DEMENTIA
    check3 = n_dementia == EXPECT_DEMENTIA
    check5 = len(dup_list) == 0

    # Check 4: sub-group warnings (informational only, never stop).
    subgroup_warnings = []
    if n_cdr05 != EXPECT_VERY_MILD:
        subgroup_warnings.append(
            f"Very Mild (CDR=0.5) count = {n_cdr05}, expected {EXPECT_VERY_MILD}")
    if n_cdr1 != EXPECT_MILD:
        subgroup_warnings.append(
            f"Mild (CDR=1.0) count = {n_cdr1}, expected {EXPECT_MILD}")
    if n_cdr2 != EXPECT_MODERATE:
        subgroup_warnings.append(
            f"Moderate (CDR=2.0) count = {n_cdr2}, expected {EXPECT_MODERATE}")

    # Console echo of checks (in order).
    print("\n--- Verification checks ---")
    if not check1:
        print(f"STOP: Subject count after CDR filter = {total}, expected 235")
    if not check2:
        print(f"STOP: No_Dementia count = {n_no_dementia}, expected 135")
    if not check3:
        print(f"STOP: Dementia count = {n_dementia}, expected 100")
    for w in subgroup_warnings:
        print(f"WARNING: {w}")
    if not check5:
        print("STOP: Duplicate subject IDs found")
        for d in dup_list:
            print(f"  {d}")

    # -- Paper comparison table --------------------------------------------
    def yn(ok):
        return "YES" if ok else "NO"

    add("--- PAPER COMPARISON TABLE ---")
    add("| Metric                  | Paper (Fig.1 / §3.1) | Observed | Match  |")
    add("|-------------------------|----------------------|----------|--------|")
    add(f"| Total after CDR filter  | 235                  | {total:<8} | {yn(check1):<6} |")
    add(f"| No_Dementia (CDR=0.0)   | 135                  | {n_no_dementia:<8} | {yn(check2):<6} |")
    add(f"| Dementia (CDR>0)        | 100                  | {n_dementia:<8} | {yn(check3):<6} |")
    add(f"| Very Mild (CDR=0.5)     | 70                   | {n_cdr05:<8} | {yn(n_cdr05 == EXPECT_VERY_MILD):<6} |")
    add(f"| Mild (CDR=1.0)          | 28                   | {n_cdr1:<8} | {yn(n_cdr1 == EXPECT_MILD):<6} |")
    add(f"| Moderate (CDR=2.0)      | 2                    | {n_cdr2:<8} | {yn(n_cdr2 == EXPECT_MODERATE):<6} |")
    add("")

    add("--- Verification ---")
    add(f"Total count check (235):     {'PASSED' if check1 else 'FAILED'}")
    add(f"No_Dementia count (135):     {'PASSED' if check2 else 'FAILED'}")
    add(f"Dementia count (100):        {'PASSED' if check3 else 'FAILED'}")
    add(f"Duplicate subject ID check:  {'PASSED' if check5 else 'FAILED'}")
    if subgroup_warnings:
        add("Sub-group warnings (informational):")
        for w in subgroup_warnings:
            add(f"  - {w}")
    add("")

    # -- Determine STOP reason (first failing check, in order) -------------
    stop_reason = None
    if not check1:
        stop_reason = f"Subject count after CDR filter = {total}, expected 235"
    elif not check2:
        stop_reason = f"No_Dementia count = {n_no_dementia}, expected 135"
    elif not check3:
        stop_reason = f"Dementia count = {n_dementia}, expected 100"
    elif not check5:
        stop_reason = "Duplicate subject IDs found"

    # -- Step 7: outputs ----------------------------------------------------
    if stop_reason is None:
        csv_path = out_path(CSV_NAME)
        labeled.to_csv(csv_path, index=False)
        add("--- Output ---")
        add(f"Saved: {csv_path} ({len(labeled)} rows)")
        add("")
        add("--- RESULT ---")
        add("[PROCEED]")
        report_path = write_report(lines)
        print(f"\nSaved: {csv_path} ({len(labeled)} rows)")
        print(f"[stage1B] Report written to: {report_path}")
        print("\nSTAGE 1B RESULT: PROCEED")
        return 0
    else:
        # Verification failed: do NOT save subject_labels.csv.
        add("--- Output ---")
        add("subject_labels.csv NOT saved (verification failed)")
        add("")
        add("--- RESULT ---")
        add(f"[STOP: {stop_reason}]")
        report_path = write_report(lines)
        print(f"\n[stage1B] Report written to: {report_path}")
        print(f"\nSTAGE 1B RESULT: STOP — {stop_reason}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
