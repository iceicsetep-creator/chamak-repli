#!/usr/bin/env python3
"""
Stage 1C: subject-level stratified three-way split.

Splits the 235 labeled subjects from Stage 1B into train / val / test at
the subject level, stratified by label, and saves a single CSV. Output is
to the console only (no .md files). No volume loading, no slice
extraction, no model decisions.

Research Supervisor decisions implemented exactly (do not modify):
    First split : test_size=0.30, stratify=label, random_state=42
    Second split: test_size=0.20, stratify=label, random_state=42
    (both via sklearn.model_selection.train_test_split)
"""

import sys

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

INPUT_CSV = "/kaggle/working/subject_labels.csv"
OUTPUT_CSV = "/kaggle/working/subject_splits.csv"

EXPECTED_COLUMNS = ["subject_id", "CDR", "label"]
FINAL_COLUMNS = ["subject_id", "CDR", "label", "split"]

# Paper reference figures.
PAPER_TEST = 71          # Fig. 5 (32 + 39)
PAPER_TRAINVAL = 164     # §5.1 (70%)
PAPER_TOTAL = 235        # §3.1


def stop(reason):
    print(f"STOP: {reason}")
    print(f"\n=== STAGE 1C RESULT: STOP — {reason} ===")
    sys.exit(1)


def main():
    # -- Step 1: environment and seed confirmation -------------------------
    print("=== STAGE 1C: SUBJECT SPLIT ===")
    print()
    print(f"Python: {sys.version.split()[0]}")
    print(f"scikit-learn: {sklearn.__version__}")
    print("Random seed: 42 — CONFIRMED")
    print()

    # -- Step 2: load and confirm input ------------------------------------
    df = pd.read_csv(INPUT_CSV)
    if df.shape != (235, 3) or list(df.columns) != EXPECTED_COLUMNS:
        print(f"Loaded shape: {df.shape}, columns: {list(df.columns)}")
        stop("Unexpected input shape or columns")

    n_no_dementia = int((df["label"] == "No_Dementia").sum())
    n_dementia = int((df["label"] == "Dementia").sum())
    total = len(df)

    # -- Step 3: first split (trainval / test) -----------------------------
    trainval_df, test_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42)

    # -- Step 4: second split (train / val) --------------------------------
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.20, stratify=trainval_df["label"],
        random_state=42)

    # -- Step 5: add split column and combine ------------------------------
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined[FINAL_COLUMNS].sort_values(
        "subject_id").reset_index(drop=True)

    # -- Step 6: verification checks (in order) ----------------------------
    train_ids = set(train_df["subject_id"])
    val_ids = set(val_df["subject_id"])
    test_ids = set(test_df["subject_id"])

    print("--- Verification Checks ---")

    # Check 1: total rows == 235
    if len(combined) != 235:
        print(f"Check 1 (total rows == 235): FAILED (got {len(combined)})")
        stop(f"Total rows = {len(combined)}, expected 235")
    print("Check 1 (total rows == 235): PASSED")

    # Check 2: train ∩ val empty
    tv = train_ids & val_ids
    if tv:
        print(f"Check 2 (train ∩ val empty): FAILED ({len(tv)} overlap)")
        stop("train ∩ val is not empty")
    print("Check 2 (train ∩ val empty): PASSED")

    # Check 3: train ∩ test empty
    tt = train_ids & test_ids
    if tt:
        print(f"Check 3 (train ∩ test empty): FAILED ({len(tt)} overlap)")
        stop("train ∩ test is not empty")
    print("Check 3 (train ∩ test empty): PASSED")

    # Check 4: val ∩ test empty
    vt = val_ids & test_ids
    if vt:
        print(f"Check 4 (val ∩ test empty): FAILED ({len(vt)} overlap)")
        stop("val ∩ test is not empty")
    print("Check 4 (val ∩ test empty): PASSED")

    # Check 5: both classes present in every split
    def classes(d):
        return set(d["label"].unique())
    required = {"Dementia", "No_Dementia"}
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        if not required.issubset(classes(d)):
            print(f"Check 5 (class coverage): FAILED in {name} split")
            stop(f"Missing a class in {name} split")
    print("Check 5 (class coverage all splits): PASSED")

    # Check 6: INFO only — test size vs paper Fig. 5 (71)
    test_size = len(test_df)
    if test_size != PAPER_TEST:
        print(f"WARNING: Test size = {test_size}, paper reports 71")
    else:
        print(f"Check 6 (test size == 71, INFO): matches paper ({test_size})")
    print()

    # -- Step 7: save ------------------------------------------------------
    combined.to_csv(OUTPUT_CSV, index=False)

    # -- Step 8: full console output ---------------------------------------
    def pct(part, whole):
        return (100.0 * part / whole) if whole else 0.0

    def split_line(label, d):
        n = len(d)
        dem = int((d["label"] == "Dementia").sum())
        nod = int((d["label"] == "No_Dementia").sum())
        return (f"{label:<15} {n:<7} "
                f"{f'{dem} ({pct(dem, n):.1f}%)':<15} "
                f"{nod} ({pct(nod, n):.1f}%)")

    trainval_n = len(train_df) + len(val_df)
    obs_test = test_size
    obs_trainval = trainval_n
    obs_total = len(combined)

    def yn(ok):
        return "YES" if ok else "NO"

    print("=== STAGE 1C: SUBJECT SPLIT ===")
    print()
    print(f"Python: {sys.version.split()[0]}")
    print(f"scikit-learn: {sklearn.__version__}")
    print("Random seed: 42 — CONFIRMED")
    print()
    print(f"Input: {total} subjects  |  "
          f"No_Dementia: {n_no_dementia} ({pct(n_no_dementia, total):.1f}%)  |  "
          f"Dementia: {n_dementia} ({pct(n_dementia, total):.1f}%)")
    print()
    print("--- Split Results ---")
    print(f"{'':<15} {'Count':<7} {'Dementia':<15} {'No_Dementia'}")
    print(split_line("Train:", train_df))
    print(split_line("Val:", val_df))
    print(split_line("Test:", test_df))
    print(f"{'Total:':<15} {len(combined)}")
    print()
    print("--- Disjointness Checks ---")
    print(f"{'Train ∩ Val:':<15} {len(train_ids & val_ids)} subjects  [PASSED]")
    print(f"{'Train ∩ Test:':<15} {len(train_ids & test_ids)} subjects  [PASSED]")
    print(f"{'Val ∩ Test:':<15} {len(val_ids & test_ids)} subjects  [PASSED]")
    print()
    print("--- Class Coverage ---")
    for label, d in (("Train:", train_df), ("Val:", val_df), ("Test:", test_df)):
        cls = classes(d)
        dm = "✓" if "Dementia" in cls else "✗"
        nd = "✓" if "No_Dementia" in cls else "✗"
        print(f"{label:<7} Dementia {dm}  No_Dementia {nd}  [PASSED]")
    print()
    print("--- PAPER COMPARISON TABLE ---")
    print("| Metric              | Paper Ref       | Expected | Observed | Match  |")
    print("|---------------------|-----------------|----------|----------|--------|")
    print(f"| Test subjects       | Fig.5 (32+39)   | 71       | {obs_test:<8} | {yn(obs_test == PAPER_TEST):<6} |")
    print(f"| Train+Val subjects  | §5.1 (70%)      | 164      | {obs_trainval:<8} | {yn(obs_trainval == PAPER_TRAINVAL):<6} |")
    print(f"| Total subjects      | §3.1            | 235      | {obs_total:<8} | {yn(obs_total == PAPER_TOTAL):<6} |")
    print()
    print("--- Output ---")
    print(f"Saved: {OUTPUT_CSV}")
    print("Columns: subject_id, CDR, label, split")
    print(f"Rows: {len(combined)}")
    print("Sorted by: subject_id")
    print()
    print("=== STAGE 1C RESULT: PROCEED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
