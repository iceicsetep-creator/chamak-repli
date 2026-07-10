#!/usr/bin/env python3
"""
Stage 1E: extraction verification.

Read-only integrity check of the PNG slice tree produced in Stage 1D.
Writes nothing except console output. Loads no volume files, creates no
.md files, modifies/moves/deletes nothing, and makes no scientific
decisions about failures -- it only reports.
"""

import os
import re
import sys
import random

from PIL import Image

SLICES_ROOT = "/kaggle/working/slices/"
SPLITS_CSV = "/kaggle/working/subject_splits.csv"

PLANES = ["sagittal", "coronal", "axial"]
SPLITS = ["train", "val", "test"]
LABELS = ["Dementia", "No_Dementia"]
SLICES_PER_PLANE = {"sagittal": 5, "coronal": 7, "axial": 7}

EXPECT_TOTAL = 4465
EXPECT_SAGITTAL = 235 * 5   # 1175
EXPECT_CORONAL = 235 * 7    # 1645
EXPECT_AXIAL = 235 * 7      # 1645

FILENAME_RE = re.compile(
    r"^OAS1_\d{4}_MR1_(sagittal|coronal|axial)_\d{3}\.png$")


def leaf_dir(plane, split, label):
    return os.path.join(SLICES_ROOT, plane, split, label)


def list_pngs(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".png"))


def load_split_counts():
    """Count subjects per (split, label) from the Stage 1C CSV."""
    counts = {(s, l): 0 for s in SPLITS for l in LABELS}
    with open(SPLITS_CSV, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        i_label = header.index("label")
        i_split = header.index("split")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            key = (parts[i_split], parts[i_label])
            if key in counts:
                counts[key] += 1
    return counts


def main():
    print("=== STAGE 1E: EXTRACTION VERIFICATION ===")
    print()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Pillow: {Image.__version__}")
    print()

    stop_reasons = []
    warnings = []

    subject_counts = load_split_counts()

    # -- Step 1: directory file counts -------------------------------------
    print("--- Directory File Counts ---")
    print("Plane      | Split | Label        | Expected | Observed | Status")
    print("-----------|-------|--------------|----------|----------|--------")
    mismatched_dirs = []
    file_counts = {}
    for plane in PLANES:
        for split in SPLITS:
            for label in LABELS:
                expected = subject_counts[(split, label)] * SLICES_PER_PLANE[plane]
                observed = len(list_pngs(leaf_dir(plane, split, label)))
                file_counts[(plane, split, label)] = observed
                status = "MATCH" if observed == expected else "MISMATCH"
                if status == "MISMATCH":
                    mismatched_dirs.append(
                        f"{plane}/{split}/{label} (exp {expected}, obs {observed})")
                print(f"{plane:<10} | {split:<5} | {label:<12} | "
                      f"{expected:<8} | {observed:<8} | {status}")
    if mismatched_dirs:
        print("STOP: Directory count mismatch")
        for d in mismatched_dirs:
            print(f"  {d}")
        stop_reasons.append(f"{len(mismatched_dirs)} directory count mismatch(es)")
    print()

    # -- Step 2: subject leakage check -------------------------------------
    def subject_ids_in(splits):
        ids = set()
        for plane in PLANES:
            for split in splits:
                for label in LABELS:
                    for f in list_pngs(leaf_dir(plane, split, label)):
                        ids.add(f[:13])  # OAS1_XXXX_MR1
        return ids

    test_ids = subject_ids_in(["test"])
    train_ids = subject_ids_in(["train"])
    val_ids = subject_ids_in(["val"])
    leak_train = sorted(test_ids & train_ids)
    leak_val = sorted(test_ids & val_ids)
    total_leak = sorted(set(leak_train) | set(leak_val))

    print("--- Subject Leakage Check ---")
    print(f"Test subject IDs extracted: {len(test_ids)}")
    print(f"Any appear in train dirs: {len(leak_train)}  "
          f"[{'PASSED' if not leak_train else 'STOP'}]")
    print(f"Any appear in val dirs:   {len(leak_val)}  "
          f"[{'PASSED' if not leak_val else 'STOP'}]")
    if total_leak:
        print(f"STOP: Leakage detected — {', '.join(total_leak)}")
        stop_reasons.append("subject leakage detected")
    print()

    # -- Step 3: filename format check -------------------------------------
    print("--- Filename Format Samples ---")
    malformed = []
    for plane in PLANES:
        # Validate every filename in this plane; collect examples.
        examples = []
        for split in SPLITS:
            for label in LABELS:
                for f in list_pngs(leaf_dir(plane, split, label)):
                    if not FILENAME_RE.match(f):
                        malformed.append(f"{plane}/{split}/{label}/{f}")
                    if len(examples) < 3:
                        examples.append(f)
        print(f"{plane} examples:")
        for ex in examples:
            print(f"  {ex}")
    if malformed:
        print("Filename format check: FAILED")
        for m in malformed[:20]:
            print(f"  malformed: {m}")
        stop_reasons.append(f"{len(malformed)} malformed filename(s)")
    else:
        print("Filename format check: PASSED")
    print()

    # -- Step 4: image property sampling (1 per directory) -----------------
    print("--- Image Property Samples (1 per directory, 18 total) ---")
    for plane in PLANES:
        for split in SPLITS:
            for label in LABELS:
                d = leaf_dir(plane, split, label)
                files = list_pngs(d)
                cell = f"{plane}/{split}/{label}:"
                if not files:
                    print(f"{cell:<28} (no files)")
                    continue
                sample = random.choice(files)
                with Image.open(os.path.join(d, sample)) as im:
                    im.load()
                    w, h = im.size
                    mode = im.mode
                    lo, hi = im.getextrema()
                flag = "OK"
                if mode != "L":
                    flag = "WARNING"
                    warnings.append(f"Non-grayscale image found in {d} ({mode})")
                if hi == 0:
                    flag = "STOP"
                    stop_reasons.append(f"Zero-only image detected in {d}")
                print(f"{cell:<28} size=[{w}x{h}]  mode={mode}  "
                      f"min={lo}  max={hi}  [{flag}]")
    print()
    for w in warnings:
        print(f"WARNING: {w}")
    for r in stop_reasons:
        if r.startswith("Zero-only"):
            print(f"STOP: {r}")

    # -- Step 5: total count verification ----------------------------------
    total_observed = sum(file_counts.values())
    sagittal_obs = sum(v for (p, _, _), v in file_counts.items() if p == "sagittal")
    coronal_obs = sum(v for (p, _, _), v in file_counts.items() if p == "coronal")
    axial_obs = sum(v for (p, _, _), v in file_counts.items() if p == "axial")

    print("--- Total Count ---")
    print(f"Expected: {EXPECT_TOTAL}")
    print(f"Observed: {total_observed}")
    print(f"Status:   {'MATCH' if total_observed == EXPECT_TOTAL else 'MISMATCH'}")
    print()

    # -- Paper comparison table --------------------------------------------
    def yn(ok):
        return "YES" if ok else "NO"

    print("--- PAPER COMPARISON TABLE ---")
    print("| Metric                  | Paper §3.1   | Expected | Observed | Match  |")
    print("|-------------------------|--------------|----------|----------|--------|")
    print(f"| Total slices            | 235 × 19     | 4465     | {total_observed:<8} | {yn(total_observed == EXPECT_TOTAL):<6} |")
    print(f"| Sagittal slices         | 235 × 5      | 1175     | {sagittal_obs:<8} | {yn(sagittal_obs == EXPECT_SAGITTAL):<6} |")
    print(f"| Coronal slices          | 235 × 7      | 1645     | {coronal_obs:<8} | {yn(coronal_obs == EXPECT_CORONAL):<6} |")
    print(f"| Axial slices            | 235 × 7      | 1645     | {axial_obs:<8} | {yn(axial_obs == EXPECT_AXIAL):<6} |")
    print(f"| Subject leakage         | §3.2         | None     | {str(len(total_leak)) + ' found':<8} | {yn(len(total_leak) == 0):<6} |")
    print()

    if stop_reasons:
        reason = "; ".join(stop_reasons)
        print(f"=== STAGE 1E RESULT: STOP — {reason} ===")
        return 1
    print("=== STAGE 1E RESULT: PROCEED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
