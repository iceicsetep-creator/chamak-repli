#!/usr/bin/env python3
"""
Stage 1D: 2D slice extraction from OASIS-1 brain-masked atlas-registered
volumes across the sagittal, coronal, and axial planes, saved as
single-channel PNGs in a Keras flow_from_directory layout.

Implements locked Research Supervisor decisions D-01, D-02, D-03, D-08
exactly. No resizing, no augmentation, no model logic, no .md files.
"""

import os
import sys
import fnmatch
import random

import numpy as np
import nibabel as nib
from PIL import Image

# ---------------------------------------------------------------------------
# Fixed paths and parameters (per the stage specification).
# ---------------------------------------------------------------------------
SPLITS_CSV = "/kaggle/working/subject_splits.csv"
OASIS_ROOT = "/kaggle/input/datasets/mdjulkarnainsiamaaub/oasis1original/"
SLICES_ROOT = "/kaggle/working/slices/"

VOLUME_PATTERN = "*_mpr_n*_anon_111_t88_masked_gfc.img"  # D-01

# D-02 axis mapping + slice index ranges (paper §3.1).
PLANES = {
    "sagittal": (range(78, 99, 5), lambda v, i: v[i, :, :]),   # 5 slices
    "coronal":  (range(75, 106, 5), lambda v, j: v[:, j, :]),  # 7 slices
    "axial":    (range(73, 104, 5), lambda v, k: v[:, :, k]),  # 7 slices
}
SPLITS = ["train", "val", "test"]
LABELS = ["Dementia", "No_Dementia"]

EXPECT_PER_PLANE = {"sagittal": 5, "coronal": 7, "axial": 7}
EXPECT_TOTAL_SLICES = 4465
EXPECT_SUBJECTS = 235


def load_split_assignments():
    """Step 1: map subject_id -> (label, split) from the split CSV.

    Reads the CSV directly (only to obtain subject IDs and split
    assignments, as permitted). No pandas dependency required.
    """
    mapping = {}
    with open(SPLITS_CSV, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        idx_id = header.index("subject_id")
        idx_label = header.index("label")
        idx_split = header.index("split")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            mapping[parts[idx_id]] = (parts[idx_label], parts[idx_split])
    return mapping


def build_volume_index(subject_ids):
    """Step 2 / D-01: single walk of the OASIS root. Map each in-scope
    subject to its volume file (alphabetically first match found within a
    directory whose name matches the subject ID)."""
    in_scope = set(subject_ids)
    matches = {sid: [] for sid in in_scope}
    for dirpath, _dirnames, filenames in os.walk(OASIS_ROOT):
        # Identify which in-scope subject (if any) owns this directory path.
        owner = None
        for component in os.path.normpath(dirpath).split(os.sep):
            if component in in_scope:
                owner = component
                break
        if owner is None:
            continue
        for name in filenames:
            if fnmatch.fnmatch(name, VOLUME_PATTERN):
                matches[owner].append(os.path.join(dirpath, name))
    index = {}
    for sid, paths in matches.items():
        if paths:
            index[sid] = sorted(paths)[0]  # alphabetically first
    return index


def create_output_dirs():
    """Step 3: create all 18 leaf directories (idempotent)."""
    for plane in PLANES:
        for split in SPLITS:
            for label in LABELS:
                os.makedirs(os.path.join(SLICES_ROOT, plane, split, label),
                            exist_ok=True)


def normalize_volume(vol):
    """D-03: per-volume min-max normalization to uint8."""
    vmin = vol.min()
    vmax = vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin) * 255
    else:
        vol = np.zeros_like(vol)
    return vol.astype(np.uint8)


def main():
    print("=== STAGE 1D: SLICE EXTRACTION ===")
    print()
    print(f"Python: {sys.version.split()[0]}")
    print(f"nibabel: {nib.__version__}")
    print(f"Pillow: {Image.__version__}")
    print()

    # -- Step 1 ------------------------------------------------------------
    assignments = load_split_assignments()
    subject_ids = sorted(assignments.keys())

    print(f"Subjects to process: {len(subject_ids)}")
    print("Planes: sagittal (5 slices), coronal (7 slices), axial (7 slices)")
    print("Slices per subject: 19")
    print(f"Expected total slices: {EXPECT_TOTAL_SLICES}")
    print()

    # -- Step 2 ------------------------------------------------------------
    volume_index = build_volume_index(subject_ids)
    missing = [sid for sid in subject_ids if sid not in volume_index]
    for sid in missing:
        print(f"  [missing volume] {sid}")
    if len(missing) > 5:
        print(f"STOP: {len(missing)} subjects missing volume files")
        print(f"\n=== STAGE 1D RESULT: STOP — {len(missing)} subjects missing volume files ===")
        sys.exit(1)

    # -- Step 3 ------------------------------------------------------------
    create_output_dirs()

    # -- Step 4 ------------------------------------------------------------
    print("--- Processing ---")
    failed = []          # list of (subject_id, reason)
    succeeded = 0
    total_written = 0
    processed = 0

    for sid in subject_ids:
        label, split = assignments[sid]
        path = volume_index.get(sid)
        try:
            if path is None:
                raise FileNotFoundError("volume not found")
            vol = nib.load(path).get_fdata()
            vol = np.squeeze(vol)
            if vol.ndim != 3:
                print(f"  [warning] {sid}: volume is {vol.ndim}D after squeeze, skipping")
                failed.append((sid, f"not 3D (ndim={vol.ndim})"))
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed: {processed}/235")
                continue
            vol = normalize_volume(vol)

            for plane, (index_range, extractor) in PLANES.items():
                out_dir = os.path.join(SLICES_ROOT, plane, split, label)
                for idx in index_range:
                    slice_2d = extractor(vol, idx)
                    img = Image.fromarray(np.ascontiguousarray(slice_2d), mode="L")
                    fname = f"{sid}_{plane}_{idx:03d}.png"
                    img.save(os.path.join(out_dir, fname))
                    total_written += 1
            succeeded += 1
        except Exception as exc:  # save error / load error / etc.
            print(f"  [error] {sid}: {exc}")
            failed.append((sid, str(exc)))

        processed += 1
        if processed % 50 == 0:
            print(f"Processed: {processed}/235")

    if processed % 50 != 0:
        print(f"Processed: {processed}/235")
    print()

    # -- Step 5 ------------------------------------------------------------
    print("--- Results ---")
    print(f"Subjects successfully processed: {succeeded}")
    print(f"Subjects failed: {len(failed)}")
    for sid, reason in failed:
        print(f"  {sid}: {reason}")
    print(f"Total PNG files written: {total_written}")
    if failed:
        print(f"WARNING: {len(failed)} subjects failed")
    print()

    # -- Slice count per directory (18 rows) -------------------------------
    print("--- Slice Count per Directory ---")
    print("Plane      | Split | Label        | Files")
    print("-----------|-------|--------------|------")
    for plane in PLANES:
        for split in SPLITS:
            for label in LABELS:
                d = os.path.join(SLICES_ROOT, plane, split, label)
                n = len([f for f in os.listdir(d) if f.endswith(".png")]) \
                    if os.path.isdir(d) else 0
                print(f"{plane:<10} | {split:<5} | {label:<12} | {n}")
    print()

    # -- Sample image properties -------------------------------------------
    print("--- Sample Image Properties ---")
    for plane in PLANES:
        sample = None
        candidates = []
        for split in SPLITS:
            for label in LABELS:
                d = os.path.join(SLICES_ROOT, plane, split, label)
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith(".png"):
                            candidates.append(os.path.join(d, f))
        if candidates:
            sample = random.choice(candidates)
            arr = np.array(Image.open(sample))
            print(f"{plane:<8} sample: shape={list(arr.shape)}, dtype={arr.dtype}, "
                  f"min={arr.min()}, max={arr.max()}")
        else:
            print(f"{plane:<8} sample: NONE (no files)")
    print()

    # -- Paper comparison table --------------------------------------------
    def yn(ok):
        return "YES" if ok else "NO"

    obs_sag = EXPECT_PER_PLANE["sagittal"]
    obs_cor = EXPECT_PER_PLANE["coronal"]
    obs_axi = EXPECT_PER_PLANE["axial"]
    obs_total = total_written
    obs_subj = succeeded

    print("--- PAPER COMPARISON TABLE ---")
    print("| Metric                   | Paper §3.1      | Expected | Observed | Match  |")
    print("|--------------------------|-----------------|----------|----------|--------|")
    print(f"| Sagittal slices/subject  | 78–98 step 5    | 5        | {obs_sag:<8} | {yn(obs_sag == 5):<6} |")
    print(f"| Coronal slices/subject   | 75–105 step 5   | 7        | {obs_cor:<8} | {yn(obs_cor == 7):<6} |")
    print(f"| Axial slices/subject     | 73–103 step 5   | 7        | {obs_axi:<8} | {yn(obs_axi == 7):<6} |")
    print(f"| Total slices             | 235 × 19        | 4465     | {obs_total:<8} | {yn(obs_total == EXPECT_TOTAL_SLICES):<6} |")
    print(f"| Subjects processed       | §3.1 (235)      | 235      | {obs_subj:<8} | {yn(obs_subj == EXPECT_SUBJECTS):<6} |")
    print()

    if len(failed) > 5:
        print(f"=== STAGE 1D RESULT: STOP — {len(failed)} subjects failed ===")
        return 1
    print("=== STAGE 1D RESULT: PROCEED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
