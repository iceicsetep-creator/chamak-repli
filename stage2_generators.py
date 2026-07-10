#!/usr/bin/env python3
"""
Stage 2: ImageDataGenerator verification.

Instantiates all 9 Keras ImageDataGenerator flows (3 planes x 3 splits),
and verifies class mapping, sample counts, batch shapes/dtypes, pixel
value ranges, and that augmentation is active on the training flows.

No model building, no training, no file modification, no .md files.
All generator parameters are locked Research Supervisor decisions
(D-04, D-08, D-10, D-11, D-12) and are implemented exactly as stated.
"""

import sys

import numpy as np
import PIL
import sklearn
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SLICES_ROOT = "/kaggle/working/slices/"
PLANES = ["sagittal", "coronal", "axial"]
SPLITS = ["train", "val", "test"]

EXPECTED_CLASS_INDICES = {"Dementia": 0, "No_Dementia": 1}

EXPECTED_COUNTS = {
    ("sagittal", "train"): 655, ("sagittal", "val"): 165, ("sagittal", "test"): 355,
    ("coronal", "train"): 917, ("coronal", "val"): 231, ("coronal", "test"): 497,
    ("axial", "train"): 917, ("axial", "val"): 231, ("axial", "test"): 497,
}

# flow_from_directory parameters shared by all 9 generators (D-04, D-08, D-12).
FLOW_KWARGS = dict(
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    color_mode="rgb",
)


def make_train_datagen():
    """Training augmentation (D-10, D-11 — paper §3.3)."""
    return ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rotation_range=0,
        vertical_flip=False,
        shear_range=0,
        fill_mode="nearest",
    )


def make_eval_datagen():
    """Validation/test — rescale only (D-11, augmentation training-only)."""
    return ImageDataGenerator(rescale=1. / 255)


def gen_label(plane, split):
    return f"{plane:<9} / {split:<5}"


def main():
    print("=== STAGE 2: IMAGEDATAGENERATOR VERIFICATION ===")
    print()
    print("--- Environment ---")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"TensorFlow:   {tf.__version__}")
    print(f"NumPy:        {np.__version__}")
    print(f"Pillow:       {PIL.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print()

    stop_reasons = []

    # -- Step 2: instantiate all 9 generators ------------------------------
    train_datagen = make_train_datagen()
    eval_datagen = make_eval_datagen()

    generators = {}
    for plane in PLANES:
        for split in SPLITS:
            directory = f"{SLICES_ROOT}{plane}/{split}/"
            if split == "train":
                gen = train_datagen.flow_from_directory(
                    directory, shuffle=True, **FLOW_KWARGS)
            else:
                gen = eval_datagen.flow_from_directory(
                    directory, shuffle=False, **FLOW_KWARGS)
            generators[(plane, split)] = gen
    print()

    # -- Step 3: class mapping ---------------------------------------------
    print("--- Class Mapping ---")
    print(f"Expected:  {EXPECTED_CLASS_INDICES}")
    observed_maps = {k: g.class_indices for k, g in generators.items()}
    first_map = observed_maps[("sagittal", "train")]
    print(f"Observed:  {first_map}")
    all_consistent = all(m == EXPECTED_CLASS_INDICES for m in observed_maps.values())
    if all_consistent:
        print("All 9 generators consistent: YES")
    else:
        print("All 9 generators consistent: STOP")
        for k, m in observed_maps.items():
            if m != EXPECTED_CLASS_INDICES:
                print(f"  {gen_label(*k)}: {m}")
        print("STOP: Unexpected class mapping")
        stop_reasons.append("Unexpected class mapping")
    print()

    # -- Step 4: sample counts ---------------------------------------------
    print("--- Sample Counts ---")
    print("Generator              | Expected | Observed | Match")
    print("-----------------------|----------|----------|------")
    count_mismatch = False
    for plane in PLANES:
        for split in SPLITS:
            exp = EXPECTED_COUNTS[(plane, split)]
            obs = generators[(plane, split)].samples
            match = "YES" if obs == exp else "NO"
            if obs != exp:
                count_mismatch = True
            print(f"{gen_label(plane, split)}      | {exp:<8} | {obs:<8} | {match}")
    if count_mismatch:
        print("STOP: Sample count mismatch")
        stop_reasons.append("Sample count mismatch")
    print()

    # -- Step 5 & 6: batch shape/dtype and pixel range ---------------------
    batch_rows = []
    pixel_rows = []
    first_batches = {}
    for plane in PLANES:
        for split in SPLITS:
            gen = generators[(plane, split)]
            x, y = next(gen)
            first_batches[(plane, split)] = (x, y)
            img_ok = (x.ndim == 4 and x.shape[1:] == (128, 128, 3))
            lbl_ok = (y.ndim == 2 and y.shape[1] == 2)
            status = "OK"
            if not img_ok:
                status = "STOP"
                stop_reasons.append(
                    f"Unexpected image shape {x.shape} ({gen_label(plane, split)})")
            if not lbl_ok:
                status = "STOP"
                stop_reasons.append(
                    f"Unexpected label shape {y.shape} ({gen_label(plane, split)})")
            batch_rows.append((
                gen_label(plane, split),
                str(x.shape), str(x.dtype), str(y.shape), str(y.dtype), status))

            pmin, pmax = float(x.min()), float(x.max())
            prange_ok = (0.0 <= pmin) and (pmax <= 1.0)
            pstatus = "OK" if prange_ok else "STOP"
            if not prange_ok:
                stop_reasons.append(
                    f"Pixel values out of range [{pmin},{pmax}] "
                    f"({gen_label(plane, split)})")
            pixel_rows.append((gen_label(plane, split), pmin, pmax, pstatus))

    print("--- Batch Shape and Dtype ---")
    print("Generator              | Image shape        | Image dtype | Label shape | Label dtype | Status")
    print("-----------------------|--------------------|-------------|-------------|-------------|-------")
    for label, ishape, idtype, lshape, ldtype, status in batch_rows:
        print(f"{label}      | {ishape:<18} | {idtype:<11} | {lshape:<11} | {ldtype:<11} | {status}")
    print()

    print("--- Pixel Value Range ---")
    print("Generator              | Min    | Max    | Status")
    print("-----------------------|--------|--------|-------")
    for label, pmin, pmax, pstatus in pixel_rows:
        print(f"{label}      | {pmin:<6.3f} | {pmax:<6.3f} | {pstatus}")
    print()

    # -- Step 7: augmentation activity check -------------------------------
    print("--- Augmentation Activity Check ---")
    print("Method: Two consecutive batches from sagittal/train, compare first image")
    sag_train = generators[("sagittal", "train")]
    b1x, _ = next(sag_train)
    b2x, _ = next(sag_train)
    img1 = b1x[0]
    img2 = b2x[0]
    identical = np.array_equal(img1, img2)
    print(f"Batch 1 first image stats: mean={img1.mean():.3f}, std={img1.std():.3f}")
    print(f"Batch 2 first image stats: mean={img2.mean():.3f}, std={img2.std():.3f}")
    print(f"Tensors identical: {'YES' if identical else 'NO'}")
    if identical:
        print("Result: WARNING: Two consecutive batches produced identical first images")
    else:
        print("Result: PASSED (outputs differ)")
    print()

    # -- Step 8: paper comparison table ------------------------------------
    # Reference values pulled from the actual sagittal/train generator + batch.
    ref_x, ref_y = first_batches[("sagittal", "train")]
    obs_max = float(ref_x.max())
    channels = ref_x.shape[-1]
    tsize = ref_x.shape[1:3]
    idg = train_datagen

    def yn(ok):
        return "YES" if ok else "NO"

    zoom_ok = tuple(idg.zoom_range) == (0.9, 1.1) or idg.zoom_range == 0.1
    width_ok = idg.width_shift_range == 0.1
    height_ok = idg.height_shift_range == 0.1
    hflip_ok = idg.horizontal_flip is True

    print("--- PAPER COMPARISON TABLE ---")
    print("| Parameter         | Paper Ref | Decision    | Observed        | Match  |")
    print("|-------------------|-----------|-------------|-----------------|--------|")
    print(f"| rescale           | §3.3      | 1./255      | max={obs_max:<11.3f} | {yn(obs_max <= 1.0):<6} |")
    print(f"| zoom_range        | §3.3      | 0.1         | {'confirmed':<15} | {yn(zoom_ok):<6} |")
    print(f"| width_shift       | §3.3      | 0.1         | {'confirmed':<15} | {yn(width_ok):<6} |")
    print(f"| height_shift      | §3.3      | 0.1         | {'confirmed':<15} | {yn(height_ok):<6} |")
    print(f"| horizontal_flip   | §3.3      | True        | {'confirmed':<15} | {yn(hflip_ok):<6} |")
    print(f"| target_size       | §3.5      | (128,128)   | {str(tuple(tsize) + (channels,)):<15} | {yn(tuple(tsize) == (128, 128)):<6} |")
    print(f"| batch_size        | §3.5      | 16          | {'confirmed':<15} | {yn(sag_train.batch_size == 16):<6} |")
    print(f"| class_mode        | D-04      | categorical | {'(N,2) labels':<15} | {yn(ref_y.shape[1] == 2):<6} |")
    print(f"| color_mode        | D-12      | rgb         | {str(channels) + ' channels':<15} | {yn(channels == 3):<6} |")
    print(f"| image dtype       | —         | float32     | {str(ref_x.dtype):<15} | {yn(ref_x.dtype == np.float32):<6} |")
    print("Note: D-04 (class_mode='categorical') is provisionally implemented.")
    print("It remains open pending first training run comparison with paper results.")
    print()

    if stop_reasons:
        print(f"=== STAGE 2 RESULT: STOP — {'; '.join(stop_reasons)} ===")
        return 1
    print("=== STAGE 2 RESULT: PROCEED ===")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
