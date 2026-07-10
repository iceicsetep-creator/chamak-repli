#!/usr/bin/env python3
"""
Stage 5: evaluation of all 9 trained models.

Loads each saved model, runs slice-level evaluation on the test set and
subject-level majority-vote evaluation, prints per-model result blocks and
a full comparison table against paper Table 2, and saves a results CSV.

No training, no model modification, no ensemble-selection decision (metrics
only — the Research Supervisor decides), no .md files.
"""

import os
import gc
import csv
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, precision_recall_fscore_support
from scipy import stats

MODELS_DIR = "/kaggle/working/models/"
SLICES_ROOT = "/kaggle/working/slices/"
SPLITS_CSV = "/kaggle/working/subject_splits.csv"
RESULTS_CSV = "/kaggle/working/stage5_results.csv"

CLASS_INDICES = {"Dementia": 0, "No_Dementia": 1}
TARGET_NAMES = ["Dementia", "No_Dementia"]
LABEL_TO_INT = {"Dementia": 0, "No_Dementia": 1}

# index, plane, architecture (Table 1 order)
CONFIGS = [
    (1, "sagittal", "InceptionV3"),
    (2, "sagittal", "Xception"),
    (3, "sagittal", "ResNet50"),
    (4, "coronal", "InceptionV3"),
    (5, "coronal", "Xception"),
    (6, "coronal", "ResNet50"),
    (7, "axial", "InceptionV3"),
    (8, "axial", "Xception"),
    (9, "axial", "ResNet50"),
]

# Paper Table 2 reference values (slice acc %, subject acc %, FN, FP).
PAPER = {
    ("sagittal", "InceptionV3"): (71, 76, 6, 11),
    ("sagittal", "Xception"): (70, 75, 7, 11),
    ("sagittal", "ResNet50"): (63, 66, 5, 19),
    ("coronal", "InceptionV3"): (72, 76, 6, 11),
    ("coronal", "Xception"): (79, 86, 4, 6),
    ("coronal", "ResNet50"): (78, 80, 5, 9),
    ("axial", "InceptionV3"): (70, 75, 7, 11),
    ("axial", "Xception"): (72, 79, 7, 8),
    ("axial", "ResNet50"): (73, 75, 8, 10),
}

EXPECTED_SLICES = {"sagittal": 355, "coronal": 497, "axial": 497}


def load_test_subject_labels():
    """subject_id -> int label (0/1) for the 71 test subjects."""
    labels = {}
    with open(SPLITS_CSV, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        i_id = header.index("subject_id")
        i_label = header.index("label")
        i_split = header.index("split")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if parts[i_split] == "test":
                labels[parts[i_id]] = LABEL_TO_INT[parts[i_label]]
    return labels


def test_generator(plane):
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_directory(
        f"{SLICES_ROOT}{plane}/test/", target_size=(128, 128),
        batch_size=16, class_mode="categorical", color_mode="rgb",
        shuffle=False)


def majority_vote(preds):
    """scipy.stats.mode across a subject's slice predictions."""
    try:
        result = stats.mode(preds, keepdims=False)
    except TypeError:  # older scipy without keepdims kwarg
        result = stats.mode(preds)
    return int(np.atleast_1d(result.mode)[0])


def evaluate_one(index, plane, arch, subject_labels):
    print("═══════════════════════════════════════════════════════")
    print(f"Model {index}/9: {plane} / {arch}")
    print("═══════════════════════════════════════════════════════")

    model_path = f"{MODELS_DIR}{plane}_{arch}.keras"
    model = tf.keras.models.load_model(model_path)

    gen = test_generator(plane)
    gen.reset()
    n_slices = gen.samples
    print(f"Slice-level ({n_slices} test slices):")

    probs = model.predict(gen, verbose=0)
    slice_pred = np.argmax(probs, axis=1)
    slice_true = gen.classes

    report = classification_report(
        slice_true, slice_pred, target_names=TARGET_NAMES,
        labels=[0, 1], zero_division=0, output_dict=True)
    for name in TARGET_NAMES:
        r = report[name]
        print(f"  {name:<11} {r['precision']:.2f}       {r['recall']:.2f}    "
              f"{r['f1-score']:.2f}      {int(r['support'])}")
    slice_acc = report["accuracy"]
    macro = report["macro avg"]
    print(f"  accuracy                       {slice_acc:.2f}      {n_slices}")
    print(f"  macro avg   {macro['precision']:.2f}       {macro['recall']:.2f}    "
          f"{macro['f1-score']:.2f}      {n_slices}")
    print()

    # -- subject-level majority voting -------------------------------------
    by_subject = {}
    for fname, pred in zip(gen.filenames, slice_pred):
        sid = os.path.basename(fname)[:13]
        by_subject.setdefault(sid, []).append(pred)

    subj_ids = sorted(by_subject.keys())
    y_true, y_pred = [], []
    for sid in subj_ids:
        y_true.append(subject_labels[sid])
        y_pred.append(majority_vote(np.array(by_subject[sid])))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_subjects = len(subj_ids)
    subj_acc = float((y_true == y_pred).mean()) * 100.0
    # FN: true Dementia(0) predicted No_Dementia(1); FP: true 1 predicted 0.
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)

    n_dem = int(np.sum(y_true == 0))
    n_nod = int(np.sum(y_true == 1))
    dem_correct = int(np.sum((y_true == 0) & (y_pred == 0)))
    nod_correct = int(np.sum((y_true == 1) & (y_pred == 1)))
    correct = int(np.sum(y_true == y_pred))

    paper_slice, paper_subj, paper_fn, paper_fp = PAPER[(plane, arch)]
    print(f"Subject-level ({n_subjects} test subjects):")
    print(f"  Accuracy:  {subj_acc:.2f}%   [Paper: {paper_subj}%]")
    print(f"  FN:        {fn}        [Paper: {paper_fn}]")
    print(f"  FP:        {fp}        [Paper: {paper_fp}]")
    print()
    print(f"  Subjects correctly classified: {correct}/{n_subjects}")
    print(f"  Dementia:    {dem_correct}/{n_dem} correct")
    print(f"  No_Dementia: {nod_correct}/{n_nod} correct")
    print()

    result = {
        "plane": plane, "architecture": arch,
        "slice_acc": round(slice_acc * 100.0, 2),
        "subject_acc": round(subj_acc, 2),
        "FN": fn, "FP": fp,
        "n_slices": n_slices, "n_subjects": n_subjects,
        "subj_precision": prec, "subj_recall": rec, "subj_f1": f1,
    }

    del model, gen
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def main():
    print("=== STAGE 5: MODEL EVALUATION ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU available: {'YES' if gpus else 'NO'}")
    print()

    subject_labels = load_test_subject_labels()
    print(f"Test subjects loaded: {len(subject_labels)}")
    if len(subject_labels) != 71:
        print(f"WARNING: expected 71 test subjects, found {len(subject_labels)}")
    print()

    results = []
    for index, plane, arch in CONFIGS:
        results.append(evaluate_one(index, plane, arch, subject_labels))

    # -- Step 5: comparison table ------------------------------------------
    print("═══════════════════════════════════════════════════════════════════════════════════════════")
    print("STAGE 5: FULL COMPARISON TABLE vs PAPER TABLE 2")
    print("═══════════════════════════════════════════════════════════════════════════════════════════")
    print("                    ── Slice-level ──────────────────  ── Subject-level ───────────────────")
    print("Plane    Arch        Acc(paper) Acc(ours) Δ     SubAcc(paper) SubAcc(ours) Δ      FN/FP(paper) FN/FP(ours)")
    print("-------- ----------- ---------- --------- ----- ------------- ------------ ------ ------------ -----------")
    subj_abs_deltas = []
    for (index, plane, arch), r in zip(CONFIGS, results):
        paper_slice, paper_subj, paper_fn, paper_fp = PAPER[(plane, arch)]
        slice_delta = r["slice_acc"] - paper_slice
        subj_delta = r["subject_acc"] - paper_subj
        subj_abs_deltas.append(abs(subj_delta))
        print(f"{plane:<8} {arch:<11} {str(paper_slice) + '%':<10} "
              f"{r['slice_acc']:<9.1f} {slice_delta:<+5.1f} "
              f"{str(paper_subj) + '%':<13} {r['subject_acc']:<12.1f} {subj_delta:<+6.1f} "
              f"{str(paper_fn) + '/' + str(paper_fp):<12} "
              f"{str(r['FN']) + '/' + str(r['FP'])}")
    mean_abs_delta = float(np.mean(subj_abs_deltas)) if subj_abs_deltas else 0.0
    print()
    print(f"Mean absolute Δ (subject-level accuracy): {mean_abs_delta:.1f} percentage points")
    print("═══════════════════════════════════════════════════════════════════════════════════════════")
    print("NOTE: D-04 provisional (Dense(2,softmax)). If gap is large, this is Investigation Priority 1.")
    print("NOTE: Test class distribution: 30 Dementia / 41 No_Dementia (paper: 32/39) — see D-09.")
    print("═══════════════════════════════════════════════════════════════════════════════════════════")

    # -- Step 6: save results ----------------------------------------------
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["plane", "architecture", "slice_acc", "subject_acc", "FN", "FP"])
        for r in results:
            writer.writerow([r["plane"], r["architecture"], r["slice_acc"],
                             r["subject_acc"], r["FN"], r["FP"]])
    print(f"\nSaved: {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
