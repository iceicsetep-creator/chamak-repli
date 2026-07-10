#!/usr/bin/env python3
"""
Stage 4: sequential training of all 9 models.

Each model is built (same strategy as Stage 3), compiled, trained, and
saved. No evaluation, no ModelCheckpoint, one model in memory at a time.
Ends when 9 .keras files and 9 history CSVs are written.

Locked Research Supervisor decisions implemented exactly:
    - Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy']
    - EarlyStopping(monitor='val_loss', patience=10,
                    restore_best_weights=True, verbose=1)
    - ExponentialDecayPerEpoch: new_lr = initial_lr * decay_rate^(epoch+1)
      assigned at on_epoch_end (epoch is 0-indexed in Keras).
    - Balanced class weights computed once, applied to all 9 models.
    - SubjectLevelAccuracyCallback implemented but DISABLED.
"""

import os
import gc
import csv
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, Xception, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Fixed configuration.
# ---------------------------------------------------------------------------
SLICES_ROOT = "/kaggle/working/slices/"
MODELS_DIR = "/kaggle/working/models/"
HISTORIES_DIR = "/kaggle/working/histories/"

INITIAL_LR = 0.0001
EPOCHS = 100
BATCH_SIZE = 16
TARGET_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)

ENABLE_SUBJECT_CALLBACK = False   # D-?? — left disabled for this run.

ARCH_FUNCS = {
    "InceptionV3": InceptionV3,
    "Xception": Xception,
    "ResNet50": ResNet50,
}

# index, plane, architecture, dropout, decay_rate, n_unfreeze
CONFIGS = [
    (1, "sagittal", "InceptionV3", 0.5, 0.9, 30),
    (2, "sagittal", "Xception", 0.5, 0.9, 15),
    (3, "sagittal", "ResNet50", 0.4, 0.8, 25),
    (4, "coronal", "InceptionV3", 0.4, 0.8, 50),
    (5, "coronal", "Xception", 0.3, 0.8, 50),
    (6, "coronal", "ResNet50", 0.5, 0.9, 25),
    (7, "axial", "InceptionV3", 0.4, 0.8, 50),
    (8, "axial", "Xception", 0.3, 0.8, 50),
    (9, "axial", "ResNet50", 0.5, 0.9, 25),
]


# ---------------------------------------------------------------------------
# Callbacks.
# ---------------------------------------------------------------------------
class ExponentialDecayPerEpoch(Callback):
    """new_lr = initial_lr * decay_rate^(epoch+1), applied at on_epoch_end.

    Keras passes a 0-indexed epoch, so after the first epoch (epoch=0) the
    LR becomes initial_lr * decay_rate^1, etc. Records the assigned LR per
    epoch for later reporting/verification.
    """

    def __init__(self, initial_lr, decay_rate):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        new_lr = self.initial_lr * (self.decay_rate ** (epoch + 1))
        self.model.optimizer.learning_rate.assign(new_lr)
        # Read the value back from the optimizer to record the actual LR.
        self.lrs.append(float(self.model.optimizer.learning_rate.numpy()))


class SubjectLevelAccuracyCallback(Callback):
    """Placeholder for subject-level accuracy aggregation. Implemented but
    NOT enabled in this stage (ENABLE_SUBJECT_CALLBACK = False)."""

    def __init__(self, generator=None):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        # Intentionally inert unless enabled by the Research Supervisor.
        return


# ---------------------------------------------------------------------------
# Model builder (same as Stage 3).
# ---------------------------------------------------------------------------
def build_model(architecture_name, n_unfreeze, dropout_rate):
    base_fn = ARCH_FUNCS[architecture_name]
    base_model = base_fn(weights="imagenet", include_top=False,
                         input_shape=INPUT_SHAPE)
    base_model.trainable = False
    for layer in base_model.layers[-n_unfreeze:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=outputs)


def make_train_datagen():
    return ImageDataGenerator(
        rescale=1. / 255, zoom_range=0.1, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, rotation_range=0,
        vertical_flip=False, shear_range=0, fill_mode="nearest")


def make_eval_datagen():
    return ImageDataGenerator(rescale=1. / 255)


def train_generator(plane):
    return make_train_datagen().flow_from_directory(
        f"{SLICES_ROOT}{plane}/train/", target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", color_mode="rgb",
        shuffle=True)


def val_generator(plane):
    return make_eval_datagen().flow_from_directory(
        f"{SLICES_ROOT}{plane}/val/", target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", color_mode="rgb",
        shuffle=False)


def analytical_lr(decay_rate, epoch_1indexed):
    return INITIAL_LR * (decay_rate ** epoch_1indexed)


def gpu_mem_str():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "GPU memory: N/A (no GPU)"
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return (f"GPU memory: current={info['current'] / 1e6:.1f} MB  "
                f"peak={info['peak'] / 1e6:.1f} MB")
    except Exception as exc:
        return f"GPU memory: unavailable ({exc})"


def save_history_csv(history, path):
    keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    n = len(history.history.get("loss", []))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for i in range(n):
            row = [i + 1] + [history.history.get(k, [None] * n)[i] for k in keys]
            writer.writerow(row)


def train_one(cfg, class_weight, verify_lr=False):
    index, plane, arch, dropout, decay_rate, n_unfreeze = cfg
    print("═══════════════════════════════════════════════════════════════════════════")
    print(f"Model {index} of 9: {plane} / {arch}")
    print(f"Dropout: {dropout}  |  Decay: {decay_rate}  |  N unfreeze: {n_unfreeze}")
    print("═══════════════════════════════════════════════════════════════════════════")

    tgen = train_generator(plane)
    vgen = val_generator(plane)

    model = build_model(arch, n_unfreeze, dropout)
    model.compile(optimizer=Adam(learning_rate=INITIAL_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    early = EarlyStopping(monitor="val_loss", patience=10,
                          restore_best_weights=True, verbose=1)
    lr_cb = ExponentialDecayPerEpoch(INITIAL_LR, decay_rate)
    callbacks = [early, lr_cb]
    if ENABLE_SUBJECT_CALLBACK:
        callbacks.append(SubjectLevelAccuracyCallback(vgen))

    history = model.fit(tgen, validation_data=vgen, epochs=EPOCHS,
                        class_weight=class_weight, callbacks=callbacks,
                        verbose=0)

    n_epochs = len(history.history["loss"])
    # -- per-epoch summary (Step 6) ----------------------------------------
    for i in range(n_epochs):
        lr_i = lr_cb.lrs[i] if i < len(lr_cb.lrs) else float("nan")
        print(f"Epoch {i + 1:03d}/{EPOCHS} — "
              f"loss: {history.history['loss'][i]:.4f}  "
              f"acc: {history.history['accuracy'][i]:.4f}  "
              f"val_loss: {history.history['val_loss'][i]:.4f}  "
              f"val_acc: {history.history['val_accuracy'][i]:.4f}  "
              f"lr: {lr_i:.2e}")

    # -- LR runtime verification (model 1 only) ----------------------------
    if verify_lr:
        print("LR runtime verification (model 1, epochs 1-3):")
        ok = True
        for e in (1, 2, 3):
            if e - 1 < len(lr_cb.lrs):
                actual = lr_cb.lrs[e - 1]
                expected = analytical_lr(decay_rate, e)
                # optimizer.learning_rate is stored as float32, so compare
                # with a float32-appropriate relative tolerance.
                match = bool(np.isclose(actual, expected, rtol=1e-5, atol=1e-12))
                ok = ok and match
                print(f"  after epoch {e}: actual={actual:.6e}  "
                      f"expected={expected:.6e}  "
                      f"{'MATCH' if match else 'MISMATCH'}")
        print(f"LR verification: {'MATCH' if ok else 'WARNING: LR mismatch'}")

    # -- training result summary -------------------------------------------
    best_val_loss = min(history.history["val_loss"])
    es_fired = early.stopped_epoch > 0
    es_epoch = early.stopped_epoch + 1 if es_fired else None
    final_lr = lr_cb.lrs[-1] if lr_cb.lrs else INITIAL_LR
    print(f"Total epochs trained: {n_epochs}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"EarlyStopping fired: {'YES at epoch ' + str(es_epoch) if es_fired else 'NO'}")
    print(f"Final LR: {final_lr:.6e}")

    # -- save (Step 5) -----------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORIES_DIR, exist_ok=True)
    model_path = f"{MODELS_DIR}{plane}_{arch}.keras"
    hist_path = f"{HISTORIES_DIR}{plane}_{arch}_history.csv"
    model.save(model_path)
    save_history_csv(history, hist_path)
    print(f"Saved model:   {model_path}")
    print(f"Saved history: {hist_path}")
    print(gpu_mem_str() + "   (after saving)")

    result = {
        "idx": index, "plane": plane, "arch": arch, "epochs": n_epochs,
        "best_val_loss": best_val_loss, "es_fired": es_fired,
        "saved": os.path.isfile(model_path) and os.path.isfile(hist_path),
    }

    # -- cleanup -----------------------------------------------------------
    del model, history, tgen, vgen, early, lr_cb
    tf.keras.backend.clear_session()
    gc.collect()
    print(gpu_mem_str() + "   (after cleanup)")
    print()
    return result


def main():
    # -- Step 1: seeds and environment -------------------------------------
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    print("=== STAGE 4: MODEL TRAINING ===")
    print("Python seed:     42")
    print("NumPy seed:      42")
    print("TensorFlow seed: 42")
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    print()

    # -- Step 2: GPU check -------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU available: YES  ({[g.name for g in gpus]})")
        print(gpu_mem_str())
    else:
        print("GPU available: NO")
        print("WARNING: No GPU detected")
    print()

    # -- Step 3: class weights (once) --------------------------------------
    label_gen = train_generator("sagittal")
    train_labels = label_gen.classes
    weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_labels)
    class_weight = {0: weights[0], 1: weights[1]}
    print("Class weights (computed once, applies to all 9 models):")
    print(f"  Class 0 (Dementia):    {weights[0]:.4f}")
    print(f"  Class 1 (No_Dementia): {weights[1]:.4f}")
    del label_gen
    print()

    # -- Step 4: LR schedule verification table ----------------------------
    print("LR Schedule Verification:")
    print("Epoch | decay=0.9        | decay=0.8")
    print("------|------------------|------------------")
    for e in range(1, 11):
        print(f"{e:<5} | {analytical_lr(0.9, e):.6e}     | {analytical_lr(0.8, e):.6e}")
    print()

    # -- Step 5: training loop ---------------------------------------------
    results = []
    for cfg in CONFIGS:
        results.append(train_one(cfg, class_weight, verify_lr=(cfg[0] == 1)))

    # -- Step 7: final summary table ---------------------------------------
    print("═══════════════════════════════════════════════════════════════════════════")
    print("STAGE 4 SUMMARY")
    print("═══════════════════════════════════════════════════════════════════════════")
    print("| # | Plane    | Arch        | Epochs | Best val_loss | ES fired | Saved |")
    print("|---|----------|-------------|--------|---------------|----------|-------|")
    for r in results:
        print(f"| {r['idx']} | {r['plane']:<8} | {r['arch']:<11} | {r['epochs']:<6} | "
              f"{r['best_val_loss']:<13.4f} | {'YES' if r['es_fired'] else 'NO':<8} | "
              f"{'YES' if r['saved'] else 'NO':<5} |")
    print()
    print("Output files:")
    print(f"  {MODELS_DIR}   — 9 .keras files")
    print(f"  {HISTORIES_DIR} — 9 .csv files")
    print("Open decisions carried into Stage 5:")
    print("  D-04 (Dense head) — Investigation Priority 1")
    print("  D-06 (LR decay)   — Investigation Priority pending LR verification result")
    print("  D-14 (BN trainable) — Investigation Priority 2")
    print("═══════════════════════════════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
