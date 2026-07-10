#!/usr/bin/env python3
"""
Stage 4R1: diagnostic retraining with a SLICE-LEVEL validation split.

Identical to Stage 4 in every respect EXCEPT how validation data is
obtained: instead of the subject-level /val/ directories, validation is a
20% slice-level split of each plane's /train/ directory, produced with two
separate ImageDataGenerator instances (validation_split=0.2, subsets
'training' and 'validation'). Everything else — architecture, dropout,
unfreeze counts, optimizer, loss, callbacks, class weights, seeds — is
unchanged. No test-set evaluation in this stage.

Outputs go to models_R1/ and histories_R1/ so Stage 4 outputs are not
overwritten.
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

SLICES_ROOT = "/kaggle/working/slices/"
MODELS_DIR = "/kaggle/working/models_R1/"
HISTORIES_DIR = "/kaggle/working/histories_R1/"

INITIAL_LR = 0.0001
EPOCHS = 100
BATCH_SIZE = 16
TARGET_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
VALIDATION_SPLIT = 0.2
SPLIT_SEED = 42

ENABLE_SUBJECT_CALLBACK = False

ARCH_FUNCS = {
    "InceptionV3": InceptionV3,
    "Xception": Xception,
    "ResNet50": ResNet50,
}

# index, plane, architecture, dropout, decay_rate, n_unfreeze  (Table 1)
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

# Stage 4 (D-05) reference results, for the comparison table.
D05_REFERENCE = {
    (1, "sagittal", "InceptionV3"): (25, 0.7005),
    (2, "sagittal", "Xception"): (11, 0.7055),
    (3, "sagittal", "ResNet50"): (17, 0.6751),
    (4, "coronal", "InceptionV3"): (24, 0.6747),
    (5, "coronal", "Xception"): (21, 0.6191),
    (6, "coronal", "ResNet50"): (31, 0.6417),
    (7, "axial", "InceptionV3"): (23, 0.7074),
    (8, "axial", "Xception"): (17, 0.6464),
    (9, "axial", "ResNet50"): (16, 0.6554),
}


class ExponentialDecayPerEpoch(Callback):
    """new_lr = initial_lr * decay_rate^(epoch+1), applied at on_epoch_end."""

    def __init__(self, initial_lr, decay_rate):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        new_lr = self.initial_lr * (self.decay_rate ** (epoch + 1))
        self.model.optimizer.learning_rate.assign(new_lr)
        self.lrs.append(float(self.model.optimizer.learning_rate.numpy()))


class SubjectLevelAccuracyCallback(Callback):
    """Implemented but disabled (ENABLE_SUBJECT_CALLBACK = False)."""

    def __init__(self, generator=None):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        return


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


# --- Generator A: training subset, full augmentation ----------------------
def train_datagen():
    return ImageDataGenerator(
        rescale=1. / 255, zoom_range=0.1, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest",
        validation_split=VALIDATION_SPLIT)


# --- Generator B: validation subset, rescale only -------------------------
def val_datagen():
    return ImageDataGenerator(rescale=1. / 255,
                              validation_split=VALIDATION_SPLIT)


def make_train_gen(plane):
    return train_datagen().flow_from_directory(
        f"{SLICES_ROOT}{plane}/train/", subset="training", shuffle=True,
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", color_mode="rgb", seed=SPLIT_SEED)


def make_val_gen(plane):
    return val_datagen().flow_from_directory(
        f"{SLICES_ROOT}{plane}/train/", subset="validation", shuffle=False,
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", color_mode="rgb", seed=SPLIT_SEED)


def test_generator(plane):
    """Unchanged from Stage 4 (subject-level /test/). Not used for
    evaluation in this stage -- retained so the config stays identical."""
    return ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        f"{SLICES_ROOT}{plane}/test/", target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", color_mode="rgb",
        shuffle=False)


def save_history_csv(history, path):
    keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    n = len(history.history.get("loss", []))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for i in range(n):
            writer.writerow([i + 1] + [history.history.get(k, [None] * n)[i] for k in keys])


def train_one(cfg, class_weight):
    index, plane, arch, dropout, decay_rate, n_unfreeze = cfg
    print("═══════════════════════════════════════════════════════════════════════════")
    print(f"Model {index} of 9: {plane} / {arch}")
    print(f"Dropout: {dropout}  |  Decay: {decay_rate}  |  N unfreeze: {n_unfreeze}")
    print("═══════════════════════════════════════════════════════════════════════════")

    tgen = make_train_gen(plane)
    vgen = make_val_gen(plane)
    print(f"Generator A (training subset):   {tgen.samples} slices")
    print(f"Generator B (validation subset): {vgen.samples} slices")

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
    for i in range(n_epochs):
        lr_i = lr_cb.lrs[i] if i < len(lr_cb.lrs) else float("nan")
        print(f"Epoch {i + 1:03d}/{EPOCHS} — "
              f"loss: {history.history['loss'][i]:.4f}  "
              f"acc: {history.history['accuracy'][i]:.4f}  "
              f"val_loss: {history.history['val_loss'][i]:.4f}  "
              f"val_acc: {history.history['val_accuracy'][i]:.4f}  "
              f"lr: {lr_i:.2e}")

    best_val_loss = min(history.history["val_loss"])
    es_fired = early.stopped_epoch > 0
    es_epoch = early.stopped_epoch + 1 if es_fired else None
    print(f"Total epochs trained: {n_epochs}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"EarlyStopping fired: {'YES at epoch ' + str(es_epoch) if es_fired else 'NO'}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORIES_DIR, exist_ok=True)
    model_path = f"{MODELS_DIR}{plane}_{arch}.keras"
    hist_path = f"{HISTORIES_DIR}{plane}_{arch}_history.csv"
    model.save(model_path)
    save_history_csv(history, hist_path)
    print(f"Saved model:   {model_path}")
    print(f"Saved history: {hist_path}")

    result = {"idx": index, "plane": plane, "arch": arch,
              "epochs": n_epochs, "best_val_loss": best_val_loss}

    del model, history, tgen, vgen, early, lr_cb
    tf.keras.backend.clear_session()
    gc.collect()
    print()
    return result


def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=== STAGE 4R1: DIAGNOSTIC RETRAINING — SLICE-LEVEL VALIDATION SPLIT ===")
    print()
    print("Scientific question: Does slice-level validation improve test performance vs subject-level?")
    print("Change from Stage 4: validation_split=0.2 from training data (two separate generators)")
    print("Unchanged:          All hyperparameters, architecture, callbacks, test set")
    print("Warning:            Slice-level validation introduces subject leakage.")
    print("                    Any improvement may reflect leakage, not better methodology.")
    print("                    Test set remains subject-level and leakage-free.")
    print()
    print("Python seed: 42  |  NumPy seed: 42  |  TensorFlow seed: 42")
    print()

    # Class weights: same method/values as Stage 4 (compute once, reuse).
    label_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        f"{SLICES_ROOT}sagittal/train/", target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", color_mode="rgb",
        shuffle=False)
    weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=label_gen.classes)
    class_weight = {0: weights[0], 1: weights[1]}
    print("Class weights (same as Stage 4, computed once):")
    print(f"  Class 0 (Dementia):    {weights[0]:.4f}")
    print(f"  Class 1 (No_Dementia): {weights[1]:.4f}")
    del label_gen
    print()

    results = []
    for cfg in CONFIGS:
        results.append(train_one(cfg, class_weight))

    # -- comparison table --------------------------------------------------
    print("═══════════════════════════════════════════════════════════════════════════════")
    print("STAGE 4R1 TRAINING SUMMARY vs STAGE 4")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print("| # | Plane    | Arch        | D-05 epochs | R1 epochs | D-05 val_loss | R1 val_loss |")
    print("|---|----------|-------------|-------------|-----------|---------------|-------------|")
    for r in results:
        d05_epochs, d05_val = D05_REFERENCE[(r["idx"], r["plane"], r["arch"])]
        print(f"| {r['idx']} | {r['plane']:<8} | {r['arch']:<11} | "
              f"{d05_epochs:<11} | {r['epochs']:<9} | {d05_val:<13.4f} | "
              f"{r['best_val_loss']:<11.4f} |")
    print()
    print("Note: Lower R1 val_loss is expected due to subject leakage. See investigation framing.")
    print("═══════════════════════════════════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
