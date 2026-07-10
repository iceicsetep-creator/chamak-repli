#!/usr/bin/env python3
"""
Stage 3: model architecture verification.

Builds each of the 9 transfer-learning models (3 planes x 3 architectures),
verifies structure against the paper, and audits exactly which layers are
trainable. No compilation, no training, no image loading, no .md files.
At most one model is held in memory at a time.

Locked Research Supervisor decisions implemented exactly:
    - Base: weights='imagenet', include_top=False, input_shape=(128,128,3)
    - Head: GlobalAveragePooling2D -> Dropout(rate) -> Dense(2, softmax)
    - Freeze: base.trainable=False, then unfreeze base.layers[-N:]
      (no special BatchNormalization handling -- D-14 open)
"""

import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, Xception, ResNet50
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dropout, Dense)
from tensorflow.keras.models import Model

ARCH_FUNCS = {
    "InceptionV3": InceptionV3,
    "Xception": Xception,
    "ResNet50": ResNet50,
}

# index, plane, architecture, N (layers to unfreeze), dropout
CONFIGS = [
    (1, "sagittal", "InceptionV3", 30, 0.5),
    (2, "sagittal", "Xception", 15, 0.5),
    (3, "sagittal", "ResNet50", 25, 0.4),
    (4, "coronal", "InceptionV3", 50, 0.4),
    (5, "coronal", "Xception", 50, 0.3),
    (6, "coronal", "ResNet50", 25, 0.5),
    (7, "axial", "InceptionV3", 50, 0.4),
    (8, "axial", "Xception", 50, 0.3),
    (9, "axial", "ResNet50", 25, 0.5),
]

INPUT_SHAPE = (128, 128, 3)


def build_model(architecture_name, n_unfreeze, dropout_rate):
    """Return an uncompiled Keras model for the given architecture,
    applying the exact freeze strategy. Does NOT call model.compile()."""
    base_fn = ARCH_FUNCS[architecture_name]
    base_model = base_fn(weights="imagenet", include_top=False,
                         input_shape=INPUT_SHAPE)

    # Freeze strategy (D-07): freeze all, then unfreeze the last N layers.
    base_model.trainable = False
    for layer in base_model.layers[-n_unfreeze:]:
        layer.trainable = True

    # Custom head (D-04 provisional), attached in exact order.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def count_trainable(weights):
    return int(sum(np.prod(w.shape) for w in weights))


def verify_model(index, plane, arch, n_unfreeze, dropout_rate):
    """Build + verify one model. Returns (summary_dict, stop_reason|None)."""
    print("═══════════════════════════════════════════")
    print(f"Model {index} of 9: {plane} / {arch}")
    print(f"N to unfreeze: {n_unfreeze}  |  Dropout: {dropout_rate}")
    print("═══════════════════════════════════════════")

    model = build_model(arch, n_unfreeze, dropout_rate)
    base_layers = model.layers[:-3]   # equivalent to base_model.layers
    summary = {"idx": index, "plane": plane, "arch": arch,
               "n": n_unfreeze, "dropout": dropout_rate}

    # -- [A] input shape ---------------------------------------------------
    in_shape = tuple(model.input.shape)
    input_ok = in_shape == (None, 128, 128, 3)
    print(f"[A] Input shape:       {in_shape}  [{'PASS' if input_ok else 'STOP'}]")
    summary["input_ok"] = input_ok
    if not input_ok:
        return summary, f"Model {index}: unexpected input shape {in_shape}"

    # -- [B] ImageNet weights ---------------------------------------------
    fresh_base = ARCH_FUNCS[arch](weights="imagenet", include_top=False,
                                  input_shape=INPUT_SHAPE)
    # Match the first convolutional layer positionally: a freshly loaded base
    # gets session-unique layer names, so name-based lookup is unreliable.
    fresh_first_conv = next(l for l in fresh_base.layers
                            if isinstance(l, tf.keras.layers.Conv2D))
    model_first_conv = next(l for l in model.layers
                            if isinstance(l, tf.keras.layers.Conv2D))
    weights_match = all(
        np.array_equal(a, b)
        for a, b in zip(model_first_conv.get_weights(),
                        fresh_first_conv.get_weights()))
    base_total_params = fresh_base.count_params()
    del fresh_base
    print(f"[B] ImageNet weights:  {'CONFIRMED' if weights_match else 'STOP: Weights do not match ImageNet'}")
    summary["weights_ok"] = weights_match
    if not weights_match:
        return summary, f"Model {index}: weights do not match ImageNet"

    # -- [C] head structure ------------------------------------------------
    gap, drop, dense = model.layers[-3], model.layers[-2], model.layers[-1]
    head_ok = (
        isinstance(gap, GlobalAveragePooling2D)
        and isinstance(drop, Dropout)
        and abs(float(drop.rate) - dropout_rate) < 1e-9
        and isinstance(dense, Dense)
        and dense.units == 2
        and dense.activation.__name__ == "softmax"
    )
    print("[C] Head layers:")
    print(f"    Layer -3: {type(gap).__name__}")
    print(f"    Layer -2: Dropout(rate={float(drop.rate)})")
    print(f"    Layer -1: Dense({dense.units}, activation={dense.activation.__name__})")
    print(f"    Head check: {'PASS' if head_ok else 'STOP'}")
    summary["head_ok"] = head_ok
    if not head_ok:
        return summary, f"Model {index}: head structure mismatch"

    # -- [D] parameter counts ---------------------------------------------
    head_params = int(sum(l.count_params() for l in model.layers[-3:]))
    total_params = model.count_params()
    trainable_params = count_trainable(model.trainable_weights)
    non_trainable_params = count_trainable(model.non_trainable_weights)
    print("[D] Parameter counts:")
    print(f"    Base model params:   {base_total_params}")
    print(f"    Head params:         {head_params}")
    print(f"    Total params:        {total_params}")
    print(f"    Trainable params:    {trainable_params}")
    print(f"    Non-trainable params: {non_trainable_params}")
    summary["trainable_params"] = trainable_params

    # -- [E] unfrozen layer list ------------------------------------------
    unfrozen_region = base_layers[-n_unfreeze:]
    trainable_entries = [l for l in unfrozen_region if l.trainable]
    print(f"[E] Unfrozen layers (last {n_unfreeze} of base model):")
    print(f"    {'Idx':<4} | {'Name':<29} | {'Type':<21} | Trainable")
    print(f"    {'-'*4}-|-{'-'*29}-|-{'-'*21}-|----------")
    base_len = len(base_layers)
    for offset, layer in enumerate(unfrozen_region):
        base_idx = base_len - n_unfreeze + offset
        print(f"    {base_idx:<4} | {layer.name:<29} | "
              f"{type(layer).__name__:<21} | {layer.trainable}")
    match = len(trainable_entries) == n_unfreeze
    print(f"    Total unfrozen entries: {len(trainable_entries)}  "
          f"[{'MATCH' if match else 'MISMATCH'}]")
    if not match:
        return summary, (f"Model {index}: unfrozen count "
                         f"{len(trainable_entries)} != {n_unfreeze}")

    # -- [F] BatchNormalization status ------------------------------------
    bn_layers = [l for l in unfrozen_region
                 if isinstance(l, tf.keras.layers.BatchNormalization)]
    print("[F] BatchNormalization in unfrozen region:")
    print(f"    Count: {len(bn_layers)}")
    for l in bn_layers:
        print(f"    {l.name}: trainable={l.trainable}")

    # -- cleanup -----------------------------------------------------------
    del model, base_layers
    tf.keras.backend.clear_session()
    print(f"--- Model {index} complete. Memory cleared. ---")
    print()
    return summary, None


def main():
    print("=== STAGE 3: MODEL ARCHITECTURE VERIFICATION ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    print("NOTE: Models are NOT compiled in this stage (compilation in Stage 4).")
    print("NOTE: D-04 provisional — Dense(2, softmax). Open pending training results.")
    print("NOTE: D-14 open — BN layers follow default Keras behavior in unfrozen region.")
    print()

    summaries = []
    for cfg in CONFIGS:
        summary, stop = verify_model(*cfg)
        summaries.append(summary)
        if stop is not None:
            print(f"=== STAGE 3 RESULT: STOP — {stop} ===")
            return 1

    # -- summary table -----------------------------------------------------
    print("═══════════════════════════════════════════")
    print("SUMMARY TABLE (all 9 models)")
    print("═══════════════════════════════════════════")
    print("| # | Plane    | Arch        | N    | Dropout | Trainable Params | Input OK | Weights OK | Head OK |")
    print("|---|----------|-------------|------|---------|------------------|----------|------------|---------|")
    for s in summaries:
        print(f"| {s['idx']} | {s['plane']:<8} | {s['arch']:<11} | {s['n']:<4} | "
              f"{s['dropout']:<7} | {s['trainable_params']:<16} | "
              f"{'YES' if s['input_ok'] else 'NO':<8} | "
              f"{'YES' if s['weights_ok'] else 'NO':<10} | "
              f"{'YES' if s['head_ok'] else 'NO':<7} |")
    print("=== STAGE 3 RESULT: PROCEED ===")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
