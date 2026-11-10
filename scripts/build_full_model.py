"""Assemble ONE deployable Keras model from the full-data pipeline:

    raw RGB pixels -> ResNet50 preprocessing -> frozen ImageNet ResNet50 (avg pool, 2048-d)
                   -> standardisation (train-set mean/std) -> trained MLP head -> P(fake)

and verify it reproduces the cached-feature pipeline's test predictions on a sample of images.
Output: benchmark_out/resnet50_full_mlp.keras and trained_models/deepfake_resnet50_full.* (+ meta json)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_out" / "full_frozen"
IMG = (224, 224)
CHUNK = 5000


def main():
    import tensorflow as tf
    from tensorflow.keras import layers

    # --- standardisation statistics from the cached TRAIN features (same data the head was fit on)
    feats = [np.load(OUT / "features" / f"train_{i:03d}.npy").astype(np.float64) for i in range(20)]
    X = np.concatenate(feats)
    mean, var = X.mean(axis=0), X.var(axis=0)
    del X, feats
    print("train feature stats:", mean.shape, flush=True)

    head = tf.keras.models.load_model(OUT / "mlp_head.keras")
    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(*IMG, 3))
    base.trainable = False

    inputs = layers.Input(shape=(*IMG, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    norm = layers.Normalization(mean=mean.astype(np.float32), variance=var.astype(np.float32), name="feature_standardise")
    x = norm(x)
    for layer in head.layers:
        x = layer(x, training=False) if not isinstance(layer, layers.InputLayer) else x
    model = tf.keras.Model(inputs, x, name="deepfake_resnet50_full_mlp")

    # --- verify against the cached-feature pipeline on test images
    probs = np.load(OUT / "test_predictions.npz")
    root = ROOT / "deepfakeimage dataset/real_vs_fake/real-vs-fake/test"
    exts = {".jpg", ".jpeg", ".png"}
    reals = sorted(p for p in (root / "real").iterdir() if p.suffix.lower() in exts)
    fakes = sorted(p for p in (root / "fake").iterdir() if p.suffix.lower() in exts)
    # cached order = sorted paths of real then fake (see train_full_frozen_features.list_split)
    reals_s, fakes_s = sorted(str(p) for p in reals), sorted(str(p) for p in fakes)
    rng = random.Random(0)
    idx = rng.sample(range(len(reals_s)), 20) + [len(reals_s) + i for i in rng.sample(range(len(fakes_s)), 20)]
    allp = reals_s + fakes_s
    from PIL import Image
    batch = np.stack([np.asarray(Image.open(allp[i]).convert("RGB").resize(IMG, Image.Resampling.BILINEAR), dtype=np.float32) for i in idx])
    pred = model.predict(batch, verbose=0).reshape(-1)
    ref = probs["mlp"][idx]
    diff = np.abs(pred - ref)
    print("max |deployable - cached pipeline| =", float(diff.max()), " mean =", float(diff.mean()), flush=True)

    model.save(ROOT / "benchmark_out" / "resnet50_full_mlp.keras")
    res = json.load(open(OUT / "results_full_frozen.json", encoding="utf-8"))
    mlp = res["heads"]["resnet50-frozen + MLP"]
    meta = {"model_name": "ResNet50 (frozen) + MLP", "input_size": [224, 224], "classes": {"real": 0, "fake": 1},
            "val_accuracy": mlp["test"]["accuracy"], "test": mlp["test"], "real_images": 50000, "fake_images": 50000,
            "dataset_root": "deepfakeimage dataset/real_vs_fake/real-vs-fake",
            "note": "Trained on all 100,000 training images; test metrics on all 20,000 held-out test images.",
            "verification_max_abs_diff_vs_pipeline": float(diff.max())}
    models_dir = ROOT / "trained_models"
    model.save(models_dir / "deepfake_resnet50_full.keras")
    json.dump(meta, open(models_dir / "deepfake_resnet50_full.meta.json", "w", encoding="utf-8"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
