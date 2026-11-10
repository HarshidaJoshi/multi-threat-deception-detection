"""Full-data (all 100,000 training images) transfer learning on CPU with a frozen ResNet50 backbone.

1. Extract 2048-d ImageNet ResNet50 features once for every image (chunked + resumable).
2. Train classification heads (logistic regression, small MLP) on the cached features.
3. Evaluate on the full 20,000-image test split with confidence intervals.

Resume after a crash by re-running the same command: finished chunks are skipped.

    python scripts/train_full_frozen_features.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSES = ("real", "fake")
IMG = (224, 224)


def list_split(split_dir: Path, cap_per_class: int | None, seed: int = 42):
    rng = random.Random(seed)
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        fs = sorted(str(p) for p in (split_dir / cls).iterdir() if p.suffix.lower() in EXTS)
        if cap_per_class and len(fs) > cap_per_class:
            fs = sorted(rng.sample(fs, cap_per_class))
        paths += fs
        labels += [label] * len(fs)
    return paths, np.array(labels, dtype=np.int8)


def extract(split: str, paths: list[str], out: Path, chunk: int, batch: int, base):
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input

    def load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG)
        img.set_shape((*IMG, 3))
        return preprocess_input(img)

    n_chunks = (len(paths) + chunk - 1) // chunk
    t0 = time.time()
    for ci in range(n_chunks):
        f = out / f"{split}_{ci:03d}.npy"
        if f.exists():
            continue
        part = paths[ci * chunk:(ci + 1) * chunk]
        ds = tf.data.Dataset.from_tensor_slices(part).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(2)
        feats = base.predict(ds, verbose=0).astype(np.float16)
        np.save(f.with_suffix(".tmp.npy"), feats)
        f.with_suffix(".tmp.npy").replace(f)
        done = ci + 1
        el = time.time() - t0
        print(f"[{split}] chunk {done}/{n_chunks}  elapsed {el/60:.1f} min", flush=True)


def load_features(split: str, out: Path, n: int, chunk: int):
    parts = [np.load(out / f"{split}_{ci:03d}.npy") for ci in range((n + chunk - 1) // chunk)]
    return np.concatenate(parts).astype(np.float32)


def metrics(y, p, thr=0.5):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = (p >= thr).astype(int)
    acc = float(accuracy_score(y, pred))
    ci = 1.96 * (acc * (1 - acc) / len(y)) ** 0.5
    return {"accuracy": round(acc, 4), "acc_ci95": round(ci, 4), "precision": round(float(precision_score(y, pred)), 4),
            "recall": round(float(recall_score(y, pred)), 4), "f1": round(float(f1_score(y, pred)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "deepfakeimage dataset/real_vs_fake/real-vs-fake"))
    ap.add_argument("--out", default=str(ROOT / "benchmark_out/full_frozen"))
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-cap", type=int, default=5000, help="validation images per class")
    ap.add_argument("--max-train-per-class", type=int, default=None, help="for smoke tests")
    ap.add_argument("--max-test-per-class", type=int, default=None, help="for smoke tests")
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    out = Path(args.out)
    (out / "features").mkdir(parents=True, exist_ok=True)
    root = Path(args.data)
    val_name = next(n for n in ("valid", "val", "validation") if (root / n).is_dir())
    splits = {
        "train": list_split(root / "train", args.max_train_per_class),
        "valid": list_split(root / val_name, args.val_cap),
        "test": list_split(root / "test", args.max_test_per_class),
    }
    print({k: len(v[0]) for k, v in splits.items()}, flush=True)

    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(*IMG, 3))
    for split in ("train", "valid", "test"):
        extract(split, splits[split][0], out / "features", args.chunk, args.batch, base)

    X = {s: load_features(s, out / "features", len(splits[s][0]), args.chunk) for s in splits}
    y = {s: splits[s][1].astype(int) for s in splits}
    print("features loaded", {s: X[s].shape for s in X}, flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X["train"])
    Z = {s: sc.transform(X[s]) for s in X}
    results = {"n_train": len(y["train"]), "n_valid": len(y["valid"]), "n_test": len(y["test"]), "heads": {}}
    probs = {}

    t0 = time.time()
    lr = LogisticRegression(C=0.05, max_iter=300).fit(Z["train"], y["train"])
    probs["logreg"] = lr.predict_proba(Z["test"])[:, 1]
    results["heads"]["resnet50-frozen + logistic regression"] = {"test": metrics(y["test"], probs["logreg"]),
                                                                 "valid": metrics(y["valid"], lr.predict_proba(Z["valid"])[:, 1]),
                                                                 "fit_seconds": round(time.time() - t0, 1)}
    print("logreg", results["heads"]["resnet50-frozen + logistic regression"]["test"], flush=True)

    t0 = time.time()
    from tensorflow.keras import layers
    mlp = tf.keras.Sequential([layers.Input(shape=(2048,)), layers.Dense(512, activation="relu"), layers.Dropout(0.4),
                               layers.Dense(128, activation="relu"), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")])
    mlp.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    mlp.fit(Z["train"], y["train"], validation_data=(Z["valid"], y["valid"]), epochs=40, batch_size=512, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
                       tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])
    probs["mlp"] = mlp.predict(Z["test"], verbose=0).reshape(-1)
    results["heads"]["resnet50-frozen + MLP"] = {"test": metrics(y["test"], probs["mlp"]),
                                                  "valid": metrics(y["valid"], mlp.predict(Z["valid"], verbose=0).reshape(-1)),
                                                  "fit_seconds": round(time.time() - t0, 1)}
    print("mlp", results["heads"]["resnet50-frozen + MLP"]["test"], flush=True)

    np.savez_compressed(out / "test_predictions.npz", y_true=y["test"], **probs)
    mlp.save(out / "mlp_head.keras")
    (out / "results_full_frozen.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("All outputs in", out, flush=True)


if __name__ == "__main__":
    main()
