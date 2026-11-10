"""Standalone benchmark for real-vs-fake image detection (run on a GPU: Kaggle / Colab).

Trains several backbones under identical conditions on the full dataset, evaluates every
model on the full held-out test split, and writes paper-ready tables and figures:

    <out>/results.json            all metrics
    <out>/results_table.md        comparison + ablation tables (Markdown)
    <out>/roc_curves.png          ROC of every model on the test set
    <out>/confusion_<name>.png    one confusion matrix per model
    <out>/<name>.keras            saved weights (EfficientNetB0 is what the Django app loads)

Dataset layout (same as the project):  <data>/{train,valid,test}/{real,fake}/*.jpg
Optional cross-dataset test:           --cross-data <dir>/{real,fake}/*.jpg

Example (Kaggle/Colab):
    python train_benchmark_gpu.py --data /kaggle/input/.../real-vs-fake --out /kaggle/working/out

Quick CPU smoke test:
    python train_benchmark_gpu.py --data <data> --out out_smoke --models efficientnetb0 simplecnn \
        --max-per-class 32 --epochs 1 --ft-epochs 0 --no-ablation --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

IMG = (224, 224)
CLASSES = ("real", "fake")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def make_ds(split_dir: Path, batch: int, training: bool, max_per_class: int | None, seed: int = 42):
    import tensorflow as tf

    rng = random.Random(seed)
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        files = list_images(split_dir / cls)
        if max_per_class and len(files) > max_per_class:
            files = rng.sample(files, max_per_class)
        paths += files
        labels += [float(label)] * len(files)
    if not paths:
        raise ValueError(f"No images under {split_dir} (expected real/ and fake/ subfolders)")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 20000), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG)
        img.set_shape((*IMG, 3))
        return img, tf.reshape(label, (1,))

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)


def build_model(name: str, freeze_base: bool = True):
    """Return (model, base). Preprocessing lives inside the model so raw 0-255 pixels go in."""
    from tensorflow.keras import Model, Sequential, layers
    from tensorflow.keras import applications as A

    inputs = layers.Input(shape=(*IMG, 3))

    if name == "simplecnn":
        x = layers.Rescaling(1 / 255.0)(inputs)
        for f in (32, 64, 128, 256):
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        base = None
    else:
        spec = {
            "efficientnetb0": (A.EfficientNetB0, A.efficientnet.preprocess_input),
            "resnet50": (A.ResNet50, A.resnet50.preprocess_input),
            "mobilenetv2": (A.MobileNetV2, A.mobilenet_v2.preprocess_input),
            "xception": (A.Xception, A.xception.preprocess_input),
        }[name]
        base = spec[0](weights="imagenet", include_top=False, input_shape=(*IMG, 3), pooling="avg")
        base.trainable = not freeze_base
        x = spec[1](inputs)
        x = base(x, training=False)

    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid", name="fake_probability", dtype="float32")(x)
    model = Model(inputs, out, name=f"deepfake_{name}")
    return model, base


def augmenter():
    from tensorflow.keras import Sequential, layers

    return Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.04), layers.RandomZoom(0.08), layers.RandomContrast(0.1)],
        name="augment",
    )


def predict(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(model.predict_on_batch(x).reshape(-1))
        y_true.append(y.numpy().reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def metrics(y_true, y_prob) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "samples": int(len(y_true)),
    }


def train_one(name, args, train_ds, val_ds, n_train, *, augment=True, fine_tune=True, tag=None):
    import tensorflow as tf

    tag = tag or name
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)

    model, base = build_model(name, freeze_base=True)
    ds = train_ds
    if augment:
        aug = augmenter()
        ds = train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    def cbs(patience):
        return [
            tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5, monitor="val_loss"),
        ]

    # Simple CNN trains end to end; pretrained backbones use two stages.
    lr = 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h1 = model.fit(ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs(3), verbose=2)
    history = {k: [float(v) for v in vals] for k, vals in h1.history.items()}

    if fine_tune and base is not None and args.ft_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-args.unfreeze]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
        h2 = model.fit(ds, validation_data=val_ds, epochs=args.ft_epochs, callbacks=cbs(2), verbose=2)
        for k, vals in h2.history.items():
            history.setdefault("ft_" + k, []).extend(float(v) for v in vals)

    return model, history, round(time.time() - t0, 1)


def save_confusion(m: dict, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc(curves: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (y, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def md_table(rows: list[tuple[str, dict]], extra: dict | None = None) -> str:
    head = "| Model | Accuracy | Precision | Recall | F1 | AUC |" + (" Train time (s) |" if extra else "")
    sep = "|---|---|---|---|---|---|" + ("---|" if extra else "")
    lines = [head, sep]
    for label, m in rows:
        t = f" {extra.get(label, '')} |" if extra else ""
        lines.append(f"| {label} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} |{t}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Dataset root with train/ valid/ test/")
    p.add_argument("--out", default="benchmark_out")
    p.add_argument("--models", nargs="+", default=["efficientnetb0", "resnet50", "mobilenetv2", "simplecnn"])
    p.add_argument("--epochs", type=int, default=6, help="Stage-1 epochs (frozen backbone)")
    p.add_argument("--ft-epochs", type=int, default=4, help="Stage-2 fine-tune epochs")
    p.add_argument("--unfreeze", type=int, default=40, help="Backbone layers to unfreeze when fine-tuning")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-per-class", type=int, default=None, help="Cap training images per class")
    p.add_argument("--max-eval-per-class", type=int, default=None, help="Cap valid/test images per class")
    p.add_argument("--cross-data", default=None, help="Optional second dataset (real/ fake/) for cross-dataset test")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--mixed-precision", action="store_true", help="Use float16 on GPU (faster, less memory)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPUs: {gpus or 'NONE (this will be slow)'}")
    if args.mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_name = next((n for n in ("valid", "val", "validation") if (root / n).is_dir()), None)
    if not (root / "train").is_dir() or val_name is None or not (root / "test").is_dir():
        raise SystemExit(f"{root} must contain train/, valid/ (or val/), and test/")

    eval_cap = args.max_eval_per_class or args.max_per_class
    train_ds, n_train = make_ds(root / "train", args.batch_size, True, args.max_per_class, args.seed)
    val_ds, n_val = make_ds(root / val_name, args.batch_size, False, eval_cap, args.seed)
    test_ds, n_test = make_ds(root / "test", args.batch_size, False, eval_cap, args.seed)
    cross_ds = make_ds(Path(args.cross_data), args.batch_size, False, eval_cap, args.seed)[0] if args.cross_data else None
    print(f"train={n_train}  valid={n_val}  test={n_test}")

    results: dict = {"config": vars(args) | {"n_train": n_train, "n_val": n_val, "n_test": n_test}, "models": {}, "ablation": {}}
    prior = out / "results.json"
    if prior.is_file():  # resume: keep finished models, only train the rest
        old = json.loads(prior.read_text(encoding="utf-8"))
        results["models"], results["ablation"] = old.get("models", {}), old.get("ablation", {})
        print(f"Resuming: already done -> {list(results['models']) + list(results['ablation'])}")
    curves, times = {}, {}

    def evaluate_and_record(label, model, history, secs, bucket):
        y, pr = predict(model, test_ds)
        m = metrics(y, pr)
        entry = {"test": m, "train_seconds": secs, "history": history, "params": int(model.count_params())}
        if cross_ds is not None:
            cy, cp = predict(model, cross_ds)
            entry["cross_dataset"] = metrics(cy, cp)
        results[bucket][label] = entry
        times[label] = secs
        print(f"[{label}] test: {m}")
        (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return y, pr, m

    for name in args.models:
        if name in results["models"] and (out / f"{name}.keras").is_file():
            print(f"\n=== {name}: already done, skipping ===")
            model = tf.keras.models.load_model(out / f"{name}.keras")
            y, pr = predict(model, test_ds)
            m = results["models"][name]["test"]
            curves[name] = (y, pr, m["auc"])
            times[name] = results["models"][name]["train_seconds"]
            continue
        print(f"\n=== {name} ===")
        model, hist, secs = train_one(name, args, train_ds, val_ds, n_train)
        model.save(out / f"{name}.keras")  # save before evaluating so training is never lost
        print(f"[{name}] training finished, saved; evaluating on test set...", flush=True)
        y, pr, m = evaluate_and_record(name, model, hist, secs, "models")
        curves[name] = (y, pr, m["auc"])
        save_confusion(m, out / f"confusion_{name}.png", name)

    if not args.no_ablation and "efficientnetb0" in args.models:
        for tag, kw in {
            "efficientnetb0 (no fine-tune)": dict(fine_tune=False),
            "efficientnetb0 (no augmentation)": dict(augment=False),
        }.items():
            if tag in results["ablation"]:
                print(f"\n=== ablation: {tag}: already done, skipping ===")
                continue
            print(f"\n=== ablation: {tag} ===")
            model, hist, secs = train_one("efficientnetb0", args, train_ds, val_ds, n_train, tag=tag, **kw)
            evaluate_and_record(tag, model, hist, secs, "ablation")

    if curves:
        save_roc(curves, out / "roc_curves.png")

    md = ["## Model comparison (test set)\n", md_table([(k, v["test"]) for k, v in results["models"].items()], times)]
    if results["ablation"]:
        full = results["models"].get("efficientnetb0")
        rows = ([("efficientnetb0 (full)", full["test"])] if full else []) + [(k, v["test"]) for k, v in results["ablation"].items()]
        md += ["\n\n## Ablation (EfficientNetB0)\n", md_table(rows)]
    if cross_ds is not None:
        md += ["\n\n## Cross-dataset test\n", md_table([(k, v["cross_dataset"]) for k, v in results["models"].items()])]
    (out / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
