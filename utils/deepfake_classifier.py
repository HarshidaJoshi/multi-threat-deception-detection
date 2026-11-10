"""Fine-tuned binary classifier for real vs AI-generated / deepfake images."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("DEEPFAKE_MODELS_DIR", PROJECT_ROOT / "trained_models"))
# ResNet50 won the benchmark (see benchmark_out/results_table.md); fall back to EfficientNetB0 if absent.
for _name in ("deepfake_resnet50_full.keras", "deepfake_resnet50.keras", "deepfake_efficientnet.keras"):
    DEFAULT_MODEL_FILE = MODELS_DIR / _name
    if DEFAULT_MODEL_FILE.is_file():
        break
INPUT_SIZE = (224, 224)

_loaded_model: Any = None
_loaded_meta: dict | None = None


def models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def model_path() -> Path:
    return Path(os.environ.get("DEEPFAKE_MODEL_PATH", DEFAULT_MODEL_FILE))


def meta_path() -> Path:
    return model_path().with_suffix(".meta.json")


def is_trained_model_available() -> bool:
    return model_path().is_file()


def get_model_metadata() -> dict | None:
    global _loaded_meta
    mp = meta_path()
    if not mp.is_file():
        return None
    if _loaded_meta is None:
        _loaded_meta = json.loads(mp.read_text(encoding="utf-8"))
    return _loaded_meta


def _load_model():
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model

    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(f"No trained weights at {path}")

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    _loaded_model = tf.keras.models.load_model(path)
    get_model_metadata()
    return _loaded_model


def build_classifier(freeze_base: bool = True):
    from tensorflow.keras import Model, layers
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input

    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*INPUT_SIZE, 3),
        pooling="avg",
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=(*INPUT_SIZE, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    model = Model(inputs=inputs, outputs=outputs, name="deepfake_efficientnet")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, base


def _count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts and p.is_file())


def _validate_class_folders(base: Path) -> None:
    real_dir = base / "real"
    fake_dir = base / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise ValueError(
            f"{base} must contain folders named exactly 'real' and 'fake'. "
            "Put ChatGPT / AI images in 'fake' and camera photos in 'real'."
        )
    if _count_images(real_dir) == 0 or _count_images(fake_dir) == 0:
        raise ValueError("Both 'real' and 'fake' folders need at least one image.")


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ("real", "fake")


def _list_images(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def _split_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        if (root / name).is_dir():
            return root / name
    return None


def _build_ds(
    split_dir: Path,
    batch_size: int,
    training: bool,
    max_per_class: int | None = None,
    seed: int = 42,
    file_subset: dict[str, list[str]] | None = None,
):
    """Return (tf.data.Dataset, class_counts) for a folder holding real/ and fake/."""
    import random

    import tensorflow as tf

    rng = random.Random(seed)
    paths: list[str] = []
    labels: list[float] = []
    counts: dict[str, int] = {}
    for label, cls in enumerate(CLASS_NAMES):
        files = file_subset[cls] if file_subset else _list_images(split_dir / cls)
        if max_per_class and len(files) > max_per_class:
            files = rng.sample(files, max_per_class)
        counts[cls] = len(files)
        paths += files
        labels += [float(label)] * len(files)
    if not paths:
        raise ValueError(f"No images found under {split_dir}")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 20000), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, INPUT_SIZE)
        img.set_shape((*INPUT_SIZE, 3))
        return img, tf.reshape(label, (1,))

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, counts


def _augment_layers():
    from tensorflow.keras import layers

    # Mild augmentation only: heavy blur/noise would destroy the artefacts the model must learn.
    return [layers.RandomFlip("horizontal"), layers.RandomRotation(0.04), layers.RandomZoom(0.08), layers.RandomContrast(0.1)]


def _split_holdout(files: dict[str, list[str]], fraction: float, seed: int = 42):
    import random

    rng = random.Random(seed)
    train, val = {}, {}
    for cls, items in files.items():
        items = list(items)
        rng.shuffle(items)
        k = max(1, int(len(items) * fraction))
        val[cls], train[cls] = items[:k], items[k:]
    return train, val


def _evaluate(model, ds) -> dict[str, float]:
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(model.predict_on_batch(x).reshape(-1))
        y_true.append(y.numpy().reshape(-1))
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    pred = (y_prob >= 0.5).astype(float)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    eps = 1e-7
    loss = float(-np.mean(y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps)))
    return {
        "accuracy": round(float((pred == y_true).mean()), 4),
        "loss": round(loss, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "samples": int(len(y_true)),
    }


def train_classifier(
    dataset_root: str | Path,
    epochs: int = 8,
    batch_size: int = 32,
    fine_tune_epochs: int = 4,
    validation_split: float = 0.2,
    max_per_class: int | None = None,
    max_eval_per_class: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train on <root>/train, validate on valid|val|validation, report final metrics on <root>/test.

    If <root> has no train/ folder, <root>/real and <root>/fake are split by ``validation_split``.
    Validation data is never the training data, so reported accuracy is an honest estimate.
    """
    import gc

    import tensorflow as tf
    from tensorflow.keras import Sequential

    tf.get_logger().setLevel("ERROR")
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(42)

    root = Path(dataset_root)
    train_dir = root / "train"
    val_dir = _split_dir(root, ("valid", "val", "validation"))
    test_dir = root / "test" if (root / "test").is_dir() else None

    eval_cap = max_eval_per_class or max_per_class
    if train_dir.is_dir():
        _validate_class_folders(train_dir)
        if val_dir is not None:
            _validate_class_folders(val_dir)
            train_ds, train_counts = _build_ds(train_dir, batch_size, True, max_per_class)
            val_ds, val_counts = _build_ds(val_dir, batch_size, False, eval_cap)
            split_used = 0.0
        else:
            files = {c: _list_images(train_dir / c) for c in CLASS_NAMES}
            tr, va = _split_holdout(files, validation_split)
            train_ds, train_counts = _build_ds(train_dir, batch_size, True, max_per_class, file_subset=tr)
            val_ds, val_counts = _build_ds(train_dir, batch_size, False, eval_cap, file_subset=va)
            split_used = validation_split
        sample_root = train_dir
    else:
        _validate_class_folders(root)
        test_dir = None
        files = {c: _list_images(root / c) for c in CLASS_NAMES}
        tr, va = _split_holdout(files, validation_split)
        train_ds, train_counts = _build_ds(root, batch_size, True, max_per_class, file_subset=tr)
        val_ds, val_counts = _build_ds(root, batch_size, False, eval_cap, file_subset=va)
        split_used = validation_split
        sample_root = root
    if test_dir is not None:
        _validate_class_folders(test_dir)

    n_real, n_fake = train_counts["real"], train_counts["fake"]
    total = n_real + n_fake
    class_weight = {0: total / (2 * n_real), 1: total / (2 * n_fake)}

    augment = Sequential(_augment_layers(), name="augment")
    train_aug = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    if resume_from:
        # Skip stage 1: continue from saved weights straight into fine-tuning.
        model = tf.keras.models.load_model(resume_from)
        base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
        epochs = 0
    else:
        model, base = build_classifier(freeze_base=True)
    out_path = model_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out_path.with_name(out_path.stem + ".best.keras")

    def callbacks(patience: int):
        return [
            tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5, monitor="val_loss"),
            tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        ]

    # Stage 1: train the new head on the frozen backbone.
    if epochs > 0:
        model.fit(train_aug, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=callbacks(3), verbose=1)

    # Stage 2: fine-tune top of the backbone; BatchNorm layers stay frozen for stability.
    if fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-40]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_aug, validation_data=val_ds, epochs=fine_tune_epochs, class_weight=class_weight, callbacks=callbacks(2), verbose=1)

    val_metrics = _evaluate(model, val_ds)
    test_metrics = None
    if test_dir is not None:
        test_ds, _ = _build_ds(test_dir, batch_size, False, eval_cap)
        test_metrics = _evaluate(model, test_ds)

    model.save(out_path)
    if ckpt.exists():
        ckpt.unlink()

    meta = {
        "model_name": "EfficientNetB0",
        "input_size": list(INPUT_SIZE),
        "classes": {"real": 0, "fake": 1},
        "val_accuracy": val_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "validation": val_metrics,
        "test": test_metrics,
        "real_images": n_real,
        "fake_images": n_fake,
        "validation_images": val_counts,
        "validation_split": split_used,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root.resolve()),
    }
    meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")

    global _loaded_model, _loaded_meta
    _loaded_model = model
    _loaded_meta = meta
    return meta


def get_loaded_model():
    """The cached trained model (loads it on first use)."""
    return _load_model()


def load_image_batch(image_path: str) -> np.ndarray:
    """(1, 224, 224, 3) float32 RGB batch, resized exactly as in training (tf.image.resize, bilinear)."""
    import tensorflow as tf

    img = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
    img = tf.image.resize(img, INPUT_SIZE)
    return np.expand_dims(img.numpy().astype(np.float32), axis=0)


def predict_fake_probability(image_path: str) -> tuple[float, list[str]]:
    """Return fake/AI probability in [0, 1]. Label 1 = fake / AI generated."""
    model = _load_model()
    meta = get_model_metadata() or {}

    batch = load_image_batch(image_path)
    prob = float(model.predict(batch, verbose=0).flatten()[0])
    prob = max(0.0, min(1.0, prob))

    indicators = [
        f"Fine-tuned {(get_model_metadata() or {}).get('model_name', 'CNN')} classifier (trained on real-vs-fake face images)",
        f"Classifier fake/AI probability: {prob * 100:.1f}%",
    ]
    if meta.get("val_accuracy") is not None:
        indicators.append(f"Benchmark held-out test accuracy: {meta['val_accuracy'] * 100:.1f}%")

    return prob, indicators


def dataset_stats(dataset_root: Path) -> dict[str, int]:
    root = Path(dataset_root)
    stats: dict[str, int] = {}
    if (root / "train").is_dir():
        for cls in ("real", "fake"):
            d = root / "train" / cls
            if d.is_dir():
                stats[f"train_{cls}"] = _count_images(d)
        for split in ("valid", "val", "validation", "test"):
            v = root / split
            if v.is_dir():
                for cls in ("real", "fake"):
                    d = v / cls
                    if d.is_dir():
                        stats[f"{split}_{cls}"] = _count_images(d)
    else:
        for cls in ("real", "fake"):
            d = root / cls
            if d.is_dir():
                stats[cls] = _count_images(d)
    return stats



def ensure_dataset_ready(dataset_root: str | Path) -> dict[str, int]:
    """Return image counts or raise ValueError if training cannot proceed."""
    root = Path(dataset_root)
    stats = dataset_stats(root)
    if not stats:
        raise ValueError(
            f"No 'real' and 'fake' folders found under {root}. "
            "Create datasets/real/ and datasets/fake/, then add images to each."
        )

    real_count = sum(v for k, v in stats.items() if "real" in k)
    fake_count = sum(v for k, v in stats.items() if "fake" in k)
    if real_count == 0 or fake_count == 0:
        raise ValueError(
            f"Both 'real' and 'fake' folders need at least one image "
            f"(found real={real_count}, fake={fake_count} under {root})."
        )
    return stats
