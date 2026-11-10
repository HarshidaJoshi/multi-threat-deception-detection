"""
DeepFake classifier evaluation for XAI-Verify.

Evaluates the latest trained EfficientNetB0 weights on held-out images.
Uses the same 224x224 RGB preprocessing as training (see utils.deepfake_classifier).

Do not retrain or overwrite models — evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "trained_models"
OUTPUT_DIR = ROOT / "data" / "deepfake_eval"

INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
DECISION_THRESHOLD = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    return sorted(
        p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_image(path: Path) -> bool:
    """Return True if the file can be opened as an RGB image."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def _has_class_folders(eval_root: Path) -> bool:
    return (eval_root / "real").is_dir() and (eval_root / "fake").is_dir()


def _count_images_recursive(class_dir: Path) -> int:
    """Count all supported images under a class folder (any nesting depth)."""
    return len(_list_images(class_dir))


def discover_eval_root(datasets_dir: Path | None = None) -> tuple[Path, str]:
    """
    Resolve evaluation data root.

    Priority:
      1. datasets/Deepfake image detection/test-20250112T065939Z-001/test/
      2. datasets/test/
      3. datasets/real/ and datasets/fake/
    """
    base = datasets_dir or DATASETS_DIR
    official_root = (
        base / "Deepfake image detection" / "test-20250112T065939Z-001" / "test"
    )
    generic_test_root = base / "test"

    candidates: list[tuple[Path, str]] = [
        (
            official_root,
            "datasets/Deepfake image detection/test-20250112T065939Z-001/test",
        ),
        (generic_test_root, "datasets/test"),
        (base, "datasets/real and datasets/fake"),
    ]

    for eval_root, layout in candidates:
        if _has_class_folders(eval_root):
            real_n = _count_images_recursive(eval_root / "real")
            fake_n = _count_images_recursive(eval_root / "fake")
            if real_n > 0 and fake_n > 0:
                return eval_root, layout

    raise FileNotFoundError(
        "Could not locate a binary image dataset with real/ and fake/ subfolders.\n"
        "Checked (in order):\n"
        f"  1. {official_root}\n"
        f"  2. {generic_test_root}\n"
        f"  3. {base / 'real'} and {base / 'fake'}\n"
        "Each layout must contain recursively discoverable images "
        f"({', '.join(sorted(ext.lstrip('.') for ext in IMAGE_EXTENSIONS))})."
    )


def collect_labeled_images(eval_root: Path) -> tuple[list[Path], list[int], dict[str, Any]]:
    """Collect valid image paths and binary labels (real=0, fake=1)."""
    real_paths = _list_images(eval_root / "real")
    fake_paths = _list_images(eval_root / "fake")

    if not real_paths and not fake_paths:
        raise ValueError(
            f"No images found under {eval_root}/real and {eval_root}/fake. "
            "Supported extensions: " + ", ".join(sorted(IMAGE_EXTENSIONS))
        )

    valid_paths: list[Path] = []
    valid_labels: list[int] = []
    skipped: list[str] = []

    for label, paths in ((0, real_paths), (1, fake_paths)):
        for path in paths:
            if _validate_image(path):
                valid_paths.append(path)
                valid_labels.append(label)
            else:
                skipped.append(str(path.relative_to(ROOT)))

    real_valid = sum(1 for y in valid_labels if y == 0)
    fake_valid = sum(1 for y in valid_labels if y == 1)

    if real_valid == 0 or fake_valid == 0:
        raise ValueError(
            "Both classes must contain at least one valid image after filtering corrupt files.\n"
            f"  Valid real: {real_valid} (discovered {len(real_paths)})\n"
            f"  Valid fake: {fake_valid} (discovered {len(fake_paths)})\n"
            + (f"  Skipped corrupt/invalid: {len(skipped)}" if skipped else "")
        )

    meta = {
        "real_discovered": len(real_paths),
        "fake_discovered": len(fake_paths),
        "real_valid": real_valid,
        "fake_valid": fake_valid,
        "skipped_invalid": skipped,
    }
    return valid_paths, valid_labels, meta


def find_latest_model(models_dir: Path | None = None) -> Path:
    """
    Locate the most recently modified trained model in trained_models/.

    Supports .keras, .h5, and TensorFlow SavedModel directories.
    """
    directory = models_dir or MODELS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Models directory not found: {directory}\n"
            "Train first: python manage.py train_deepfake --dataset datasets"
        )

    candidates: list[tuple[float, Path]] = []

    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".keras", ".h5"}:
            candidates.append((path.stat().st_mtime, path))
        elif path.is_dir() and (path / "saved_model.pb").is_file():
            candidates.append((path.stat().st_mtime, path))

    for path in directory.rglob("saved_model.pb"):
        parent = path.parent
        candidates.append((parent.stat().st_mtime, parent))

    if not candidates:
        raise FileNotFoundError(
            f"No trained EfficientNetB0 model found in {directory}.\n"
            "Expected a .keras, .h5, or SavedModel directory.\n"
            "Train first: python manage.py train_deepfake --dataset datasets"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_trained_model(model_path: Path):
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    return tf.keras.models.load_model(str(model_path))


def _load_image_array(path: Path) -> np.ndarray:
    """Match training inference preprocessing (224x224 RGB, float32 [0, 255])."""
    from tensorflow.keras.preprocessing import image as kimage

    img = kimage.load_img(str(path), target_size=INPUT_SIZE)
    return kimage.img_to_array(img)


def predict_in_batches(model, paths: list[Path], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Return fake probabilities for each image path."""
    scores: list[float] = []
    n = len(paths)

    for start in range(0, n, batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = np.stack([_load_image_array(p) for p in batch_paths], axis=0)
        probs = model.predict(batch, verbose=0).reshape(-1)
        scores.extend(float(max(0.0, min(1.0, p))) for p in probs)

        done = min(start + batch_size, n)
        if done % (batch_size * 2) == 0 or done == n:
            print(f"  Processed {done}/{n} images")

    return np.asarray(scores, dtype=np.float64)


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics: dict[str, Any] = {
        "n": int(len(yt)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(float(accuracy_score(yt, yp)) * 100, 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)) * 100, 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)) * 100, 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)) * 100, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": ["real", "fake"],
    }

    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(yt, y_score)), 4)
    else:
        metrics["roc_auc"] = None

    metrics["classification_report_sklearn"] = classification_report(
        yt, yp, target_names=["real", "fake"], digits=4, zero_division=0
    )
    return metrics


def save_confusion_matrix_plot(cm: list[list[int]], labels: list[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("DeepFake Classifier — Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot(y_true: list[int], y_score: np.ndarray, output_path: Path) -> None:
    from sklearn.metrics import auc, roc_curve
    import matplotlib.pyplot as plt

    yt = np.asarray(y_true, dtype=int)
    fpr, tpr, _ = roc_curve(yt, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"EfficientNetB0 (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("DeepFake Classifier — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_predictions_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "predicted", "fake_probability", "correct"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_deepfake_results_md(metrics: dict[str, Any], output_path: Path) -> None:
    roc = metrics.get("roc_auc")
    roc_display = "—" if roc is None else f"{roc:.4f}"
    lines = [
        "# DeepFake Evaluation Results",
        "",
        "Auto-generated by `scripts/evaluate_deepfake.py`.",
        "",
        "| Metric | EfficientNetB0 |",
        "|---------|----------------|",
        f"| Accuracy | {metrics['accuracy']:.2f} |",
        f"| Precision | {metrics['precision']:.2f} |",
        f"| Recall | {metrics['recall']:.2f} |",
        f"| F1 | {metrics['f1']:.2f} |",
        f"| ROC-AUC | {roc_display} |",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_dataset_summary(eval_root: Path, real_n: int, fake_n: int) -> None:
    print(f"Dataset path used: {eval_root.relative_to(ROOT)}")
    print(f"Real images: {real_n}")
    print(f"Fake images: {fake_n}")
    print(f"Total images: {real_n + fake_n}")


def print_evaluation_banner(
    eval_root: Path,
    real_n: int,
    fake_n: int,
    metrics: dict[str, Any],
) -> None:
    roc = metrics.get("roc_auc")
    roc_display = "N/A" if roc is None else f"{roc:.4f}"

    print("\n" + "=" * 41)
    print("DeepFake Evaluation")
    print("=" * 41)
    print()
    print(f"Dataset path used: {eval_root.relative_to(ROOT)}")
    print(f"Real images: {real_n}")
    print(f"Fake images: {fake_n}")
    print(f"Total images: {real_n + fake_n}")
    print()
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"F1:        {metrics['f1']:.2f}%")
    print(f"ROC AUC:   {roc_display}")
    print()
    print("Confusion Matrix")
    print(f"  TP: {metrics['tp']}")
    print(f"  TN: {metrics['tn']}")
    print(f"  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}")
    print("=" * 41)


def _load_model_metadata(model_path: Path) -> dict[str, Any] | None:
    meta_candidates = [
        model_path.with_suffix(".meta.json"),
        model_path.parent / f"{model_path.name}.meta.json",
    ]
    if model_path.suffix:
        meta_candidates.append(model_path.with_name(model_path.stem + ".meta.json"))

    for candidate in meta_candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def evaluate(
    datasets_dir: Path | None = None,
    models_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    eval_root, layout = discover_eval_root(datasets_dir)
    real_discovered = _count_images_recursive(eval_root / "real")
    fake_discovered = _count_images_recursive(eval_root / "fake")
    print_dataset_summary(eval_root, real_discovered, fake_discovered)

    paths, y_true, collection_meta = collect_labeled_images(eval_root)

    model_path = find_latest_model(models_dir)
    print(f"\nLoading model: {model_path.relative_to(ROOT)}")
    model = load_trained_model(model_path)

    print(f"Evaluating {len(paths)} images (batch_size={batch_size}) ...")
    y_score = predict_in_batches(model, paths, batch_size=batch_size)
    y_pred = (y_score >= DECISION_THRESHOLD).astype(int).tolist()

    metrics = compute_metrics(y_true, y_pred, y_score)
    model_meta = _load_model_metadata(model_path)

    prediction_rows: list[dict[str, Any]] = []
    for path, label, pred, score in zip(paths, y_true, y_pred, y_score):
        prediction_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "label": "fake" if label == 1 else "real",
                "predicted": "fake" if pred == 1 else "real",
                "fake_probability": round(float(score), 6),
                "correct": label == pred,
            }
        )

    payload = {
        "module": "deepfake",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.relative_to(ROOT)),
        "model_metadata": model_meta,
        "dataset_root": str(eval_root.relative_to(ROOT)),
        "dataset_layout": layout,
        "input_size": list(INPUT_SIZE),
        "batch_size": batch_size,
        "decision_threshold": DECISION_THRESHOLD,
        "class_balance": {
            "real": collection_meta["real_valid"],
            "fake": collection_meta["fake_valid"],
        },
        "collection": collection_meta,
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report_sklearn"},
        "predictions": prediction_rows,
    }

    return {
        "payload": payload,
        "metrics": metrics,
        "layout": layout,
        "eval_root": eval_root,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "prediction_rows": prediction_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained EfficientNetB0 deepfake classifier on held-out images."
    )
    parser.add_argument(
        "--datasets-dir",
        default=str(DATASETS_DIR),
        help="Root datasets folder (default: datasets/)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help="Directory containing trained model weights (default: trained_models/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Inference batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for evaluation artifacts (default: data/deepfake_eval/)",
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.is_absolute():
        datasets_dir = ROOT / datasets_dir

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    try:
        result = evaluate(
            datasets_dir=datasets_dir,
            models_dir=models_dir,
            batch_size=max(1, args.batch_size),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    payload = result["payload"]
    metrics = result["metrics"]
    collection = payload["collection"]

    if collection.get("skipped_invalid"):
        warnings.warn(
            f"Skipped {len(collection['skipped_invalid'])} corrupt or unreadable image(s). "
            "See results.json for paths.",
            stacklevel=1,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = output_dir / "results.json"
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(metrics["classification_report_sklearn"], encoding="utf-8")

    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        metrics["confusion_matrix_labels"],
        output_dir / "confusion_matrix.png",
    )
    save_roc_curve_plot(result["y_true"], result["y_score"], output_dir / "roc_curve.png")
    save_predictions_csv(result["prediction_rows"], output_dir / "predictions.csv")
    generate_deepfake_results_md(metrics, output_dir / "deepfake_results.md")

    eval_root = result["eval_root"]
    print_evaluation_banner(
        eval_root=eval_root,
        real_n=collection["real_discovered"],
        fake_n=collection["fake_discovered"],
        metrics=metrics,
    )

    print("\nClassification report:")
    print(metrics["classification_report_sklearn"])

    rel_out = output_dir.relative_to(ROOT)
    print(f"\nSaved: {rel_out}/results.json")
    print(f"Saved: {rel_out}/classification_report.txt")
    print(f"Saved: {rel_out}/confusion_matrix.png")
    print(f"Saved: {rel_out}/roc_curve.png")
    print(f"Saved: {rel_out}/predictions.csv")
    print(f"Saved: {rel_out}/deepfake_results.md")


if __name__ == "__main__":
    main()
