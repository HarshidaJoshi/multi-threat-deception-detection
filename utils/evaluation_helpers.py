"""
Evaluation module for XAI-Verify
Created for dissertation and Scopus journal evaluation.
Compatible with existing project structure.
Do not modify existing project logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_class_images(class_dir: Path) -> list[Path]:
    """Return sorted image paths under a class folder (supports one nesting level)."""
    if not class_dir.is_dir():
        return []
    paths: list[Path] = []
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def discover_binary_image_dataset(test_root: Path) -> tuple[list[Path], list[int]]:
    """
    Discover datasets/test/real and datasets/test/fake layout.
    Label convention: real=0, fake=1 (positive class = fake).
    """
    real_dir = test_root / "real"
    fake_dir = test_root / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{real_dir}' and '{fake_dir}'. "
            "Place held-out test images in datasets/test/real and datasets/test/fake."
        )

    real_paths = list_class_images(real_dir)
    fake_paths = list_class_images(fake_dir)
    if not real_paths or not fake_paths:
        raise ValueError(
            f"Both class folders must contain images (found real={len(real_paths)}, fake={len(fake_paths)})."
        )

    paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    return paths, labels


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Compute standard binary classification metrics.
    Positive class = 1 (fake / scam).
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    n = int(len(yt))

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics: dict[str, Any] = {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy * 100, 4),
        "precision": round(precision * 100, 4),
        "recall": round(recall * 100, 4),
        "f1": round(f1 * 100, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": ["real", "fake"],
    }

    if y_score is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = round(_roc_auc_score(yt, np.asarray(y_score, dtype=float)), 4)
    else:
        metrics["roc_auc"] = None

    return metrics


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via rank statistic (Mann-Whitney U), no sklearn dependency."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def classification_report_text(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_names: tuple[str, str] = ("real", "fake"),
) -> str:
    """Human-readable per-class precision, recall, and F1."""
    lines = ["Classification Report", "=" * 60]
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    for class_idx, name in enumerate(target_names):
        tp = int(np.sum((yt == class_idx) & (yp == class_idx)))
        fp = int(np.sum((yt != class_idx) & (yp == class_idx)))
        fn = int(np.sum((yt == class_idx) & (yp != class_idx)))
        support = int(np.sum(yt == class_idx))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        lines.append(f"\nClass: {name} (label={class_idx})")
        lines.append(f"  Precision: {precision * 100:.2f}%")
        lines.append(f"  Recall:    {recall * 100:.2f}%")
        lines.append(f"  F1-score:  {f1 * 100:.2f}%")
        lines.append(f"  Support:   {support}")

    overall = compute_binary_metrics(y_true, y_pred)
    lines.append("\nOverall (positive=fake)")
    lines.append(f"  Accuracy:  {overall['accuracy']:.2f}%")
    lines.append(f"  Precision: {overall['precision']:.2f}%")
    lines.append(f"  Recall:    {overall['recall']:.2f}%")
    lines.append(f"  F1-score:  {overall['f1']:.2f}%")
    if overall.get("roc_auc") is not None:
        lines.append(f"  ROC-AUC:   {overall['roc_auc']:.4f}")
    lines.append(
        f"\nConfusion matrix [[TN, FP], [FN, TP]]: {overall['confusion_matrix']}"
    )
    return "\n".join(lines) + "\n"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_confusion_matrix_plot(
    confusion_matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
    title: str,
) -> bool:
    """Save confusion matrix PNG. Returns False if matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matrix = np.asarray(confusion_matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def save_roc_curve_plot(
    y_true: Sequence[int],
    y_score: Sequence[float],
    output_path: Path,
    title: str,
) -> bool:
    """Save ROC curve PNG. Returns False if matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    yt = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    thresholds = np.unique(scores)[::-1]
    tprs: list[float] = []
    fprs: list[float] = []
    pos = max(int(np.sum(yt == 1)), 1)
    neg = max(int(np.sum(yt == 0)), 1)

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (yt == 1)))
        fp = int(np.sum((pred == 1) & (yt == 0)))
        tprs.append(tp / pos)
        fprs.append(fp / neg)

    fprs.append(0.0)
    tprs.append(0.0)
    order = np.argsort(fprs)
    fprs_sorted = np.asarray(fprs)[order]
    tprs_sorted = np.asarray(tprs)[order]
    auc = _roc_auc_score(yt, scores)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fprs_sorted, tprs_sorted, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def print_metrics_banner(module_name: str, metrics: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {module_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Samples:   {metrics['n']}")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-score:  {metrics['f1']:.2f}%")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {cm}")
    print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"{'=' * 60}\n")


def generate_results_tables_md(output_path: Path | None = None) -> Path:
    """
    Build journal-ready markdown table from saved evaluation JSON files.
    Reads data/deepfake_eval/results.json and data/fake_news_eval/results.json.
    """
    root = project_root()
    output_path = output_path or root / "data" / "results_tables.md"

    deepfake_path = root / "data" / "deepfake_eval" / "results.json"
    fake_news_path = root / "data" / "fake_news_eval" / "results.json"

    def _load_metrics(path: Path) -> dict[str, str]:
        if not path.is_file():
            return {
                "Accuracy": "—",
                "Precision": "—",
                "Recall": "—",
                "F1-score": "—",
                "ROC-AUC": "—",
            }
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data.get("metrics", {})
        roc = m.get("roc_auc")
        return {
            "Accuracy": f"{m.get('accuracy', '—')}%",
            "Precision": f"{m.get('precision', '—')}%",
            "Recall": f"{m.get('recall', '—')}%",
            "F1-score": f"{m.get('f1', '—')}%",
            "ROC-AUC": "—" if roc is None else str(roc),
        }

    df_metrics = _load_metrics(deepfake_path)
    fn_metrics = _load_metrics(fake_news_path)

    lines = [
        "# XAI-Verify — Journal Evaluation Summary",
        "",
        "Auto-generated from evaluation scripts. Re-run after updating test data.",
        "",
        "| Metric | DeepFake | Fake News |",
        "|---------|----------|-----------|",
    ]
    for key in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"):
        lines.append(f"| {key} | {df_metrics[key]} | {fn_metrics[key]} |")
    lines.extend(
        [
            "",
            "## Source files",
            "",
            f"- DeepFake: `{deepfake_path.relative_to(root)}`",
            f"- Fake News: `{fake_news_path.relative_to(root)}`",
            "",
            "Generate with:",
            "",
            "```powershell",
            "python scripts/evaluate_deepfake.py",
            "python scripts/evaluate_fake_news.py",
            "python scripts/generate_results_tables.py",
            "```",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
