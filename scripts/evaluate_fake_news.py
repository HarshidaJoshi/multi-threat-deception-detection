"""
Evaluation module for XAI-Verify
Created for dissertation and Scopus journal evaluation.
Compatible with existing project structure.
Do not modify existing project logic.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import text_models  # noqa: E402
from utils.evaluation_helpers import (  # noqa: E402
    classification_report_text,
    compute_binary_metrics,
    generate_results_tables_md,
    print_metrics_banner,
    save_confusion_matrix_plot,
    save_json,
    save_roc_curve_plot,
)
from utils.model_registry import DEFAULT_MISINFO_MODEL, get_misinfo_model  # noqa: E402

OUTPUT_DIR = ROOT / "data" / "fake_news_eval"
DEFAULT_CSV = ROOT / "data" / "fake_news_eval" / "labeled_news.csv"
DECISION_THRESHOLD = 50.0  # fake_score > 50 => predicted fake (matches production UI)


def _load_csv_dataset(csv_path: Path) -> tuple[list[str], list[int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Expected CSV columns: text, label (0=Real, 1=Fake)"
        )

    texts: list[str] = []
    labels: list[int] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: text, label")
        for row_idx, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                label = int(str(row.get("label", "")).strip())
            except ValueError as exc:
                raise ValueError(f"Invalid label on CSV row {row_idx}: {row.get('label')!r}") from exc
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 (Real) or 1 (Fake); row {row_idx} has {label}")
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError(f"No usable rows found in {csv_path}")
    return texts, labels


def _evaluate(csv_path: Path, model_key: str) -> dict:
    texts, y_true = _load_csv_dataset(csv_path)
    meta = get_misinfo_model(model_key)

    y_pred: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []

    print(f"Evaluating {len(texts)} news samples with model '{model_key}' ...")

    for idx, text in enumerate(texts, start=1):
        result = text_models.analyze(text, model_key=model_key)
        fake_score = float(result["fake_score"])
        prob_fake = fake_score / 100.0
        pred = 1 if fake_score > DECISION_THRESHOLD else 0
        label = y_true[idx - 1]

        y_score.append(prob_fake)
        y_pred.append(pred)
        rows.append(
            {
                "row": idx,
                "label": label,
                "predicted": pred,
                "fake_score": fake_score,
                "prediction_label": result["label"],
                "confidence": result["confidence"],
                "model_used": result["model_used"],
                "inference_backend": result["inference_backend"],
                "correct": label == pred,
                "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
            }
        )
        if idx % 20 == 0 or idx == len(texts):
            print(f"  Processed {idx}/{len(texts)} samples")

    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    report = classification_report_text(y_true, y_pred)

    real_n = sum(1 for y in y_true if y == 0)
    fake_n = sum(1 for y in y_true if y == 1)
    payload = {
        "module": "fake_news",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(csv_path.relative_to(ROOT)),
        "model_key": model_key,
        "model_label": meta["label"],
        "hf_model": meta.get("hf_model"),
        "decision_threshold": f"fake_score > {DECISION_THRESHOLD} => fake",
        "class_balance": {"real": real_n, "fake": fake_n},
        "metrics": metrics,
        "predictions": rows,
    }
    return {"payload": payload, "report": report, "y_true": y_true, "y_pred": y_pred, "y_score": y_score}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fake news BERT/RoBERTa classifier on a labeled CSV dataset."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to CSV with columns: text, label (0=Real, 1=Fake)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MISINFO_MODEL,
        choices=["bert", "roberta", "forensics"],
        help="Misinformation model key (default: bert)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    result = _evaluate(csv_path, args.model)
    payload = result["payload"]
    metrics = payload["metrics"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "results.json", payload)
    (OUTPUT_DIR / "classification_report.txt").write_text(result["report"], encoding="utf-8")

    cm_saved = save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        metrics["confusion_matrix_labels"],
        OUTPUT_DIR / "confusion_matrix.png",
        title=f"Fake News ({args.model.upper()}) — Confusion Matrix",
    )
    roc_saved = save_roc_curve_plot(
        result["y_true"],
        result["y_score"],
        OUTPUT_DIR / "roc_curve.png",
        title=f"Fake News ({args.model.upper()}) — ROC Curve",
    )

    print_metrics_banner(f"Fake News ({payload['model_label']})", metrics)
    print("Classification report:")
    print(result["report"])
    print(f"Saved: {OUTPUT_DIR.relative_to(ROOT)}/results.json")
    print(f"Saved: {OUTPUT_DIR.relative_to(ROOT)}/classification_report.txt")
    if cm_saved:
        print(f"Saved: {OUTPUT_DIR.relative_to(ROOT)}/confusion_matrix.png")
    else:
        print("Note: confusion_matrix.png skipped (install matplotlib for plots)")
    if roc_saved:
        print(f"Saved: {OUTPUT_DIR.relative_to(ROOT)}/roc_curve.png")
    else:
        print("Note: roc_curve.png skipped (install matplotlib for plots)")

    table_path = generate_results_tables_md()
    print(f"Updated journal table: {table_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
