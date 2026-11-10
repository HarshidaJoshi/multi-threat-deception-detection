"""Evaluate recruitment heuristic module on manually labeled job posts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.recruitment import analyze  # noqa: E402

DATA_PATH = ROOT / "data" / "recruitment_eval" / "labeled_posts.json"
RESULTS_PATH = ROOT / "data" / "recruitment_eval" / "results.json"


def _predict_scam(result: dict) -> bool:
    return "Fake" in result.get("prediction", "")


def _metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    n = len(y_true)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
    }


def main() -> None:
    samples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []
    y_true: list[bool] = []
    y_pred: list[bool] = []

    for sample in samples:
        result = analyze(
            job_text=sample.get("job_text", ""),
            company_name=sample.get("company_name", ""),
            website_url=sample.get("website_url", ""),
        )
        actual_scam = sample["label"] == "scam"
        predicted_scam = _predict_scam(result)
        y_true.append(actual_scam)
        y_pred.append(predicted_scam)
        rows.append(
            {
                "id": sample["id"],
                "label": sample["label"],
                "predicted": "scam" if predicted_scam else "real",
                "correct": actual_scam == predicted_scam,
                "fraud_risk_score": result["fraud_risk_score"],
                "trust_score": result["trust_score"],
                "website_status": result["website_status"],
                "indicators": result["indicators"],
            }
        )

    metrics = _metrics(y_true, y_pred)
    scam_n = sum(1 for s in samples if s["label"] == "scam")
    real_n = len(samples) - scam_n
    output = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "class_balance": {"scam": scam_n, "real": real_n},
        "threshold": "fraud_risk_score >= 50 => scam",
        "metrics": metrics,
        "predictions": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Recruitment heuristic evaluation")
    print(f"  Samples: {metrics['n']} (real={real_n}, scam={scam_n})")
    print(f"  Accuracy:  {metrics['accuracy']}%")
    print(f"  Precision: {metrics['precision']}%")
    print(f"  Recall:    {metrics['recall']}%")
    print(f"  F1:        {metrics['f1']}%")
    print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"  Results written to {RESULTS_PATH.relative_to(ROOT)}")

    errors = [r for r in rows if not r["correct"]]
    if errors:
        print("\nMisclassified:")
        for row in errors:
            print(f"  {row['id']}: expected {row['label']}, got {row['predicted']} (risk={row['fraud_risk_score']})")


if __name__ == "__main__":
    main()
