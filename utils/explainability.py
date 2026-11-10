"""Explainability helpers."""

import base64
import io
import re
from typing import Any

import numpy as np
from PIL import Image


def generate_gradcam_heatmap(image_path: str, fake_score: float = 0.0) -> str:
    """Real Grad-CAM overlay (base64 PNG) from the trained classifier; empty string if unavailable.

    No placeholder is drawn when the trained model cannot be loaded: a fake heatmap would
    misrepresent what the model attends to.
    """
    try:
        from utils import deepfake_classifier, gradcam

        if not deepfake_classifier.is_trained_model_available():
            return ""
        return gradcam.gradcam_overlay_base64(deepfake_classifier.get_loaded_model(), image_path)["png_base64"]
    except Exception:
        return ""


def lime_text_explanation(text: str, fake_score: float) -> list[dict[str, Any]]:
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    if not tokens:
        return []
    suspicious = {"fake", "scam", "fraud", "urgent", "fee", "pay", "guaranteed", "free"}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    items = []
    for word, count in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        importance = 0.05 * count + (0.35 if word in suspicious else 0)
        importance *= fake_score / 50.0
        items.append({
            "token": word,
            "importance": round(max(-1, min(1, importance)), 4),
            "direction": "fake" if importance > 0 else "real",
        })
    return sorted(items, key=lambda x: abs(x["importance"]), reverse=True)[:10]


def shap_feature_importance(features: dict[str, float]) -> list[dict[str, Any]]:
    total = sum(abs(v) for v in features.values()) or 1.0
    return sorted(
        [
            {
                "feature": name,
                "value": round(value, 4),
                "contribution_pct": round(abs(value) / total * 100, 2),
                "direction": "risk" if value > 0 else "trust",
            }
            for name, value in features.items()
        ],
        key=lambda x: abs(x["value"]),
        reverse=True,
    )


def build_explanation_summary(indicators: list[str], module: str) -> str:
    if not indicators:
        return f"No strong signals for {module}."
    return f"Key signals: " + "; ".join(indicators[:4]) + "."
