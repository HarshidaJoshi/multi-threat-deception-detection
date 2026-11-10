"""Deepfake / AI image detection — multiple model variants."""

from __future__ import annotations

from typing import Any

from utils import deepfake_classifier, ml_utils
from utils.model_registry import DEEPFAKE_MODELS, DEFAULT_DEEPFAKE_MODEL, get_deepfake_model


def is_trained_model_loaded() -> bool:
    return deepfake_classifier.is_trained_model_available()


def _label_from_score(fake_score: float, classifier_prob: float | None = None) -> str:
    if classifier_prob is not None and classifier_prob >= 0.5:
        return "AI Generated" if classifier_prob >= 0.55 else "Deepfake"
    if fake_score > 50:
        return "Deepfake"
    return "Real"


def analyze(image_path: str, model_key: str = DEFAULT_DEEPFAKE_MODEL) -> dict[str, Any]:
    model_key = model_key if model_key in DEEPFAKE_MODELS else DEFAULT_DEEPFAKE_MODEL
    meta = get_deepfake_model(model_key)
    forensic = ml_utils.image_forensics(image_path)
    trained = deepfake_classifier.is_trained_model_available()
    classifier_prob = None
    indicators: list[str] = []
    val_accuracy = None
    demo_training = False

    if model_key == "forensics":
        fake_score = forensic["fake_score"]
        backend = "forensics_only"
        indicators = list(forensic["indicators"])
        model_used = meta["architecture"]
    else:
        if trained:
            try:
                classifier_prob, indicators = deepfake_classifier.predict_fake_probability(image_path)
                fake_score = ml_utils.clamp_score(classifier_prob * 100)
                backend = "trained_classifier"
                model_meta = deepfake_classifier.get_model_metadata() or {}
                val_accuracy = model_meta.get("val_accuracy")
                demo_training = bool(model_meta.get("real_images", 0) <= 100 and model_meta.get("fake_images", 0) <= 100)
            except Exception as exc:
                fake_score = forensic["fake_score"]
                backend = "forensics_fallback"
                indicators = [f"EfficientNet error: {exc}", *forensic["indicators"][:2]]
        else:
            fake_score = forensic["fake_score"]
            backend = "forensics_fallback"
            indicators = [
                "EfficientNet not trained — run: python manage.py train_demo_deepfake --dataset datasets",
                *forensic["indicators"][:2],
            ]
        model_used = meta["architecture"]

    real_score = round(100 - fake_score, 2)
    label = _label_from_score(fake_score, classifier_prob)

    result: dict[str, Any] = {
        "label": label,
        "fake_score": fake_score,
        "real_score": real_score,
        "confidence": round(max(fake_score, real_score), 2),
        "indicators": indicators,
        "model_key": model_key,
        "model_used": model_used,
        "model_label": meta["label"],
        "model_description": meta["description"],
        "inference_backend": backend,
        "trained_model_loaded": trained and classifier_prob is not None,
        "classifier_fake_probability": round(classifier_prob * 100, 2) if classifier_prob is not None else None,
        "features": forensic.get("features", {}),
    }
    if val_accuracy is not None:
        result["model_val_accuracy"] = round(float(val_accuracy) * 100, 1)
        result["demo_training_data"] = demo_training
    return result
