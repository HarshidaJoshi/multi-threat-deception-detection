"""Analysis pipelines for each module."""

from utils import deepfake_models, recruitment, text_models
from utils.explainability import (
    build_explanation_summary,
    generate_gradcam_heatmap,
    lime_text_explanation,
    shap_feature_importance,
)


def analyze_deepfake(image_path: str, model_key: str = "efficientnet") -> dict:
    result = deepfake_models.analyze(image_path, model_key=model_key)
    heatmap = generate_gradcam_heatmap(image_path, result["fake_score"])
    return {
        "module": "deepfake",
        **result,
        "prediction": result["label"],
        "indicators": result["indicators"],
        "explainability": {
            "grad_cam": heatmap,
            "summary": build_explanation_summary(result["indicators"], "deepfake detection"),
            "reasons": result["indicators"],
        },
    }


def analyze_misinformation(text: str, model_key: str = "tfidf") -> dict:
    result = text_models.analyze(text, model_key=model_key)
    # Exact word contributions from the trained linear model; none are shown for models that cannot provide them.
    contributions = result.get("token_contributions") or []
    return {
        "module": "misinformation",
        **result,
        "prediction": result["label"],
        "indicators": result["indicators"],
        "explainability": {
            "lime_tokens": contributions,
            "shap_features": [],
            "summary": build_explanation_summary(result["indicators"], "misinformation detection"),
        },
    }


def analyze_recruitment(job_text: str = "", company_name: str = "", website_url: str = "") -> dict:
    return recruitment.analyze(job_text, company_name, website_url)
