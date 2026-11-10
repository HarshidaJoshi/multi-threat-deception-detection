"""
Explainable AI service layer — Grad-CAM, LIME, and SHAP-style explanations.

Used by deepfake and misinformation detection pipelines via utils.orchestration.
"""

from utils.explainability import (
    build_explanation_summary,
    generate_gradcam_heatmap,
    lime_text_explanation,
    shap_feature_importance,
)

__all__ = [
    "build_explanation_summary",
    "generate_gradcam_heatmap",
    "lime_text_explanation",
    "shap_feature_importance",
]
