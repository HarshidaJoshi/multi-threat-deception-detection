"""Available ML model variants for UI selection and comparison."""

from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAINED_DIR = _PROJECT_ROOT / "trained_models"
_BERT_LOCAL = _TRAINED_DIR / "fake_news_bert_welfake"
_ROBERTA_LOCAL = _TRAINED_DIR / "fake_news_roberta_welfake"
_HF_BERT = "elozano/bert-base-cased-fake-news"
_HF_ROBERTA = "hamzab/roberta-fake-news-classification"


_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _checkpoint_has_weights(checkpoint_dir: Path) -> bool:
    return any((checkpoint_dir / name).is_file() for name in _WEIGHT_FILES)


def _local_checkpoint(name: str) -> Path | None:
    """Return local HF checkpoint only when config and weight files are present."""
    d = _TRAINED_DIR / name
    if not (d / "config.json").is_file():
        return None
    if not _checkpoint_has_weights(d):
        return None
    return d


def fake_news_tfidf_available() -> bool:
    return (_TRAINED_DIR / "fake_news_welfake_logreg.joblib").is_file()


def _read_test_acc(sidecar: Path) -> float | None:
    if not sidecar.is_file():
        return None
    try:
        return float(json.loads(sidecar.read_text(encoding="utf-8")).get("test_accuracy", 0)) * 100
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def resolve_misinfo_hf_model(key: str) -> str | None:
    if key == "bert":
        loc = _local_checkpoint("fake_news_bert_welfake")
        return str(loc) if loc else _HF_BERT
    if key == "roberta":
        loc = _local_checkpoint("fake_news_roberta_welfake")
        return str(loc) if loc else _HF_ROBERTA
    return None


def misinfo_model_is_project_trained(key: str) -> bool:
    if key == "bert":
        return _local_checkpoint("fake_news_bert_welfake") is not None
    if key == "roberta":
        return _local_checkpoint("fake_news_roberta_welfake") is not None
    return key == "tfidf"

DEEPFAKE_MODELS: dict[str, dict] = {
    "efficientnet": {
        "label": "ResNet50 + MLP — trained classifier (95.4% test accuracy)",
        "description": "ImageNet ResNet50 backbone with an MLP head trained on 100,000 real and AI-generated face images; 95.4% accuracy on 20,000 held-out images. Grad-CAM explanations come from the model itself.",
        "architecture": "ResNet50 + MLP head",
    },
    "forensics": {
        "label": "Forensics — heuristic analysis",
        "description": "EXIF, noise, and texture heuristics without deep learning.",
        "architecture": "Rule-based forensics",
    },
}

MISINFORMATION_MODELS: dict[str, dict] = {
    "tfidf": {
        "label": "TF-IDF + Logistic Regression — trained on WELFake",
        "description": (
            "Trained on 44k cleaned news articles (WELFake). ~96% accuracy on WELFake, ~82% on a different "
            "news corpus, and near chance on short claims (LIAR) - use on full articles, not one-line claims."
        ),
        "architecture": "TF-IDF + Logistic Regression",
        "hf_model": None,
    },
    "bert": {
        "label": "BERT — Hugging Face comparator (not project-trained)",
        "description": (
            "Third-party elozano/bert-base-cased-fake-news. Works on full articles; on short headlines "
            "it often predicts Fake (~50% accuracy on one-line claims). For pasted Google News, use "
            "TF-IDF or include the full story text."
        ),
        "architecture": "BERT",
        "hf_model": "elozano/bert-base-cased-fake-news",
    },
    "roberta": {
        "label": "RoBERTa — Hugging Face comparator (not project-trained)",
        "description": (
            "Third-party hamzab/roberta-fake-news-classification. Better on short text than BERT but "
            "still trained elsewhere — disagree with TF-IDF is normal. Use full articles when possible."
        ),
        "architecture": "RoBERTa",
        "hf_model": "hamzab/roberta-fake-news-classification",
    },
    "forensics": {
        "label": "Forensics — keyword heuristics",
        "description": "Rule-based suspicious keyword and credibility scoring.",
        "architecture": "Rule-based forensics",
        "hf_model": None,
    },
}

DEFAULT_DEEPFAKE_MODEL = "efficientnet"
DEFAULT_MISINFO_MODEL = "tfidf"

# Legacy keys from earlier tiny-model builds
_MISINFO_ALIASES = {
    "bert_tiny": "bert",
    "roberta_tiny": "roberta",
}


def deepfake_choices() -> list[tuple[str, str]]:
    return [(k, v["label"]) for k, v in DEEPFAKE_MODELS.items()]


def misinfo_choices() -> list[tuple[str, str]]:
    return [(k, v["label"]) for k, v in MISINFORMATION_MODELS.items()]


def get_deepfake_model(key: str) -> dict:
    return DEEPFAKE_MODELS.get(key, DEEPFAKE_MODELS[DEFAULT_DEEPFAKE_MODEL])


def resolve_misinfo_key(key: str) -> str:
    key = _MISINFO_ALIASES.get(key, key)
    if key in MISINFORMATION_MODELS:
        return key
    return DEFAULT_MISINFO_MODEL


def get_misinfo_model(key: str) -> dict:
    key = resolve_misinfo_key(key)
    meta = dict(MISINFORMATION_MODELS[key])
    if key == "bert":
        meta["hf_model"] = resolve_misinfo_hf_model("bert")
        if misinfo_model_is_project_trained("bert"):
            acc = _read_test_acc(_TRAINED_DIR / "fake_news_bert_welfake.meta.json")
            meta["label"] = "BERT — fine-tuned on WELFake (project)"
            meta["description"] = (
                "Fine-tuned on the same cleaned WELFake articles as TF-IDF (0=real, 1=fake). "
                + (f"Held-out WELFake test accuracy ~{acc:.1f}%. " if acc else "")
                + "Still not ground-truth fact checking — use full article text."
            )
        else:
            meta["hf_model"] = _HF_BERT
    elif key == "roberta":
        meta["hf_model"] = resolve_misinfo_hf_model("roberta")
        if misinfo_model_is_project_trained("roberta"):
            acc = _read_test_acc(_TRAINED_DIR / "fake_news_roberta_welfake.meta.json")
            meta["label"] = "RoBERTa — fine-tuned on WELFake (project)"
            meta["description"] = (
                "Fine-tuned on WELFake with the same splits as the TF-IDF model. "
                + (f"Held-out WELFake test accuracy ~{acc:.1f}%. " if acc else "")
                + "Use full articles for current news."
            )
        else:
            meta["hf_model"] = _HF_ROBERTA
    meta["project_trained"] = misinfo_model_is_project_trained(key)
    return meta
