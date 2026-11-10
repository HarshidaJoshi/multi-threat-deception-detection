"""Fake news detection — multiple transformer and heuristic model variants."""

from __future__ import annotations

from typing import Any

from utils import ml_utils, text_ml
from utils.model_registry import (
    DEFAULT_MISINFO_MODEL,
    fake_news_tfidf_available,
    get_misinfo_model,
    misinfo_model_is_project_trained,
    resolve_misinfo_hf_model,
    resolve_misinfo_key,
)

_pipelines: dict[str, Any] = {}


def _get_pipeline(hf_model: str):
    if hf_model not in _pipelines:
        from transformers import pipeline

        _pipelines[hf_model] = pipeline(
            "text-classification",
            model=hf_model,
            truncation=True,
            max_length=512,
        )
    return _pipelines[hf_model]


def _label_to_fake_score(label: str, confidence_pct: float) -> float:
    label_u = label.upper()
    if any(k in label_u for k in ("FAKE", "FALSE", "NOT_REAL", "MISINFORMATION")):
        return ml_utils.clamp_score(confidence_pct)
    if any(k in label_u for k in ("REAL", "TRUE", "GENUINE", "NOT_FAKE")):
        return ml_utils.clamp_score(100 - confidence_pct)
    # Legacy HF checkpoints only (elozano: 0=Fake). Project WELFake models use REAL/FAKE strings above.
    if label_u in {"LABEL_0", "0"}:
        return ml_utils.clamp_score(confidence_pct)
    if label_u in {"LABEL_1", "1"}:
        return ml_utils.clamp_score(100 - confidence_pct)
    return ml_utils.clamp_score(100 - confidence_pct)


def _run_transformer(text: str, hf_model: str, model_key: str) -> tuple[float, list[str], str]:
    # Same cleaning as WELFake/ISOT training — strips URLs, "Read more", source tags, etc.
    cleaned = text_ml.clean_news(text)
    out = _get_pipeline(hf_model)(cleaned[:2000])[0]
    model_fake = _label_to_fake_score(str(out["label"]), float(out["score"]) * 100)
    backend = "local_welfake" if misinfo_model_is_project_trained(model_key) else "huggingface"
    indicators = [f"Transformer label: {out['label']} ({float(out['score']) * 100:.1f}%)"]
    if backend == "huggingface":
        indicators.append(
            "Using a third-party Hugging Face model — run scripts/train_fake_news_transformers.py "
            "to fine-tune BERT/RoBERTa on WELFake and replace this."
        )
    return model_fake, indicators, backend


def _word_count(text: str) -> int:
    return len(text.strip().split())


def _short_input_warning(n_words: int) -> str:
    return (
        f"Short input ({n_words} words): classifiers were trained on full news articles. "
        "Scores can be overconfident and often disagree across models — paste a full article "
        "or treat this as a demo only."
    )


def _transformer_inference(
    text: str,
    transformer_key: str,
    extra_indicators: list[str] | None = None,
) -> tuple[float, list[str], str, str]:
    """Run BERT/RoBERTa; returns (fake_score, indicators, backend, reliability)."""
    hf_model = resolve_misinfo_hf_model(transformer_key)
    if not hf_model:
        raise ValueError(f"No Hugging Face model configured for {transformer_key}")
    model_fake, indicators, backend = _run_transformer(text, hf_model, transformer_key)
    if extra_indicators:
        indicators = list(extra_indicators) + indicators
    reliability = "normal"
    n_words = _word_count(text_ml.clean_news(text))
    if n_words < 30:
        reliability = "low"
        indicators.append(_short_input_warning(n_words))
    if transformer_key == "bert" and n_words < 80 and not misinfo_model_is_project_trained("bert"):
        indicators.append(
            "Third-party BERT often labels short headlines as Fake — paste the full article or train "
            "project BERT: python scripts/train_fake_news_transformers.py"
        )
    return model_fake, indicators, backend, reliability


def analyze(text: str, model_key: str = DEFAULT_MISINFO_MODEL) -> dict[str, Any]:
    requested_key = resolve_misinfo_key(model_key)
    meta = get_misinfo_model(requested_key)
    forensic = ml_utils.text_forensics(text)
    backend = "huggingface"
    indicators: list[str] = []

    token_contributions: list[dict[str, Any]] = []
    reliability = "normal"
    model_key = requested_key

    if requested_key == "tfidf":
        if text_ml.is_available(text_ml.FAKE_NEWS_FILE):
            pred = text_ml.predict_fake_news(text)
            model_fake = pred["fake_score"]
            token_contributions = pred["token_contributions"]
            backend = "trained_sklearn"
            indicators = [f"TF-IDF + Logistic Regression decision score: {pred['decision_score']:+.2f}"]
            if pred["n_words"] < 30:
                reliability = "low"
                indicators.append(
                    f"Short input ({pred['n_words']} words): this model was trained on full articles and is "
                    "close to chance on short claims - treat the result as unreliable."
                )
        else:
            indicators.append(
                "TF-IDF model not found in trained_models/ — using BERT until you run: "
                "python manage.py train_fake_news"
            )
            model_key = "bert"
            meta = get_misinfo_model("bert")
            try:
                model_fake, indicators, backend, reliability = _transformer_inference(
                    text, "bert", extra_indicators=indicators
                )
            except Exception as exc:
                backend = "rule_fallback"
                model_fake = forensic["fake_score"]
                reliability = "low"
                indicators.append(f"Transformer unavailable ({exc}) — keyword heuristics only (unreliable).")
    elif requested_key == "forensics":
        model_fake = forensic["fake_score"]
        backend = "rule_forensics"
        indicators = list(forensic["indicators"])
        n_words = _word_count(text)
        if n_words < 30:
            reliability = "low"
            indicators.append(_short_input_warning(n_words))
        indicators.append(
            "Keyword heuristics only: no match on a scam-word list does not verify the story as true."
        )
    else:
        try:
            model_fake, indicators, backend, reliability = _transformer_inference(text, requested_key)
        except Exception as exc:
            if fake_news_tfidf_available():
                pred = text_ml.predict_fake_news(text)
                model_fake = pred["fake_score"]
                token_contributions = pred["token_contributions"]
                backend = "trained_sklearn_fallback"
                model_key = "tfidf"
                meta = get_misinfo_model("tfidf")
                reliability = "normal"
                indicators = [
                    f"{requested_key} failed ({exc}); fell back to TF-IDF + LogReg.",
                    f"Decision score: {pred['decision_score']:+.2f}",
                ]
            else:
                backend = "rule_fallback"
                model_fake = forensic["fake_score"]
                reliability = "low"
                indicators = [
                    f"{meta['architecture']} unavailable ({exc}).",
                    "Install transformers/torch or train models — see python manage.py train_fake_news",
                    *forensic["indicators"][:3],
                ]

    fake_score = ml_utils.clamp_score(model_fake)
    real_score = round(100 - fake_score, 2)
    label = "Fake News" if fake_score > 50 else "Real News"

    return {
        "label": label,
        "fake_score": fake_score,
        "real_score": real_score,
        "confidence": round(max(fake_score, real_score), 2),
        "indicators": indicators,
        "model_key": model_key,
        "model_used": meta["architecture"],
        "model_label": meta["label"],
        "model_description": meta["description"],
        "inference_backend": backend,
        "features": forensic.get("features", {}),
        "token_contributions": token_contributions,
        "reliability": reliability,
    }
