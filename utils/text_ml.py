"""Trained TF-IDF + linear models for fake news and recruitment scams, with exact explanations.

For a linear model the decision score is  score = intercept + sum_i  w_i * x_i,  so each term's
contribution w_i * x_i is exact (no sampling / approximation as in LIME or kernel-SHAP). The same
quantity is what SHAP's LinearExplainer reports up to a constant baseline shift.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("TEXT_MODELS_DIR", PROJECT_ROOT / "trained_models"))

FAKE_NEWS_FILE = "fake_news_welfake_logreg.joblib"
RECRUITMENT_FILE = "recruitment_text_logreg.joblib"

# Models were trained on full news articles; short snippets are unreliable.
FAKE_NEWS_MIN_INPUT_CHARS = 500
FAKE_NEWS_MIN_CLEANED_CHARS = 350

_cache: dict[str, dict] = {}

_REUTERS_DATELINE = re.compile(r"^[^\n]{0,80}?\(reuters\)\s*[-–—:]?\s*", re.I)
_LEAK_PATTERNS = [
    re.compile(r"\(reuters\)", re.I),
    re.compile(r"\breuters\b", re.I),
    re.compile(r"https?://\S+"),
    re.compile(r"\bpic\.twitter\.com/\S+"),
    re.compile(r"\b(featured|image)\s+(image|via|credit)[^.\n]{0,60}", re.I),
    re.compile(r"\b21st century wire says\b", re.I),
    # Common noise when copy-pasting from Google News / publisher pages (not in training corpora)
    re.compile(r"\bread\s+more\b[^.\n]{0,80}", re.I),
    re.compile(r"\bshare\s+(this\s+)?(article|story|on)\b[^.\n]{0,40}", re.I),
    re.compile(r"\bsubscribe\b[^.\n]{0,60}", re.I),
    re.compile(r"\ball\s+rights\s+reserved\b.*", re.I),
    re.compile(r"\b\d+\s+min(ute)?s?\s+read\b", re.I),
    re.compile(r"\b(getty\s+images|shutterstock|afp|associated\s+press)\b[^.\n]{0,40}", re.I),
]


def clean_news(text: str) -> str:
    """Same cleaning used at training time (removes source tags that leak the label)."""
    t = _REUTERS_DATELINE.sub("", text.strip())
    for p in _LEAK_PATTERNS:
        t = p.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def validate_fake_news_article(text: str) -> str | None:
    """Return an error message if ``text`` is too short for reliable detection, else None."""
    raw = text.strip()
    if len(raw) < FAKE_NEWS_MIN_INPUT_CHARS:
        return (
            f"Paste at least {FAKE_NEWS_MIN_INPUT_CHARS} characters of the full article "
            f"(you entered {len(raw)}). Headlines or one paragraph are not enough for accurate detection."
        )
    cleaned = clean_news(raw)
    if len(cleaned) < FAKE_NEWS_MIN_CLEANED_CHARS:
        return (
            f"After removing URLs and page boilerplate, only {len(cleaned)} characters remain. "
            f"Include the main story body (at least ~{FAKE_NEWS_MIN_CLEANED_CHARS} characters of article text)."
        )
    return None


def is_available(filename: str) -> bool:
    return (MODELS_DIR / filename).is_file()


def _load(filename: str) -> dict:
    if filename not in _cache:
        import joblib

        _cache[filename] = joblib.load(MODELS_DIR / filename)
    return _cache[filename]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _contributions(bundle: dict, text: str, top: int = 10) -> tuple[float, list[dict[str, Any]]]:
    vec, model = bundle["vectorizer"], bundle["model"]
    X = vec.transform([text]).tocoo()
    names = vec.get_feature_names_out()
    coef = model.coef_.ravel()
    items = [(names[j], float(v * coef[j])) for j, v in zip(X.col, X.data)]
    score = float(model.intercept_[0] + sum(c for _, c in items))
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    out = [
        {
            "token": w,
            "importance": round(c, 4),
            "weight": round(c, 4),
            "direction": "fake" if c > 0 else "real",
        }
        for w, c in items[:top]
    ]
    return score, out


def predict_fake_news(text: str) -> dict[str, Any]:
    """Probability (0-100) that ``text`` is fake news, plus exact word contributions."""
    bundle = _load(FAKE_NEWS_FILE)
    cleaned = clean_news(text)
    score, contribs = _contributions(bundle, cleaned)
    return {
        "fake_score": round(100 * _sigmoid(score), 2),
        "decision_score": round(score, 4),
        "token_contributions": contribs,
        "n_words": len(cleaned.split()),
    }


def predict_recruitment(job_text: str) -> dict[str, Any]:
    """Scam probability (0-100, 50 = the validation-tuned decision threshold) plus word contributions."""
    bundle = _load(RECRUITMENT_FILE)
    score, contribs = _contributions(bundle, re.sub(r"\s+", " ", job_text).strip())
    thr = float(bundle.get("threshold", 0.0))
    return {
        "fraud_score": round(100 * _sigmoid(score - thr), 2),
        "decision_score": round(score, 4),
        "threshold": round(thr, 4),
        "token_contributions": contribs,
        "n_words": len(job_text.split()),
    }
