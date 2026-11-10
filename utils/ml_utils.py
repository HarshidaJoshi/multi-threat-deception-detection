"""Shared helpers for text, image, and recruitment analysis."""

import re
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ExifTags

_FAKE_WORDS = {
    "fake": 8, "false": 7, "hoax": 10, "scam": 10, "fraud": 10,
    "conspiracy": 12, "misleading": 9, "shocking": 6, "breaking": 5,
    "urgent": 6, "guaranteed": 7, "miracle": 8, "click here": 8,
    "registration fee": 15, "pay now": 12, "immediate joining": 14,
}

_CREDIBLE = ["according to", "published", "research", "study", "official", "verified"]

_REGISTERED_COMPANIES = {
    "tata", "reliance", "infosys", "tcs", "wipro", "hdfc", "icici", "sbi",
    "google", "microsoft", "amazon", "flipkart", "paytm", "airtel", "jio",
}

_SCAM_PHRASES = [
    "registration fee", "interview fee", "guaranteed placement",
    "pay rs", "pay ₹", "immediate joining without interview",
    "work from home earn", "whatsapp earning", "security deposit",
]

_RECRUITMENT_PAYMENT_PATTERNS: list[tuple[str, int, str]] = [
    (r"\bpay\s*\d+\s*(rs|₹|rupees?|/-)?", 25, "Upfront payment demanded"),
    (r"\b(registration|interview|joining)\s*(fee|fees|charge|deposit)\b", 20, "Fee mentioned in job post"),
    (r"\bpay\s*(rs|₹|now|before|to\s+apply)\b", 16, "Payment instruction in advert"),
]


def clamp_score(value: float, low: float = 8, high: float = 92) -> float:
    return float(max(low, min(high, value)))


def _has_exif(img: Image.Image) -> bool:
    try:
        exif = img.getexif()
        if not exif:
            return False
        tags = {ExifTags.TAGS.get(k, k) for k in exif}
        return bool(tags & {"Make", "Model", "DateTimeOriginal", "DateTime"})
    except Exception:
        return False


def image_forensics(filepath: str) -> dict:
    """Lightweight image signals used when no trained model is loaded."""
    indicators: list[str] = []
    fake_score = 50.0

    img = Image.open(filepath)
    has_exif = _has_exif(img)
    ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
    width, height = img.size

    arr = np.array(img.convert("RGB").resize((512, 512)), dtype=np.float32)
    gray = np.mean(arr, axis=2)
    noise = float(np.std(gray))
    lap = float(gray.var())

    if has_exif:
        fake_score -= 20
        indicators.append("Camera EXIF present")
    elif ext in ("png", "webp"):
        fake_score += 6
        indicators.append("PNG/WebP without camera EXIF")
    else:
        fake_score += 4
        indicators.append("No camera EXIF metadata")

    if lap >= 60:
        fake_score -= 10
        indicators.append(f"Strong image detail ({lap:.0f})")
    elif lap < 25:
        fake_score += 8
        indicators.append("Very soft / smooth image detail")

    if noise >= 2.0:
        fake_score -= 6
        indicators.append("Natural texture variation")
    elif noise < 0.8:
        fake_score += 6
        indicators.append("Low texture noise")

    if max(width, height) >= 1200 and has_exif:
        fake_score -= 6
        indicators.append(f"High-resolution photo ({width}x{height})")

    fake_score = clamp_score(fake_score)
    return {
        "fake_score": fake_score,
        "real_score": round(100 - fake_score, 2),
        "indicators": indicators,
        "features": {"no_exif": 0.15 if not has_exif else -0.1},
    }


def text_forensics(text: str) -> dict:
    text_lower = text.lower()
    indicators: list[str] = []
    fake_score = 50.0

    found = [w for w in _FAKE_WORDS if w in text_lower]
    for w in found:
        fake_score += _FAKE_WORDS[w]
    if found:
        indicators.append("Suspicious phrases: " + ", ".join(found[:5]))
    else:
        fake_score -= 10
        indicators.append("No suspicious phrases")

    if any(p in text_lower for p in _CREDIBLE):
        fake_score -= 18
        indicators.append("Credible language detected")

    fake_score = clamp_score(fake_score)
    return {
        "fake_score": fake_score,
        "real_score": round(100 - fake_score, 2),
        "indicators": indicators,
        "features": {f"phrase_{i}": 0.1 for i, _ in enumerate(found[:5])},
    }


def extract_domain(url: str) -> str:
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def generic_company_name(name: str) -> bool:
    lower = name.strip().lower()
    if not lower:
        return False
    if any(lower.startswith(p) for p in ("abc", "xyz", "test", "dummy", "sample")):
        return True
    return len(lower.replace(" ", "")) <= 6


def recruitment_job_forensics(job_text: str) -> dict:
    text_lower = job_text.lower().strip()
    indicators: list[str] = []
    risk_delta = 0.0
    payment_demand = False

    if not text_lower:
        return {"risk_delta": 0.0, "indicators": [], "payment_demand": False}

    for pattern, points, message in _RECRUITMENT_PAYMENT_PATTERNS:
        if re.search(pattern, text_lower):
            risk_delta += points
            payment_demand = True
            indicators.append(message)

    for phrase in _SCAM_PHRASES:
        if phrase in text_lower:
            risk_delta += 14
            indicators.append(f"Scam phrase: '{phrase}'")

    if re.search(r"\b(urgent|immediate|hurry|act\s+now)\b", text_lower):
        risk_delta += 8
        indicators.append("Urgency language")

    return {"risk_delta": risk_delta, "indicators": indicators, "payment_demand": payment_demand}


def verify_company(name: str) -> dict:
    if not name.strip():
        return {"registered": None, "message": "No company provided"}
    lower = name.strip().lower()
    for reg in _REGISTERED_COMPANIES:
        if reg in lower:
            return {"registered": True, "message": f"'{name}' matches reference employer list."}
    return {"registered": False, "message": f"'{name}' not in reference employer list."}
