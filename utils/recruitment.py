"""Recruitment scam analysis."""

from typing import Any

from utils import ml_utils, text_ml
from utils.explainability import lime_text_explanation, shap_feature_importance


def analyze(job_text: str = "", company_name: str = "", website_url: str = "") -> dict[str, Any]:
    fraud_score = 50.0
    indicators: list[str] = []
    features: dict[str, float] = {}
    company_registered = False
    payment_demand = False

    if job_text.strip():
        job = ml_utils.recruitment_job_forensics(job_text)
        fraud_score += job["risk_delta"]
        payment_demand = job["payment_demand"]
        indicators.extend(job["indicators"])
        if payment_demand:
            features["payment_demand"] = 0.45

    company = ml_utils.verify_company(company_name) if company_name.strip() else None
    if company:
        if company.get("registered"):
            company_registered = True
            fraud_score -= 18
            indicators.append(company["message"])
            features["company_verified"] = -0.35
        else:
            fraud_score += 20
            indicators.append(company["message"])
            features["company_unverified"] = 0.4
            if ml_utils.generic_company_name(company_name):
                fraud_score += 10
                indicators.append(f"Generic company name: '{company_name.strip()}'")

    domain = ml_utils.extract_domain(website_url)
    if domain:
        if domain.endswith((".xyz", ".top", ".click", ".tk")):
            fraud_score += 18
            indicators.append(f"Suspicious TLD: {domain}")
            features["suspicious_tld"] = 0.35
        elif not company_registered:
            fraud_score += 10
            indicators.append(f"Unverified domain: {domain}")
    else:
        fraud_score += 10
        indicators.append("No website URL provided")

    if payment_demand and not company_registered:
        fraud_score += 12
        indicators.append("Payment demand + unverified employer")

    fraud_score = ml_utils.clamp_score(fraud_score)
    trust_score = round(100 - fraud_score, 2)
    label = "Fake Recruitment Post" if fraud_score >= 50 else "Real Recruitment Post"
    lime = lime_text_explanation(job_text, fraud_score) if job_text else []
    shap_items = shap_feature_importance(features)
    method = "heuristic rules"
    if job_text.strip() and text_ml.is_available(text_ml.RECRUITMENT_FILE):
        # Trained model on EMSCAD (F1 ~0.80 on held-out data; the old hand-written rules scored ~0.06).
        pred = text_ml.predict_recruitment(job_text)
        ml_fraud = pred["fraud_score"]
        fraud_score = ml_utils.clamp_score(ml_fraud)
        trust_score = round(100 - fraud_score, 2)
        label = "Fake Recruitment Post" if ml_fraud >= 50 else "Real Recruitment Post"
        lime = pred["token_contributions"]
        shap_items = []
        method = "TF-IDF + Logistic Regression trained on EMSCAD"
        indicators = [f"Trained model scam probability: {ml_fraud:.1f}% (decision threshold tuned on validation data)"] + [
            f"Company/website check: {i}" for i in indicators if not i.startswith(("Scam phrase", "Urgency", "Upfront", "Fee ", "Payment instruction"))
        ]
        if pred["n_words"] < 40:
            indicators.append("Very short job text - the model was trained on full postings, so treat the result with caution.")

    return {
        "module": "recruitment",
        "prediction": label,
        "confidence": max(fraud_score, trust_score),
        "fraud_risk_score": fraud_score,
        "trust_score": trust_score,
        "employer_in_reference_list": company_registered,
        "website_status": "Trusted" if company_registered and trust_score >= 65 else "Suspicious",
        "domain": domain,
        "indicators": indicators[:10],
        "explainability": {
            "lime_tokens": lime,
            "shap_features": shap_items,
            "method": method,
            "reasons": indicators[:6],
            "summary": (
                "Likely scam because: " + "; ".join(indicators[:3])
                if fraud_score >= 50
                else "Appears trustworthy based on available signals."
            ),
        },
    }
