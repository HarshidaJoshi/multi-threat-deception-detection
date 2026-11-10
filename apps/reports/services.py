from django.conf import settings
from django.utils import timezone

from .models import AnalysisRecord


def persist_analysis(
    request,
    *,
    module: str,
    input_type: str,
    label: str,
    confidence: float,
    result: dict,
    trust_score: float | None = None,
    upload=None,
) -> AnalysisRecord | dict | None:
    """Save analysis to MySQL, or session-only when DEMO_MODE is enabled."""
    if settings.DEMO_MODE:
        return _append_demo_session(request, module, label, confidence, trust_score, result)

    return AnalysisRecord.objects.create(
        user=request.user if request.user.is_authenticated else None,
        module=module,
        input_type=input_type,
        label=label,
        confidence=confidence,
        trust_score=trust_score,
        result_json=result,
        upload=upload,
        session_key=request.session.session_key or "",
    )


def _append_demo_session(
    request,
    module: str,
    label: str,
    confidence: float,
    trust_score: float | None,
    result: dict,
) -> dict:
    history = request.session.get(settings.DEMO_SESSION_KEY, [])
    demo_id = request.session.get("demo_id_counter", 0) + 1
    request.session["demo_id_counter"] = demo_id

    entry = {
        "id": demo_id,
        "module": module,
        "label": label,
        "confidence": confidence,
        "trust_score": trust_score,
        "result": result,
        "created_at": timezone.now().isoformat(),
    }
    history.insert(0, entry)
    request.session[settings.DEMO_SESSION_KEY] = history[: settings.DEMO_SESSION_LIMIT]
    request.session.modified = True
    return entry


def get_demo_history(request) -> list[dict]:
    return request.session.get(settings.DEMO_SESSION_KEY, [])
