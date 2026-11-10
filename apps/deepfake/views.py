import os
import uuid

from django import forms
from django.conf import settings
from django.shortcuts import render

from apps.reports.models import AnalysisRecord
from apps.reports.services import persist_analysis
from utils import deepfake_classifier
from utils.model_registry import DEFAULT_DEEPFAKE_MODEL, deepfake_choices
from utils.orchestration import analyze_deepfake


class DeepfakeForm(forms.Form):
    model = forms.ChoiceField(
        choices=deepfake_choices(),
        initial=DEFAULT_DEEPFAKE_MODEL,
        label="Detection model",
        widget=forms.Select(attrs={"class": "input-field"}),
    )
    file = forms.ImageField(
        label="Image",
        widget=forms.ClearableFileInput(attrs={"class": "input-field", "accept": "image/*"}),
    )


def analyze_view(request):
    result = None
    record_id = None
    form = DeepfakeForm(request.POST or None, request.FILES or None)

    if request.method == "POST" and form.is_valid():
        upload = form.cleaned_data["file"]
        model_key = form.cleaned_data["model"]
        ext = os.path.splitext(upload.name)[1] or ".jpg"
        temp_dir = settings.MEDIA_ROOT / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"deepfake_{uuid.uuid4().hex[:8]}{ext}"

        with open(temp_path, "wb") as dest:
            for chunk in upload.chunks():
                dest.write(chunk)

        try:
            result = analyze_deepfake(str(temp_path), model_key=model_key)
            record_ref = persist_analysis(
                request,
                module=AnalysisRecord.Module.DEEPFAKE,
                input_type=AnalysisRecord.InputType.FILE,
                label=result["prediction"],
                confidence=result["confidence"],
                result=result,
                upload=None if settings.DEMO_MODE else upload,
            )
            record_id = record_ref["id"] if isinstance(record_ref, dict) else getattr(record_ref, "pk", None)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return render(
        request,
        "deepfake/analyze.html",
        {
            "form": form,
            "result": result,
            "record_id": record_id,
            "trained_model_loaded": deepfake_classifier.is_trained_model_available(),
        },
    )
