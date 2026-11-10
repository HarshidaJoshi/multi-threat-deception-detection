from django import forms
from django.core.exceptions import ValidationError
from django.shortcuts import render

from apps.reports.models import AnalysisRecord
from apps.reports.services import persist_analysis
from utils.model_registry import DEFAULT_MISINFO_MODEL, fake_news_tfidf_available, misinfo_choices
from utils.orchestration import analyze_misinformation, analyze_recruitment
from utils.text_ml import FAKE_NEWS_MIN_INPUT_CHARS, validate_fake_news_article


class MisinformationForm(forms.Form):
    model = forms.ChoiceField(
        choices=misinfo_choices(),
        initial=DEFAULT_MISINFO_MODEL,
        label="Detection model",
        widget=forms.Select(attrs={"class": "input-field"}),
    )
    text = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 10,
                "class": "input-field",
                "id": "fake-news-text",
                "minlength": FAKE_NEWS_MIN_INPUT_CHARS,
                "placeholder": "Paste the full news article (not headline only). Minimum 500 characters.",
            }
        ),
        label="News / social post text",
        help_text=(
            f"Minimum {FAKE_NEWS_MIN_INPUT_CHARS} characters — copy the complete article body for reliable Real/Fake prediction."
        ),
    )

    def clean_text(self):
        text = self.cleaned_data["text"]
        err = validate_fake_news_article(text)
        if err:
            raise ValidationError(err)
        return text


class RecruitmentForm(forms.Form):
    job_text = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 6, "class": "input-field"}),
        required=False,
        label="Job description",
    )
    company_name = forms.CharField(
        required=False,
        label="Company name",
        widget=forms.TextInput(attrs={"class": "input-field"}),
    )
    website_url = forms.CharField(
        required=False,
        label="Company website URL",
        widget=forms.TextInput(attrs={"class": "input-field"}),
    )


def index_view(request):
    return render(request, "misinformation/index.html")


def fake_news_view(request):
    result = None
    record_ref = None
    form = MisinformationForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        result = analyze_misinformation(
            form.cleaned_data["text"],
            model_key=form.cleaned_data["model"],
        )
        record_ref = persist_analysis(
            request,
            module=AnalysisRecord.Module.MISINFORMATION,
            input_type=AnalysisRecord.InputType.TEXT,
            label=result["prediction"],
            confidence=result["confidence"],
            result=result,
        )

    return render(
        request,
        "misinformation/fake_news.html",
        {
            "form": form,
            "result": result,
            "record_ref": record_ref,
            "tfidf_ready": fake_news_tfidf_available(),
            "fake_news_min_chars": FAKE_NEWS_MIN_INPUT_CHARS,
        },
    )


def recruitment_scam_view(request):
    result = None
    record_ref = None
    form = RecruitmentForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        result = analyze_recruitment(
            job_text=form.cleaned_data["job_text"],
            company_name=form.cleaned_data["company_name"],
            website_url=form.cleaned_data["website_url"],
        )
        record_ref = persist_analysis(
            request,
            module=AnalysisRecord.Module.RECRUITMENT,
            input_type=AnalysisRecord.InputType.FORM,
            label=result["prediction"],
            confidence=result["confidence"],
            trust_score=result.get("trust_score"),
            result=result,
        )

    return render(
        request,
        "misinformation/recruitment_scam.html",
        {"form": form, "result": result, "record_ref": record_ref},
    )
