from django.conf import settings
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, render

from utils.reports import build_report_sections, build_report_sections_from_item, generate_pdf_bytes

from .models import AnalysisRecord
from .services import get_demo_history


def history_list(request):
    if settings.DEMO_MODE:
        records = get_demo_history(request)
        return render(
            request,
            "reports/history.html",
            {"records": records, "demo_mode": True},
        )

    qs = AnalysisRecord.objects.all()
    if request.user.is_authenticated:
        qs = qs.filter(user=request.user) | qs.filter(user__isnull=True)
    records = qs[:100]
    return render(
        request,
        "reports/history.html",
        {"records": records, "demo_mode": False},
    )


def history_detail(request, record_id):
    if settings.DEMO_MODE:
        for item in get_demo_history(request):
            if str(item.get("id")) == str(record_id):
                return render(
                    request,
                    "reports/detail.html",
                    {"record": item, "demo_mode": True, "result": item.get("result", {})},
                )
        raise Http404("Demo analysis not found")

    record = get_object_or_404(AnalysisRecord, pk=record_id)
    return render(
        request,
        "reports/detail.html",
        {
            "record": record,
            "demo_mode": False,
            "result": record.result_json,
        },
    )


def download_report(request, record_id):
    if settings.DEMO_MODE:
        for item in get_demo_history(request):
            if str(item.get("id")) == str(record_id):
                result = item.get("result", {})
                sections = build_report_sections_from_item(item, result)
                pdf = generate_pdf_bytes(f"{item.get('module', 'Analysis')} Report", sections)
                response = HttpResponse(pdf, content_type="application/pdf")
                response["Content-Disposition"] = f'attachment; filename="xai_verify_{record_id}.pdf"'
                return response
        raise Http404("Demo analysis not found")

    record = get_object_or_404(AnalysisRecord, pk=record_id)
    if record.report_file:
        response = HttpResponse(record.report_file.read(), content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="{record.report_file.name.split("/")[-1]}"'
        return response

    sections = build_report_sections(record)
    pdf = generate_pdf_bytes(f"{record.get_module_display()} Report", sections)
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="xai_verify_{record.id}.pdf"'
    return response

