from django.contrib import admin

from .models import AnalysisRecord


@admin.register(AnalysisRecord)
class AnalysisRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "module", "label", "confidence", "user", "created_at")
    list_filter = ("module", "created_at")
    search_fields = ("label", "session_key")
    readonly_fields = ("result_json", "created_at")
