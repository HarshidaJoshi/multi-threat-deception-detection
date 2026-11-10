from django.db import models


class AnalysisRecord(models.Model):
    class Module(models.TextChoices):
        DEEPFAKE = "deepfake", "Deepfake Detection"
        MISINFORMATION = "misinformation", "Fake News"
        RECRUITMENT = "recruitment", "Recruitment Scam (Misinformation)"

    class InputType(models.TextChoices):
        FILE = "file", "File upload"
        TEXT = "text", "Text input"
        FORM = "form", "Form fields"

    user = models.ForeignKey(
        "auth.User",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="analyses",
    )
    module = models.CharField(max_length=32, choices=Module.choices, db_index=True)
    input_type = models.CharField(max_length=16, choices=InputType.choices, default=InputType.FORM)
    label = models.CharField(max_length=128)
    confidence = models.FloatField()
    trust_score = models.FloatField(null=True, blank=True)
    result_json = models.JSONField(default=dict)
    upload = models.FileField(upload_to="uploads/%Y/%m/%d/", null=True, blank=True)
    report_file = models.FileField(upload_to="reports/%Y/%m/%d/", null=True, blank=True)
    session_key = models.CharField(max_length=64, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "analysis_analysisrecord"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.get_module_display()} — {self.label} ({self.created_at:%Y-%m-%d})"

    @property
    def is_fake_label(self) -> bool:
        label = self.label.lower()
        return "fake" in label or "deepfake" in label or "ai generated" in label
