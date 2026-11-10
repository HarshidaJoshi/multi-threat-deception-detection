import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name="AnalysisRecord",
                    fields=[
                        ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                        (
                            "module",
                            models.CharField(
                                choices=[
                                    ("deepfake", "Deepfake Detection"),
                                    ("misinformation", "Fake News"),
                                    ("recruitment", "Recruitment Scam (Misinformation)"),
                                ],
                                db_index=True,
                                max_length=32,
                            ),
                        ),
                        (
                            "input_type",
                            models.CharField(
                                choices=[("file", "File upload"), ("text", "Text input"), ("form", "Form fields")],
                                default="form",
                                max_length=16,
                            ),
                        ),
                        ("label", models.CharField(max_length=128)),
                        ("confidence", models.FloatField()),
                        ("trust_score", models.FloatField(blank=True, null=True)),
                        ("result_json", models.JSONField(default=dict)),
                        ("upload", models.FileField(blank=True, null=True, upload_to="uploads/%Y/%m/%d/")),
                        ("report_file", models.FileField(blank=True, null=True, upload_to="reports/%Y/%m/%d/")),
                        ("session_key", models.CharField(blank=True, db_index=True, max_length=64)),
                        ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                        (
                            "user",
                            models.ForeignKey(
                                blank=True,
                                null=True,
                                on_delete=django.db.models.deletion.SET_NULL,
                                related_name="analyses",
                                to=settings.AUTH_USER_MODEL,
                            ),
                        ),
                    ],
                    options={
                        "db_table": "analysis_analysisrecord",
                        "ordering": ["-created_at"],
                    },
                ),
            ],
            database_operations=[],
        ),
    ]
