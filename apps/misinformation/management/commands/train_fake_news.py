"""Train deployed fake-news TF-IDF model (WELFake) into trained_models/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from django.core.management.base import BaseCommand

from utils.model_registry import fake_news_tfidf_available


class Command(BaseCommand):
    help = (
        "Train fake_news_welfake_logreg.joblib (TF-IDF + LogReg on WELFake). "
        "Requires datasets/text/WELFake_Dataset.csv."
    )

    def handle(self, *args, **options):
        root = Path(__file__).resolve().parents[4]
        csv_path = root / "datasets" / "text" / "WELFake_Dataset.csv"
        if not csv_path.is_file():
            self.stderr.write(
                self.style.ERROR(f"Missing dataset: {csv_path}\nDownload WELFake and place CSV there.")
            )
            return

        script = root / "scripts" / "train_text_benchmark.py"
        cmd = [sys.executable, str(script), "--task", "fake_news"]
        self.stdout.write("Training fake news models (may take several minutes)...")
        subprocess.run(cmd, cwd=str(root), check=True)

        if fake_news_tfidf_available():
            self.stdout.write(self.style.SUCCESS("Saved trained_models/fake_news_welfake_logreg.joblib"))
        else:
            self.stderr.write(
                self.style.WARNING("Training finished but joblib not found — check script output.")
            )
