"""Fine-tune BERT and RoBERTa on WELFake for the fake-news page."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Train BERT and/or RoBERTa on WELFake (saves under trained_models/). "
        "On CPU, defaults to a capped subset unless --full is set."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--architecture",
            choices=["bert", "roberta", "both"],
            default="both",
        )
        parser.add_argument("--epochs", type=int, default=2)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument(
            "--max-train",
            type=int,
            default=None,
            help="Cap training rows (recommended on CPU). Default: 12000 on CPU, full set on CUDA.",
        )
        parser.add_argument(
            "--full",
            action="store_true",
            help="Use full WELFake training split (slow on CPU, may take many hours).",
        )
        parser.add_argument("--smoke", action="store_true", help="Tiny run for pipeline check only.")

    def handle(self, *args, **options):
        root = Path(__file__).resolve().parents[4]
        csv_path = root / "datasets" / "text" / "WELFake_Dataset.csv"
        if not csv_path.is_file():
            self.stderr.write(self.style.ERROR(f"Missing dataset: {csv_path}"))
            return

        try:
            import torch
        except ImportError as exc:
            self.stderr.write(self.style.ERROR(f"PyTorch required: {exc}"))
            return

        script = root / "scripts" / "train_fake_news_transformers.py"
        cmd = [
            sys.executable,
            str(script),
            "--architecture",
            options["architecture"],
            "--epochs",
            str(options["epochs"]),
            "--batch-size",
            str(options["batch_size"]),
        ]

        if options["smoke"]:
            cmd.append("--smoke")
        else:
            max_train = options["max_train"]
            if options["full"]:
                pass
            elif max_train is not None:
                cmd.extend(["--max-train", str(max_train)])
            elif not torch.cuda.is_available():
                cap = 12000
                cmd.extend(["--max-train", str(cap)])
                self.stdout.write(
                    self.style.WARNING(
                        f"No GPU — training on {cap} WELFake articles (~1–3 h). "
                        "Use --full for all ~44k rows (very slow on CPU)."
                    )
                )

        self.stdout.write("Command: " + " ".join(cmd))
        self.stdout.write("Training started — keep this window open until 'Done.' appears.")
        subprocess.run(cmd, cwd=str(root), check=True)
        self.stdout.write(
            self.style.SUCCESS(
                "Finished. Restart Django, then choose BERT or RoBERTa on Fake News page."
            )
        )
