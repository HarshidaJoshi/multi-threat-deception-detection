from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Seed demo dataset (if needed) and train the deepfake classifier"

    def add_arguments(self, parser):
        parser.add_argument("--dataset", default="datasets", help="Dataset folder (default: datasets/)")
        parser.add_argument("--count", type=int, default=60, help="Demo images per class (default: 60)")
        parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
        parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
        parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Fine-tune epochs")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Regenerate demo images before training",
        )

    def handle(self, *args, **options):
        self.stdout.write("Step 1/2: Preparing demo dataset...")
        call_command(
            "seed_demo_dataset",
            dataset=options["dataset"],
            count=options["count"],
            overwrite=options["overwrite"],
        )
        self.stdout.write("")
        self.stdout.write("Step 2/2: Training classifier...")
        call_command(
            "train_deepfake",
            dataset=options["dataset"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            fine_tune_epochs=options["fine_tune_epochs"],
        )
