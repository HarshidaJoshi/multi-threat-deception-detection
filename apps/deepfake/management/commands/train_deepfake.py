from django.core.management.base import BaseCommand, CommandError

from utils.deepfake_classifier import ensure_dataset_ready, is_trained_model_available, model_path, train_classifier


class Command(BaseCommand):
    help = "Train the EfficientNet deepfake / AI-image classifier on datasets/real and datasets/fake"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            default="datasets",
            help="Path to dataset folder (default: datasets/)",
        )
        parser.add_argument("--epochs", type=int, default=8, help="Head-training epochs")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size (use 8 if you hit MemoryError)")
        parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Extra fine-tune epochs (use 0 to skip)")
        parser.add_argument("--resume-from", default=None, help="Saved .keras model to continue from (skips head training)")
        parser.add_argument("--max-per-class", type=int, default=None, help="Cap training images per class (random sample; useful on CPU)")
        parser.add_argument("--max-eval-per-class", type=int, default=None, help="Cap validation/test images per class")

    def handle(self, *args, **options):
        dataset_path = options["dataset"]
        try:
            stats = ensure_dataset_ready(dataset_path)
        except ValueError as exc:
            raise CommandError(
                f"{exc}\n\n"
                "Add your own images to datasets/real/ and datasets/fake/, or create demo samples:\n"
                "  python manage.py seed_demo_dataset --dataset datasets\n"
                "  python manage.py train_deepfake --dataset datasets"
            ) from exc

        self.stdout.write("Dataset summary:")
        for key, count in sorted(stats.items()):
            self.stdout.write(f"  {key}: {count}")

        if is_trained_model_available():
            self.stdout.write(self.style.WARNING("Overwriting existing trained model in trained_models/"))

        self.stdout.write("Training started (this may take several minutes)...")
        meta = train_classifier(
            dataset_path,
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            fine_tune_epochs=options["fine_tune_epochs"],
            resume_from=options["resume_from"],
            max_per_class=options["max_per_class"],
            max_eval_per_class=options["max_eval_per_class"],
        )

        self.stdout.write(self.style.SUCCESS("Training complete."))
        self.stdout.write(f"  Validation: {meta['validation']}")
        if meta.get("test"):
            self.stdout.write(f"  Held-out test: {meta['test']}")
        self.stdout.write(f"  Weights saved to: {model_path()}")
        self.stdout.write("Restart the server and test /deepfake/ again.")
