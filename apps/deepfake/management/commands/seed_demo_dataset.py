from django.core.management.base import BaseCommand, CommandError

from utils.demo_dataset import seed_demo_dataset


class Command(BaseCommand):
    help = "Create synthetic demo images in datasets/real and datasets/fake for training tests"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            default="datasets",
            help="Dataset root folder (default: datasets/)",
        )
        parser.add_argument(
            "--count",
            type=int,
            default=60,
            help="Images per class (default: 60)",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace existing images in real/ and fake/",
        )

    def handle(self, *args, **options):
        try:
            stats = seed_demo_dataset(
                options["dataset"],
                count_per_class=options["count"],
                overwrite=options["overwrite"],
            )
        except FileExistsError as exc:
            raise CommandError(str(exc)) from exc
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("Demo dataset ready." if not stats.get("created") else "Demo dataset created."))
        for key in ("real", "fake"):
            if key in stats:
                self.stdout.write(f"  {key}: {stats[key]}")
        if not stats.get("created"):
            self.stdout.write(self.style.WARNING("  (existing images kept — use --overwrite to regenerate)"))
        self.stdout.write("")
        self.stdout.write("Next step:")
        self.stdout.write("  python manage.py train_demo_deepfake --dataset datasets")
        self.stdout.write("  (or: python manage.py train_deepfake --dataset datasets)")
