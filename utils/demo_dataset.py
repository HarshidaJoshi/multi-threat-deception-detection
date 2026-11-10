"""Generate a small synthetic demo dataset for pipeline testing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _real_like_image(seed: int, size: tuple[int, int] = (256, 256)) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(*size, 3), dtype=np.uint8)
    noise = rng.integers(0, 35, size=(*size, 3), dtype=np.uint8)
    arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(rng.integers(2, 6)):
        x0, y0 = rng.integers(0, size[0] - 40, size=2)
        draw.ellipse((x0, y0, x0 + 40, y0 + 40), fill=tuple(rng.integers(0, 255, size=3).tolist()))
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def _fake_like_image(seed: int, size: tuple[int, int] = (256, 256)) -> Image.Image:
    rng = np.random.default_rng(seed)
    palette = [
        tuple(rng.integers(0, 255, size=3).tolist()),
        tuple(rng.integers(0, 255, size=3).tolist()),
        tuple(rng.integers(0, 255, size=3).tolist()),
    ]
    img = Image.new("RGB", size, palette[0])
    draw = ImageDraw.Draw(img)
    step = rng.integers(18, 32)
    for y in range(0, size[1], step):
        for x in range(0, size[0], step):
            color = palette[(x // step + y // step) % len(palette)]
            draw.rectangle((x, y, x + step - 2, y + step - 2), fill=color)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for _ in range(rng.integers(4, 8)):
        odraw.line(
            (rng.integers(0, size[0]), rng.integers(0, size[1]), rng.integers(0, size[0]), rng.integers(0, size[1])),
            fill=(*rng.integers(120, 255, size=3).tolist(), 180),
            width=2,
        )
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _count_class_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def seed_demo_dataset(
    dataset_root: str | Path,
    *,
    count_per_class: int = 60,
    overwrite: bool = False,
) -> dict[str, int | bool]:
    """Write synthetic demo images into datasets/real and datasets/fake."""
    if count_per_class < 2:
        raise ValueError("count_per_class must be at least 2")

    root = Path(dataset_root)
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing_real = _count_class_images(real_dir)
    existing_fake = _count_class_images(fake_dir)

    if not overwrite and existing_real >= count_per_class and existing_fake >= count_per_class:
        return {
            "real": existing_real,
            "fake": existing_fake,
            "created": False,
        }

    if (existing_real or existing_fake) and not overwrite:
        raise FileExistsError(
            f"Dataset already has images (real={existing_real}, fake={existing_fake}) "
            f"but fewer than {count_per_class} per class. Use --overwrite to replace demo samples."
        )

    if overwrite:
        for folder in (real_dir, fake_dir):
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    path.unlink()

    random.seed(42)
    for i in range(count_per_class):
        _real_like_image(1000 + i).save(real_dir / f"demo_real_{i:03d}.jpg", quality=90)
        _fake_like_image(2000 + i).save(fake_dir / f"demo_fake_{i:03d}.jpg", quality=90)

    return {"real": count_per_class, "fake": count_per_class, "created": True}
