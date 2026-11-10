"""Where do the image classifiers look? Quantitative Grad-CAM analysis on held-out test images.

The dataset images are aligned face crops, so the face occupies the central region. For every image
we compute the fraction of Grad-CAM mass that falls in the central 50% x 50% window (25% of the area)
and report it by class and by correct/incorrect prediction. A ratio near 1.0 means attention is spread
uniformly; > 1 means attention concentrates on the face; < 1 means it concentrates on the border/background.

    python scripts/gradcam_attention_analysis.py --per-class 250
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXTS = {".jpg", ".jpeg", ".png"}


def central_mass(cam: np.ndarray) -> float:
    """Fraction of total (upsampled) Grad-CAM mass inside the central 50% x 50% window."""
    big = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float64)
    total = big.sum()
    if total <= 0:
        return float("nan")
    return float(big[56:168, 56:168].sum() / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "deepfakeimage dataset/real_vs_fake/real-vs-fake/test"))
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--models", nargs="+", default=["resnet50:benchmark_out/resnet50.keras", "efficientnetb0:benchmark_out/efficientnetb0.keras"])
    ap.add_argument("--out", default=str(ROOT / "benchmark_out/gradcam_attention.json"))
    args = ap.parse_args()

    import tensorflow as tf
    from utils import gradcam

    rng = random.Random(42)
    files = {}
    for cls in ("real", "fake"):
        fs = sorted(p for p in (Path(args.data) / cls).iterdir() if p.suffix.lower() in EXTS)
        files[cls] = rng.sample(fs, args.per_class)

    result = {"per_class": args.per_class, "central_window_area_share": 0.25, "models": {}}
    for spec in args.models:
        name, path = spec.split(":")
        model = tf.keras.models.load_model(ROOT / path)
        rows = []
        for cls, label in (("real", 0), ("fake", 1)):
            for p in files[cls]:
                img = Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)
                cam, prob, _ = gradcam.gradcam_map(model, x)
                rows.append((cls, label, prob, central_mass(cam)))
            print(f"[{name}] {cls} done", flush=True)
        arr = np.array([(l, p, c) for _, l, p, c in rows], dtype=float)
        label, prob, cm = arr[:, 0], arr[:, 1], arr[:, 2]
        correct = (prob >= 0.5) == (label == 1)
        summ = lambda m: {"n": int(m.sum()), "central_mass_mean": round(float(np.nanmean(cm[m])), 4),
                          "central_mass_median": round(float(np.nanmedian(cm[m])), 4),
                          "ratio_vs_uniform": round(float(np.nanmean(cm[m])) / 0.25, 3)}
        result["models"][name] = {
            "all": summ(np.ones(len(cm), bool)),
            "real_images": summ(label == 0), "fake_images": summ(label == 1),
            "correct": summ(correct), "incorrect": summ(~correct),
            "accuracy_on_sample": round(float(correct.mean()), 4),
        }
        print(name, json.dumps(result["models"][name]), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
