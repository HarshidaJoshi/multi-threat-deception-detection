"""Grad-CAM example grid for the paper: randomly chosen (fixed seed, NOT hand-picked) test images.

    python scripts/make_gradcam_figure.py
"""

from __future__ import annotations

import base64
import io
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    from utils import gradcam

    model = tf.keras.models.load_model(ROOT / "benchmark_out" / "resnet50_full_mlp.keras")
    test = ROOT / "deepfakeimage dataset/real_vs_fake/real-vs-fake/test"
    rng = random.Random(0)
    picks = []
    for cls in ("real", "fake"):
        fs = sorted(p for p in (test / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        picks += [(cls, p) for p in rng.sample(fs, 4)]

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
    for i, (cls, p) in enumerate(picks):
        out = gradcam.gradcam_overlay_base64(model, str(p))
        overlay = Image.open(io.BytesIO(base64.b64decode(out["png_base64"]))).convert("RGB")
        orig = Image.open(p).convert("RGB").resize(overlay.size)
        axes[0, i].imshow(orig); axes[1, i].imshow(overlay)
        axes[0, i].set_title(f"true: {cls}", fontsize=9)
        axes[1, i].set_title(f"P(fake) = {out['fake_probability']:.2f}", fontsize=9)
        for r in (0, 1):
            axes[r, i].axis("off")
    fig.tight_layout()
    dest = ROOT / "docs" / "journal_manuscript" / "figs" / "gradcam_examples.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", dest)


if __name__ == "__main__":
    main()
