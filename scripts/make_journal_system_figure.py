"""Generate the journal manuscript's system architecture figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "journal_manuscript" / "figs" / "system_architecture.png"


def box(ax, x, y, w, h, title, detail, facecolor):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#263238",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + h * 0.31, detail, ha="center", va="center", fontsize=8)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#455a64"))


def main():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.03, 0.67, 0.25, 0.19, "Synthetic-image module", "ResNet50 / CNN\nGrad-CAM", "#e3f2fd")
    box(ax, 0.375, 0.67, 0.25, 0.19, "Fake-news module", "TF-IDF + linear model\nExact term contributions", "#e8f5e9")
    box(ax, 0.72, 0.67, 0.25, 0.19, "Recruitment module", "TF-IDF + linear model\nExact term contributions", "#fff3e0")

    box(ax, 0.32, 0.36, 0.36, 0.17, "Orchestration and audit record", "Model, backend, score, reliability,\nand explanation artefact", "#f3e5f5")
    box(ax, 0.32, 0.08, 0.36, 0.15, "Human review", "Decision support, not automatic verification", "#eceff1")

    for x in (0.155, 0.5, 0.845):
        arrow(ax, (x, 0.67), (0.5, 0.53))
    arrow(ax, (0.5, 0.36), (0.5, 0.23))

    ax.text(
        0.5,
        0.95,
        "XAI-Verify: independent sensing modules with a shared human-audit layer",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {DEST}")


if __name__ == "__main__":
    main()
