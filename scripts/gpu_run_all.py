"""One-command GPU run (Kaggle / Colab): full-data image benchmark x seeds, cross-domain tests,
transformer baselines x seeds, then aggregation into paper-ready tables.

Typical Kaggle notebook cell:

    !python scripts/gpu_run_all.py \
        --image-data /kaggle/input/<your-dataset>/real_vs_fake/real-vs-fake \
        --cifake-test /kaggle/input/cifake-real-and-ai-generated-synthetic-images/test \
        --root /kaggle/working/project --out /kaggle/working/out --mixed-precision

Run a quick sanity check first (a few minutes):   add  --smoke
Skip parts:  --skip-image  /  --skip-text  /  --skip-robust
Re-running the same command continues: finished seeds are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!! command exited with {r.returncode} (continuing)", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-data", help="folder with train/ valid/ test/ (each with real/ and fake/)")
    ap.add_argument("--cifake-test", default=None, help="CIFAKE test folder with REAL/ and FAKE/ (optional)")
    ap.add_argument("--root", default=str(HERE.parent), help="project root with datasets/text and fake-and-real-news-dataset")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--image-models", nargs="+", default=["efficientnetb0", "resnet50", "mobilenetv2", "simplecnn"])
    ap.add_argument("--text-models", nargs="+", default=["distilbert-base-uncased", "bert-base-uncased"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--ft-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--ablation-all-seeds", action="store_true", help="ablations for every seed (default: first seed only)")
    ap.add_argument("--skip-image", action="store_true")
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--skip-robust", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets / 1 epoch: verify everything works first")
    args = ap.parse_args()

    py, out = sys.executable, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_image:
        if not args.image_data:
            sys.exit("--image-data is required unless --skip-image")
        for i, seed in enumerate(args.seeds):
            d = out / f"image_seed{seed}"
            if (d / "results_table.md").is_file():
                print(f"seed {seed}: already finished, skipping", flush=True)
                continue
            cmd = [py, str(HERE / "train_benchmark_gpu.py"), "--data", args.image_data, "--out", str(d), "--seed", str(seed),
                   "--models", *args.image_models, "--epochs", str(args.epochs), "--ft-epochs", str(args.ft_epochs),
                   "--batch-size", str(args.batch_size)]
            if args.mixed_precision:
                cmd.append("--mixed-precision")
            if i > 0 and not args.ablation_all_seeds:
                cmd.append("--no-ablation")
            if args.smoke:
                cmd += ["--max-per-class", "64", "--max-eval-per-class", "64"]
                cmd[cmd.index("--epochs") + 1] = "1"
                cmd[cmd.index("--ft-epochs") + 1] = "1"
            run(cmd)

        if not args.skip_robust:
            # robustness / cross-domain test on the first seed's models
            d = out / f"image_seed{args.seeds[0]}"
            cmd = [py, str(HERE / "evaluate_robustness.py"), "--models-dir", str(d), "--faces-test", str(Path(args.image_data) / "test"),
                   "--models", *args.image_models]
            if args.cifake_test:
                cmd += ["--cifake-test", args.cifake_test]
            else:
                cmd += ["--cifake-test", str(out / "_no_cifake")]
            cmd += ["--local-images", str(out / "_no_local")]
            if args.smoke:
                cmd += ["--cifake-per-class", "32", "--faces-per-class", "32"]
            run(cmd)

    if not args.skip_text:
        for m in args.text_models:
            cmd = [py, str(HERE / "gpu_text_transformers.py"), "--root", args.root, "--out", str(out / "text"), "--model", m,
                   "--seeds", *[str(s) for s in args.seeds]]
            if args.smoke:
                cmd += ["--smoke", "--seeds", str(args.seeds[0])]
            run(cmd)

    run([py, str(HERE / "gpu_aggregate.py"), "--out", str(out)])
    print(f"\nDone. Download the folder: {out.resolve()}  (start with aggregated_tables.md)", flush=True)


if __name__ == "__main__":
    main()
