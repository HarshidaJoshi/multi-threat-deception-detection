"""Aggregate repeated runs (seeds) into mean +/- std tables for the paper.

Reads   <out>/image_seed*/results.json           (scripts/train_benchmark_gpu.py)
        <out>/text/results_text_*.json           (scripts/gpu_text_transformers.py)
Writes  <out>/aggregated_tables.md and <out>/aggregated.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ms(vals, pct=True):
    a = np.array(vals, dtype=float)
    if len(a) == 0:
        return "-"
    f = 100 if pct else 1
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return f"{a.mean() * f:.2f} ± {sd * f:.2f}" if pct else f"{a.mean():.3f} ± {sd:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = Path(args.out)
    md, agg = [], {"image": {}, "text": {}}

    # ---- images -----------------------------------------------------------------------------------
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("image_seed*/results.json"))]
    if runs:
        n_seeds = len(runs)
        md += [f"## Image classifiers (mean ± std over {n_seeds} seed(s); test accuracy / F1 in %, AUC)\n",
               "| Model | Accuracy | F1 | AUC |", "|---|---|---|---|"]
        for bucket in ("models", "ablation"):
            names = []
            for r in runs:
                for n in r.get(bucket, {}):
                    if n not in names:
                        names.append(n)
            for n in names:
                rows = [r[bucket][n]["test"] for r in runs if n in r.get(bucket, {})]
                agg["image"][n] = {k: [x[k] for x in rows] for k in ("accuracy", "f1", "auc")}
                tag = n if bucket == "models" else f"{n} [ablation, n={len(rows)}]"
                md.append(f"| {tag} | {ms([x['accuracy'] for x in rows])} | {ms([x['f1'] for x in rows])} | {ms([x['auc'] for x in rows], pct=False)} |")
        md.append("")

    # ---- text transformers ----------------------------------------------------------------------------
    for f in sorted((out / "text").glob("results_text_*.json")) if (out / "text").is_dir() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        model = r["config"]["model"]
        fn = r.get("fake_news", {})
        if fn:
            seeds = list(fn)
            trains = sorted({k for s in seeds for k in fn[s]})
            tests = ["ISOT", "WELFake", "LIAR"]
            md += [f"## Fake news, {model} (mean ± std over {len(seeds)} seed(s); accuracy % / F1 %)\n",
                   "| Train \\ Test | " + " | ".join(tests) + " |", "|---|" + "---|" * len(tests)]
            for tr in trains:
                cells = []
                for te in tests:
                    acc = [fn[s][tr][te]["accuracy"] for s in seeds if tr in fn[s]]
                    f1 = [fn[s][tr][te]["f1"] for s in seeds if tr in fn[s]]
                    cells.append(f"{ms(acc)} / {ms(f1)}")
                md.append(f"| {tr.replace('train=', '')} | " + " | ".join(cells) + " |")
                agg["text"][f"{model}|fake_news|{tr}"] = {te: [fn[s][tr][te]["accuracy"] for s in seeds if tr in fn[s]] for te in tests}
            md.append("")
        rc = r.get("recruitment", {})
        if rc:
            seeds = list(rc)
            md += [f"## Recruitment scams, {model} (mean ± std over {len(seeds)} seed(s))\n", "| Precision | Recall | F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|---|"]
            md.append("| " + " | ".join(ms([rc[s][k] for s in seeds], pct=False) for k in ("precision", "recall", "f1", "pr_auc", "roc_auc")) + " |\n")
            agg["text"][f"{model}|recruitment"] = {k: [rc[s][k] for s in seeds] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc")}

    # ---- robustness (cross-domain + McNemar) from the first seed ------------------------------------
    for p in sorted(out.glob("image_seed*/robustness_tables.md"))[:1]:
        md += [f"## Robustness (from {p.parent.name})\n", p.read_text(encoding="utf-8")]

    (out / "aggregated_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "aggregated.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
