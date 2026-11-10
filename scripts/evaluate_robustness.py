"""Statistical and cross-domain evaluation of the trained image classifiers (run AFTER training).

1. Save test-set probabilities of every benchmark model (same 6,000 test images as the benchmark).
2. Bootstrap 95% confidence intervals for accuracy / F1 / AUC and McNemar tests between models.
3. Cross-domain tests with no retraining:
     - CIFAKE test split (real vs Stable-Diffusion images, 32x32 upscaled)  -> different generator and resolution
     - the project's own 984 local images (datasets/Deepfake image detection)
Outputs: benchmark_out/robustness.json and benchmark_out/robustness_tables.md

    python scripts/evaluate_robustness.py --cifake-per-class 1500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMG = (224, 224)
SEED = 42


def list_dir(d: Path, cap: int | None = None):
    fs = sorted(str(p) for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if cap and len(fs) > cap:
        fs = sorted(random.Random(SEED).sample(fs, cap))
    return fs


def predict_paths(model, paths, batch=64):
    import tensorflow as tf

    def load(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG)
        img.set_shape((*IMG, 3))
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(2)
    return model.predict(ds, verbose=0).reshape(-1)


def metrics(y, p):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = (p >= 0.5).astype(int)
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4), "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None, "n": int(len(y))}


def bootstrap_ci(y, p, n_boot=1000):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    rng = np.random.RandomState(SEED)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        yy, pp = y[i], p[i]
        accs.append(accuracy_score(yy, pp >= 0.5)); f1s.append(f1_score(yy, pp >= 0.5)); aucs.append(roc_auc_score(yy, pp))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"accuracy_ci95": q(accs), "f1_ci95": q(f1s), "auc_ci95": q(aucs)}


def mcnemar(y, p1, p2):
    """Exact two-sided McNemar test (binomial) on paired correctness."""
    from scipy.stats import binomtest
    c1, c2 = (p1 >= 0.5) == (y == 1), (p2 >= 0.5) == (y == 1)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    pval = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"only_first_correct": b, "only_second_correct": c, "p_value": pval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "benchmark_out"))
    ap.add_argument("--faces-test", default=str(ROOT / "deepfakeimage dataset/real_vs_fake/real-vs-fake/test"))
    ap.add_argument("--cifake-test", default=str(ROOT / "datasets/cifake/test"))
    ap.add_argument("--local-images", default=str(ROOT / "datasets/Deepfake image detection"))
    ap.add_argument("--cifake-per-class", type=int, default=1500)
    ap.add_argument("--faces-per-class", type=int, default=3000)
    ap.add_argument("--models", nargs="+", default=["resnet50_full_mlp", "resnet50", "efficientnetb0", "mobilenetv2", "simplecnn"])
    args = ap.parse_args()

    import tensorflow as tf
    out_dir = Path(args.models_dir)

    # ---- data sets -------------------------------------------------------------------------------
    sets = {}
    fp = list_dir(Path(args.faces_test) / "real", args.faces_per_class) + list_dir(Path(args.faces_test) / "fake", args.faces_per_class)
    sets["faces_test"] = (fp, np.array([0] * (len(fp) // 2) + [1] * (len(fp) // 2)))
    cdir = Path(args.cifake_test)
    cr, cf = list_dir(cdir / "REAL", args.cifake_per_class), list_dir(cdir / "FAKE", args.cifake_per_class)
    if cr and cf:
        sets["cifake_test"] = (cr + cf, np.array([0] * len(cr) + [1] * len(cf)))
    loc = Path(args.local_images)
    lr_, lf_ = [], []
    for d in loc.rglob("*"):
        if d.is_dir():
            n = d.name.lower()
            if n in ("real", "realimages", "real_images"):
                lr_ += list_dir(d)
            elif n in ("fake", "fakeimages", "fake_images", "sample_fake_images"):
                lf_ += list_dir(d)
    if lr_ and lf_:
        sets["local_own_images"] = (lr_ + lf_, np.array([0] * len(lr_) + [1] * len(lf_)))
    print({k: (len(v[0]), int(v[1].sum())) for k, v in sets.items()}, flush=True)

    results = {"datasets": {k: {"n": len(v[0]), "fake": int(v[1].sum())} for k, v in sets.items()}, "models": {}, "mcnemar_faces_test": {}}
    probs = {}
    for name in args.models:
        path = out_dir / f"{name}.keras"
        if not path.is_file():
            continue
        cache = out_dir / f"probs_{name}.npz"
        cached = dict(np.load(cache)) if cache.is_file() else {}
        model = tf.keras.models.load_model(path) if any(sname not in cached for sname in sets) else None
        results["models"][name] = {}
        for sname, (paths, y) in sets.items():
            p = cached[sname] if sname in cached and len(cached[sname]) == len(paths) else predict_paths(model, paths)
            probs[(name, sname)] = p
            m = metrics(y, p)
            if sname == "faces_test":
                m.update(bootstrap_ci(y, p))
            results["models"][name][sname] = m
            print(name, sname, m, flush=True)
            np.savez_compressed(cache, **{s: probs[(name, s)] for s in sets if (name, s) in probs})

    y = sets["faces_test"][1]
    names = [n for n in args.models if (n, "faces_test") in probs]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            results["mcnemar_faces_test"][f"{a} vs {b}"] = mcnemar(y, probs[(a, "faces_test")], probs[(b, "faces_test")])

    (out_dir / "robustness.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = ["## In-domain test with bootstrap 95% CI\n", "| Model | Accuracy [95% CI] | F1 [95% CI] | AUC [95% CI] |", "|---|---|---|---|"]
    for n, r in results["models"].items():
        f = r["faces_test"]
        md.append(f"| {n} | {f['accuracy']:.3f} {f['accuracy_ci95']} | {f['f1']:.3f} {f['f1_ci95']} | {f['auc']:.3f} {f['auc_ci95']} |")
    md += ["\n## Cross-domain accuracy / F1 / AUC (no retraining)\n", "| Model | " + " | ".join(k for k in sets if k != "faces_test") + " |",
           "|---|" + "---|" * (len(sets) - 1)]
    for n, r in results["models"].items():
        md.append(f"| {n} | " + " | ".join(f"{r[k]['accuracy']:.3f} / {r[k]['f1']:.3f} / {r[k]['auc']}" for k in sets if k != "faces_test") + " |")
    md += ["\n## McNemar tests (faces test set)\n", "| Pair | only first correct | only second correct | p |", "|---|---|---|---|"]
    for k, v in results["mcnemar_faces_test"].items():
        md.append(f"| {k} | {v['only_first_correct']} | {v['only_second_correct']} | {v['p_value']:.2e} |")
    (out_dir / "robustness_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("All outputs in", out_dir)


if __name__ == "__main__":
    main()
