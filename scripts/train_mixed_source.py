"""Does training on several image sources fix the cross-domain failure?  (CPU, resumable)

Features: frozen ImageNet ResNet50 (same extractor as the full-data run); faces features are reused from
benchmark_out/full_frozen/features. New features are extracted for a CIFAKE training sample, a CIFAKE test
sample, and the 999 local images, in resumable chunks.

Training conditions (MLP head, 3 seeds each):
    faces only | CIFAKE only | faces + CIFAKE (all faces) | faces + CIFAKE (faces subsampled to CIFAKE size)
Test domains: faces test (20,000) | CIFAKE test sample | local images (999, NEVER used for training)

Outputs: benchmark_out/mixed/results_mixed.json and results_mixed.md.   Re-run to resume.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import train_full_frozen_features as tff  # noqa: E402  (extract / load_features / metrics)

OUT = ROOT / "benchmark_out" / "mixed"
FACES = ROOT / "benchmark_out" / "full_frozen" / "features"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_dir(d: Path):
    return sorted(str(p) for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def sample(paths, k, seed=42):
    return sorted(random.Random(seed).sample(paths, k)) if len(paths) > k else paths


def cifake_set(split_dir: Path, per_class: int):
    r, f = sample(list_dir(split_dir / "REAL"), per_class), sample(list_dir(split_dir / "FAKE"), per_class)
    return r + f, np.array([0] * len(r) + [1] * len(f), dtype=np.int8)


def local_set():
    loc = ROOT / "datasets" / "Deepfake image detection"
    real, fake = [], []
    for d in loc.rglob("*"):
        if d.is_dir():
            n = d.name.lower()
            if n in ("real", "realimages", "real_images"):
                real += list_dir(d)
            elif n in ("fake", "fakeimages", "fake_images", "sample_fake_images"):
                fake += list_dir(d)
    return real + fake, np.array([0] * len(real) + [1] * len(fake), dtype=np.int8)


def faces_features():
    n_chunks = {"train": 20, "valid": 2, "test": 4}
    X = {s: np.concatenate([np.load(FACES / f"{s}_{i:03d}.npy") for i in range(n)]).astype(np.float32) for s, n in n_chunks.items()}
    # labels: sorted real then fake (see train_full_frozen_features.list_split)
    y = {"train": np.array([0] * 50000 + [1] * 50000), "valid": np.array([0] * 5000 + [1] * 5000), "test": np.array([0] * 10000 + [1] * 10000)}
    return X, y


def train_head(Xtr, ytr, Xva, yva, seed):
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = tf.keras.Sequential([layers.Input(shape=(2048,)), layers.Dense(512, activation="relu"), layers.Dropout(0.4),
                             layers.Dense(128, activation="relu"), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    m.fit((Xtr - mu) / sd, ytr, validation_data=((Xva - mu) / sd, yva), epochs=40, batch_size=512, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])
    return lambda X: m.predict((X - mu) / sd, verbose=0).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifake", default=str(ROOT / "datasets" / "cifake"))
    ap.add_argument("--cifake-train-per-class", type=int, default=5000)
    ap.add_argument("--cifake-test-per-class", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=2500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf

    (OUT / "features").mkdir(parents=True, exist_ok=True)
    base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    cdir = Path(args.cifake)
    sets = {
        "cifake_train": cifake_set(cdir / "train", args.cifake_train_per_class),
        "cifake_test": cifake_set(cdir / "test", args.cifake_test_per_class),
        "local": local_set(),
    }
    print({k: (len(v[0]), int(v[1].sum())) for k, v in sets.items()}, flush=True)
    for name, (paths, _) in sets.items():
        tff.extract(name, paths, OUT / "features", args.chunk, 64, base)
    if args.extract_only:
        return

    F = {name: tff.load_features(name, OUT / "features", len(paths), args.chunk) for name, (paths, _) in sets.items()}
    Y = {name: y.astype(int) for name, (_, y) in sets.items()}
    FX, FY = faces_features()

    # CIFAKE: hold out 10% of its training sample as validation for the CIFAKE-containing conditions
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(Y["cifake_train"]))
    nv = len(perm) // 10
    c_va, c_tr = perm[:nv], perm[nv:]
    cX_tr, cY_tr, cX_va, cY_va = F["cifake_train"][c_tr], Y["cifake_train"][c_tr], F["cifake_train"][c_va], Y["cifake_train"][c_va]

    sub = rng.choice(len(FY["train"]), size=len(cY_tr), replace=False)  # faces subsample, class balance preserved on average
    conds = {
        "faces only": (FX["train"], FY["train"], FX["valid"], FY["valid"]),
        "CIFAKE only": (cX_tr, cY_tr, cX_va, cY_va),
        "faces + CIFAKE (all faces)": (np.concatenate([FX["train"], cX_tr]), np.concatenate([FY["train"], cY_tr]),
                                       np.concatenate([FX["valid"], cX_va]), np.concatenate([FY["valid"], cY_va])),
        "faces + CIFAKE (balanced sizes)": (np.concatenate([FX["train"][sub], cX_tr]), np.concatenate([FY["train"][sub], cY_tr]),
                                            np.concatenate([FX["valid"], cX_va]), np.concatenate([FY["valid"], cY_va])),
    }
    tests = {"faces test": (FX["test"], FY["test"]), "CIFAKE test": (F["cifake_test"], Y["cifake_test"]), "local images (unseen)": (F["local"], Y["local"])}

    results = {"sizes": {c: int(len(v[1])) for c, v in conds.items()}, "tests": {t: int(len(v[1])) for t, v in tests.items()}, "runs": {}}
    prior = OUT / "results_mixed.json"
    if prior.is_file():  # resume: keep conditions that already finished all seeds
        old = json.loads(prior.read_text(encoding="utf-8"))
        results["runs"] = {c: r for c, r in old.get("runs", {}).items() if all(len(v) == len(args.seeds) for v in r.values())}
        print("resuming; already finished:", list(results["runs"]), flush=True)
    for cname, (Xtr, ytr, Xva, yva) in conds.items():
        if cname in results["runs"]:
            continue
        results["runs"][cname] = {t: [] for t in tests}
        for seed in args.seeds:
            predict = train_head(Xtr, ytr, Xva, yva, seed)
            for tname, (Xte, yte) in tests.items():
                results["runs"][cname][tname].append(tff.metrics(yte, predict(Xte)))
            print(f"[{cname} | seed {seed}] " + "  ".join(f"{t}: acc={results['runs'][cname][t][-1]['accuracy']:.3f}" for t in tests), flush=True)
        (OUT / "results_mixed.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    ms = lambda L, k: f"{np.mean([r[k] for r in L]):.3f} ± {np.std([r[k] for r in L], ddof=1):.3f}"
    md = [f"## Training-source ablation (frozen ResNet50 + MLP head; mean ± std over {len(args.seeds)} seeds; accuracy / AUC)\n",
          "| Training data (n) | " + " | ".join(f"{t} (n={results['tests'][t]})" for t in tests) + " |", "|---|" + "---|" * len(tests)]
    for cname in conds:
        md.append(f"| {cname} ({results['sizes'][cname]}) | " + " | ".join(f"{ms(results['runs'][cname][t], 'accuracy')} / {ms(results['runs'][cname][t], 'auc')}" for t in tests) + " |")
    (OUT / "results_mixed.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("All outputs in", OUT, flush=True)


if __name__ == "__main__":
    main()
