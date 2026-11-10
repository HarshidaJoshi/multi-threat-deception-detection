"""Transformer baselines (DistilBERT / BERT) for fake news and recruitment scams -- run on a GPU.

Uses exactly the same cleaning, splits and metrics as scripts/train_text_benchmark.py so the numbers are
directly comparable with the TF-IDF models. Repeats every experiment for several seeds.

Fake news   : fine-tune on ISOT / WELFake / LIAR train split; test in-domain and cross-dataset.
Recruitment : fine-tune on EMSCAD (class-weighted loss), threshold chosen on validation only.

    python scripts/gpu_text_transformers.py --root /kaggle/working/project --out /kaggle/working/out_text
    python scripts/gpu_text_transformers.py --root . --smoke        # tiny sanity run (minutes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # for train_text_benchmark (loaders, cleaning, split, metrics)
sys.path.insert(0, str(HERE.parent))  # for utils.text_ml

import train_text_benchmark as ttb  # noqa: E402


def set_seed(seed: int):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset:
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def encode(tok, texts, max_len):
    return tok(texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")


def predict_probs(model, tok, texts, max_len, batch, device):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def fine_tune(model_name, train_rows, val_rows, args, seed, class_weighted=False):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    texts = [t for t, _ in train_rows]
    y = torch.tensor([lab for _, lab in train_rows])
    enc = encode(tok, texts, args.max_len)
    dl = DataLoader(TextDataset(enc, y), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    if class_weighted:
        pos = float(y.sum())
        w = torch.tensor([1.0, (len(y) - pos) / max(pos, 1.0)], device=device)
    else:
        w = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best_state, best_loss = None, 1e9
    for ep in range(args.epochs):
        model.train()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
        pv = predict_probs(model, tok, [t for t, _ in val_rows], args.max_len, args.batch_size * 2, device)
        yv = np.array([lab for _, lab in val_rows])
        vloss = float(-np.mean(yv * np.log(pv + 1e-7) + (1 - yv) * np.log(1 - pv + 1e-7)))
        print(f"  epoch {ep + 1}/{args.epochs}  val_loss={vloss:.4f}", flush=True)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok, device


def prob_metrics(y, p, thr=0.5):
    """Same fields as train_text_benchmark.metrics, computed from probabilities."""
    return ttb.metrics(y, np.log(p + 1e-7) - np.log(1 - p + 1e-7), thr=float(np.log(thr + 1e-7) - np.log(1 - thr + 1e-7)))


def run_fake_news(args, out, seed):
    root = Path(args.root)
    corp = {"ISOT": ttb.load_isot(root), "WELFake": ttb.load_welfake(root)}
    splits = {k: ttb.stratified_split(v, seed=ttb.SEED) for k, v in corp.items()}
    liar = {s: ttb.load_liar(root, s) for s in ("train", "valid", "test")}
    tests = {"ISOT": splits["ISOT"][2], "WELFake": splits["WELFake"][2], "LIAR": liar["test"]}
    trains = {"ISOT": (splits["ISOT"][0], splits["ISOT"][1]), "WELFake": (splits["WELFake"][0], splits["WELFake"][1]),
              "LIAR": (liar["train"], liar["valid"])}
    res = {}
    for tr_name, (tr, va) in trains.items():
        if args.smoke:
            tr, va = tr[:200], va[:100]
        elif args.max_train and len(tr) > args.max_train:
            rng = np.random.RandomState(seed)
            tr = [tr[i] for i in rng.choice(len(tr), args.max_train, replace=False)]
        print(f"[fake news | {args.model} | seed {seed}] train on {tr_name} ({len(tr)})", flush=True)
        t0 = time.time()
        model, tok, device = fine_tune(args.model, tr, va, args, seed)
        row = {}
        for te_name, te in tests.items():
            if args.smoke:
                te = te[:100]
            p = predict_probs(model, tok, [t for t, _ in te], args.max_len, args.batch_size * 2, device)
            row[te_name] = prob_metrics(np.array([y for _, y in te]), p)
            print(f"   test {te_name}: acc={row[te_name]['accuracy']:.4f} f1={row[te_name]['f1']:.4f} auc={row[te_name]['roc_auc']:.4f}", flush=True)
        row["train_seconds"] = round(time.time() - t0, 1)
        res[f"train={tr_name}"] = row
        del model
    return res


def run_recruitment(args, out, seed):
    data = ttb.load_emscad(Path(args.root))
    y_all = np.array([y for _, _, y in data])
    rows = [(i, y) for i, y in enumerate(y_all)]
    tr, va, te = ttb.stratified_split(rows)  # same split as the TF-IDF experiments
    texts = [t for t, _, _ in data]
    mk = lambda part: [(texts[i], int(y_all[i])) for i, _ in part]
    tr_r, va_r, te_r = mk(tr), mk(va), mk(te)
    if args.smoke:
        tr_r, va_r, te_r = tr_r[:300], va_r[:150], te_r[:150]
    print(f"[recruitment | {args.model} | seed {seed}] train={len(tr_r)} val={len(va_r)} test={len(te_r)}", flush=True)
    t0 = time.time()
    model, tok, device = fine_tune(args.model, tr_r, va_r, args, seed, class_weighted=True)
    pv = predict_probs(model, tok, [t for t, _ in va_r], args.max_len, args.batch_size * 2, device)
    yv = np.array([y for _, y in va_r])
    from sklearn.metrics import precision_recall_curve
    p_, r_, t_ = precision_recall_curve(yv, pv)
    f_ = 2 * p_[:-1] * r_[:-1] / np.maximum(p_[:-1] + r_[:-1], 1e-9)
    thr = float(t_[int(np.argmax(f_))])  # chosen on validation only
    pt = predict_probs(model, tok, [t for t, _ in te_r], args.max_len, args.batch_size * 2, device)
    m = prob_metrics(np.array([y for _, y in te_r]), pt, thr=thr)
    m["val_threshold"] = round(thr, 4)
    m["train_seconds"] = round(time.time() - t0, 1)
    print(f"   test: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(HERE.parent), help="Folder containing datasets/text/*, fake-and-real-news-dataset/")
    ap.add_argument("--out", default="out_text")
    ap.add_argument("--model", default="distilbert-base-uncased", help="e.g. distilbert-base-uncased or bert-base-uncased")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=None, help="cap training texts per corpus (speed)")
    ap.add_argument("--tasks", nargs="+", default=["fake_news", "recruitment"], choices=["fake_news", "recruitment"])
    ap.add_argument("--smoke", action="store_true", help="tiny subsets to verify the pipeline")
    args = ap.parse_args()

    import torch
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_")
    results = {"config": vars(args), "fake_news": {}, "recruitment": {}}
    for seed in args.seeds:
        if "fake_news" in args.tasks:
            results["fake_news"][f"seed{seed}"] = run_fake_news(args, out, seed)
        if "recruitment" in args.tasks:
            results["recruitment"][f"seed{seed}"] = run_recruitment(args, out, seed)
        (out / f"results_text_{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")  # save after each seed
    print("All outputs in", out, flush=True)


if __name__ == "__main__":
    main()
