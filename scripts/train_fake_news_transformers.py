"""Fine-tune BERT / RoBERTa on WELFake and save into trained_models/ for the web app.

Same cleaning and train/val/test split as scripts/train_text_benchmark.py (label 0=real, 1=fake).

Examples:
    python scripts/train_fake_news_transformers.py --architecture both
    python scripts/train_fake_news_transformers.py --architecture bert --epochs 2 --max-train 8000
    python scripts/train_fake_news_transformers.py --smoke

Requires: torch, transformers, datasets in WELFake at datasets/text/WELFake_Dataset.csv
GPU recommended for full training; use --max-train on CPU for a smaller run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import train_text_benchmark as ttb  # noqa: E402

ARCHITECTURES = {
    "bert": ("bert-base-uncased", "fake_news_bert_welfake"),
    "roberta": ("roberta-base", "fake_news_roberta_welfake"),
}


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _EncDataset:
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def _predict_fake_prob(model, tok, texts, max_len, batch, device):
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                texts[i : i + batch],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            use_cuda = device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(**enc).logits.float()
            chunks.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(chunks)


def train_one(
    base_model: str,
    save_name: str,
    train_rows,
    val_rows,
    test_rows,
    args,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    print(f"\n=== {save_name} | base={base_model} | device={device} | train={len(train_rows)} ===", flush=True)

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
    ).to(device)

    texts = [t for t, _ in train_rows]
    y = torch.tensor([lab for _, lab in train_rows], dtype=torch.long)
    enc = tok(texts, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
    dl = DataLoader(_EncDataset(enc, y), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(len(dl) * args.epochs, 1)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    pos = float(y.sum())
    w = torch.tensor([1.0, (len(y) - pos) / max(pos, 1.0)], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best_state, best_loss = None, 1e9
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            use_cuda = device == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()

        val_texts = [t for t, _ in val_rows]
        pv = _predict_fake_prob(model, tok, val_texts, args.max_len, args.batch_size * 2, device)
        yv = np.array([y for _, y in val_rows])
        vloss = float(-np.mean(yv * np.log(pv + 1e-7) + (1 - yv) * np.log(1 - pv + 1e-7)))
        print(f"  epoch {ep + 1}/{args.epochs}  val_loss={vloss:.4f}", flush=True)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    save_dir = Path(args.models_dir) / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    te_texts = [t for t, _ in test_rows]
    pt = _predict_fake_prob(model, tok, te_texts, args.max_len, args.batch_size * 2, device)
    yt = np.array([y for _, y in test_rows])
    test_metrics = ttb.metrics(yt, np.log(pt + 1e-7) - np.log(1 - pt + 1e-7))

    meta = {
        "architecture": save_name,
        "base_model": base_model,
        "train_corpus": "WELFake",
        "label_0": "real",
        "label_1": "fake",
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "train_seconds": round(time.time() - t0, 1),
        "epochs": args.epochs,
        "max_len": args.max_len,
    }
    (save_dir.parent / f"{save_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"  saved {save_dir} | test acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}",
        flush=True,
    )
    return meta


def _preflight(root: Path, welfake_csv: Path | None) -> None:
    missing: list[str] = []
    if not (root / "utils" / "text_ml.py").is_file():
        missing.append(f"{root / 'utils' / 'text_ml.py'}  (zip the utils/ folder from your project)")
    try:
        ttb.welfake_csv_path(root, welfake_csv)
    except FileNotFoundError as exc:
        missing.append(str(exc))
    if missing:
        print("Colab / path setup incomplete:\n", file=sys.stderr)
        for m in missing:
            print(f"  - {m}\n", file=sys.stderr)
        print("See docs/COLAB_FAKE_NEWS_TRAINING.md", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--architecture", choices=["bert", "roberta", "both"], default="both")
    ap.add_argument(
        "--welfake-csv",
        type=Path,
        default=None,
        help="Path to WELFake_Dataset.csv if not at datasets/text/ under project root",
    )
    ap.add_argument("--models-dir", default=str(ROOT / "trained_models"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=None, help="Cap training rows (use on CPU)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Tiny subset for pipeline check")
    args = ap.parse_args()

    _preflight(ROOT, args.welfake_csv)
    rows = ttb.load_welfake(ROOT, csv_file=args.welfake_csv)
    train, val, test = ttb.stratified_split(rows, seed=ttb.SEED)
    if args.smoke:
        train, val, test = train[:300], val[:100], test[:200]
    elif args.max_train and len(train) > args.max_train:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(train), args.max_train, replace=False)
        train = [train[i] for i in idx]

    to_run = ["bert", "roberta"] if args.architecture == "both" else [args.architecture]
    summary = {}
    for key in to_run:
        base, save_name = ARCHITECTURES[key]
        summary[key] = train_one(base, save_name, train, val, test, args)

    out = Path(args.models_dir) / "fake_news_transformers_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Restart Django and select BERT/RoBERTa — app loads from {args.models_dir} when present.", flush=True)


if __name__ == "__main__":
    main()
