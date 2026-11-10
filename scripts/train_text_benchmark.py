"""Text benchmarks for fake news and recruitment-scam detection (scikit-learn, CPU friendly).

Fake news   : train on cleaned ISOT or cleaned WELFake, test in-domain AND on the other corpus and
              on LIAR (cross-dataset generalisation). Source tags that leak the label are removed.
Recruitment : EMSCAD with stratified train/val/test, class-imbalance handling, decision threshold
              tuned on validation only, compared with the project's existing rule-based baseline.

Outputs (in --out):  results.json, results_tables.md, top_terms.json, and trained models in --models-dir.

    python scripts/train_text_benchmark.py --task all
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 42

# ----------------------------------------------------------------------------- loading / cleaning

from utils.text_ml import clean_news  # noqa: E402  (same cleaning at train and inference time)


def _dedupe(rows: list[tuple[str, int]], min_words: int = 15) -> list[tuple[str, int]]:
    seen, out = set(), []
    for text, y in rows:
        if len(text.split()) < min_words or text in seen:
            continue
        seen.add(text)
        out.append((text, y))
    return out


def load_isot(root: Path):
    rows = []
    for fn, y in (("Fake.csv", 1), ("True.csv", 0)):
        with open(root / "fake-and-real-news-dataset" / fn, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append((clean_news(f"{r['title']}. {r['text']}"), y))
    return _dedupe(rows)


def welfake_csv_path(root: Path, csv_file: Path | None = None) -> Path:
    path = csv_file if csv_file is not None else root / "datasets" / "text" / "WELFake_Dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"WELFake dataset not found: {path}\n"
            "On Colab: upload WELFake_Dataset.csv from your PC (datasets/text/) into\n"
            "  /content/datasets/text/WELFake_Dataset.csv\n"
            "Or run training with: --welfake-csv /content/WELFake_Dataset.csv\n"
            "You also need the utils/ folder (utils/text_ml.py) next to scripts/."
        )
    return path


def load_welfake(root: Path, csv_file: Path | None = None):
    """WELFake labels: in the released CSV, 1 = fake (37,106 rows) and 0 = real (35,028 rows)."""
    path = welfake_csv_path(root, csv_file)
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((clean_news(f"{r['title']}. {r['text']}"), 1 if r["label"].strip() == "1" else 0))
    return _dedupe(rows)


_LIAR_FAKE = {"false", "barely-true", "pants-fire"}


def load_liar(root: Path, split: str):
    rows = []
    with open(root / "datasets" / "text" / "liar" / f"{split}.tsv", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 2:
                rows.append((p[2].strip(), 1 if p[1] in _LIAR_FAKE else 0))
    return rows


def stratified_split(rows, fracs=(0.7, 0.15, 0.15), seed=SEED):
    rng = np.random.RandomState(seed)
    idx = {0: [], 1: []}
    for i, (_, y) in enumerate(rows):
        idx[y].append(i)
    parts = ([], [], [])
    for y, ids in idx.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        a, b = int(len(ids) * fracs[0]), int(len(ids) * (fracs[0] + fracs[1]))
        parts[0].extend(ids[:a]); parts[1].extend(ids[a:b]); parts[2].extend(ids[b:])
    return [[rows[i] for i in p] for p in parts]


# ----------------------------------------------------------------------------- metrics / models

def metrics(y, score, thr=None, name_thr="default"):
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y = np.asarray(y)
    thr = 0.0 if thr is None else thr
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y)),
    }


def make_models(balanced: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    cw = "balanced" if balanced else None
    return {
        "tfidf+logreg": LogisticRegression(C=4.0, max_iter=2000, class_weight=cw, solver="liblinear"),
        "tfidf+linearsvm": LinearSVC(C=0.5, class_weight=cw),
    }


def scores_of(model, X):
    return model.decision_function(X)


def top_terms(vec, model, k=25):
    names = np.array(vec.get_feature_names_out())
    coef = model.coef_.ravel()
    order = np.argsort(coef)
    return {"toward_class_0": names[order[:k]].tolist(), "toward_class_1": names[order[-k:][::-1]].tolist()}


# ----------------------------------------------------------------------------- fake news

def run_fake_news(args, out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    print("Loading datasets ...", flush=True)
    corp = {"ISOT": load_isot(ROOT), "WELFake": load_welfake(ROOT)}
    splits = {k: stratified_split(v) for k, v in corp.items()}
    liar_tr, liar_va, liar_te = load_liar(ROOT, "train"), load_liar(ROOT, "valid"), load_liar(ROOT, "test")
    stats = {k: {"after_cleaning": len(v), "fake": sum(y for _, y in v)} for k, v in corp.items()}
    stats["LIAR"] = {"train": len(liar_tr), "valid": len(liar_va), "test": len(liar_te),
                     "fake_train": sum(y for _, y in liar_tr)}
    print("Corpus sizes after cleaning/dedup:", json.dumps(stats), flush=True)

    tests = {"ISOT": splits["ISOT"][2], "WELFake": splits["WELFake"][2], "LIAR": liar_te}
    train_sets = {"ISOT": splits["ISOT"][0], "WELFake": splits["WELFake"][0], "LIAR": liar_tr}
    result = {"corpus_stats": stats, "runs": {}}
    tops = {}
    for tr_name, tr_rows in train_sets.items():
        print(f"\n=== fake news: train on {tr_name} ({len(tr_rows)}) ===", flush=True)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True, lowercase=True,
                              stop_words="english")
        Xtr = vec.fit_transform([t for t, _ in tr_rows])
        ytr = np.array([y for _, y in tr_rows])
        for mname, model in make_models(balanced=True).items():
            t0 = time.time()
            model.fit(Xtr, ytr)
            res = {}
            for te_name, te_rows in tests.items():
                Xte = vec.transform([t for t, _ in te_rows])
                res[te_name] = metrics([y for _, y in te_rows], scores_of(model, Xte))
            res["train_seconds"] = round(time.time() - t0, 1)
            result["runs"][f"{mname} | train={tr_name}"] = res
            print(f"[{mname} | {tr_name}] " + "  ".join(f"{k}: acc={v['accuracy']:.3f} f1={v['f1']:.3f} auc={v['roc_auc']:.3f}"
                                                          for k, v in res.items() if isinstance(v, dict)), flush=True)
            if mname == "tfidf+logreg":
                tops[f"fake_news | train={tr_name}"] = top_terms(vec, model)
                joblib.dump({"vectorizer": vec, "model": model, "label_1": "fake"},
                            Path(args.models_dir) / f"fake_news_{tr_name.lower()}_logreg.joblib")
    (out / "results_fake_news.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, tops


# ----------------------------------------------------------------------------- recruitment

_TEXT_FIELDS = ("title", "company_profile", "description", "requirements", "benefits")


def load_emscad(root: Path):
    rows = []
    with open(root / "datasets" / "text" / "emscad" / "fake_job_postings.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            text = re.sub(r"\s+", " ", " . ".join(r[f] for f in _TEXT_FIELDS if r[f])).strip()
            meta = [int(r["telecommuting"] or 0), int(r["has_company_logo"] or 0), int(r["has_questions"] or 0),
                    int(not r["salary_range"].strip()), int(not r["company_profile"].strip()),
                    int(not r["requirements"].strip()), int(not r["employment_type"].strip()),
                    int(not r["required_experience"].strip())]
            rows.append((text, meta, int(r["fraudulent"])))
    seen, out = set(), []
    for t, m, y in rows:  # drop exact duplicate postings so they cannot straddle train/test
        if t in seen or len(t.split()) < 5:
            continue
        seen.add(t)
        out.append((t, m, y))
    return out


def best_threshold(y, s):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
    return float(t[int(np.argmax(f))])


def run_recruitment(args, out):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    import joblib
    from utils import ml_utils

    data = load_emscad(ROOT)
    y_all = np.array([y for _, _, y in data])
    print(f"\n=== recruitment: EMSCAD {len(data)} postings after dedup, fraud={y_all.sum()} ({y_all.mean():.1%}) ===", flush=True)
    idx = np.arange(len(data))
    rows = [(i, y) for i, y in zip(idx, y_all)]
    tr, va, te = stratified_split(rows)
    tr_i, va_i, te_i = [np.array([i for i, _ in part]) for part in (tr, va, te)]

    texts = [t for t, _, _ in data]
    meta = csr_matrix(np.array([m for _, m, _ in data], dtype=float))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True, stop_words="english")
    vec.fit([texts[i] for i in tr_i])
    X = hstack([vec.transform(texts), meta]).tocsr()

    result = {"n": len(data), "fraud": int(y_all.sum()), "split": {"train": len(tr_i), "val": len(va_i), "test": len(te_i)}, "models": {}}

    # Baseline: the project's existing hand-written rules (text signals only).
    rule_score = np.array([ml_utils.recruitment_job_forensics(t)["risk_delta"] for t in texts])
    result["models"]["rule-based (existing)"] = {"test": metrics(y_all[te_i], rule_score[te_i], thr=1e-9)}

    models = make_models(balanced=True)
    models["random_forest"] = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1,
                                                     random_state=SEED, min_samples_leaf=2)
    tops = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X[tr_i], y_all[tr_i])
        s_va = model.predict_proba(X[va_i])[:, 1] if name == "random_forest" else model.decision_function(X[va_i])
        thr = best_threshold(y_all[va_i], s_va)  # chosen on validation only
        s_te = model.predict_proba(X[te_i])[:, 1] if name == "random_forest" else model.decision_function(X[te_i])
        entry = {"test": metrics(y_all[te_i], s_te, thr=thr), "val_threshold": round(thr, 4)}
        # 5-fold CV F1 on the training portion for a stability estimate
        f1s = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr_i, y_all[tr_i]):
            m = make_models(True).get(name) or RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                                                        n_jobs=-1, random_state=SEED, min_samples_leaf=2)
            m.fit(X[tr_i[a]], y_all[tr_i[a]])
            sv = m.predict_proba(X[tr_i[b]])[:, 1] if name == "random_forest" else m.decision_function(X[tr_i[b]])
            f1s.append(metrics(y_all[tr_i[b]], sv, thr=best_threshold(y_all[tr_i[b]], sv))["f1"])
        entry["cv5_f1_mean"], entry["cv5_f1_std"] = round(float(np.mean(f1s)), 4), round(float(np.std(f1s)), 4)
        entry["train_seconds"] = round(time.time() - t0, 1)
        result["models"][name] = entry
        print(f"[{name}] test f1={entry['test']['f1']:.3f} P={entry['test']['precision']:.3f} R={entry['test']['recall']:.3f} "
              f"pr_auc={entry['test']['pr_auc']:.3f} roc_auc={entry['test']['roc_auc']:.3f} cv5_f1={entry['cv5_f1_mean']:.3f}", flush=True)
        if name == "tfidf+logreg":
            names = np.array(list(vec.get_feature_names_out()) + [f"META:{n}" for n in
                             ("telecommuting", "has_logo", "has_questions", "no_salary", "no_profile", "no_requirements",
                              "no_emp_type", "no_experience")])
            order = np.argsort(model.coef_.ravel())
            tops["recruitment | tfidf+logreg"] = {"toward_legit": names[order[:25]].tolist(),
                                                  "toward_scam": names[order[-25:][::-1]].tolist()}
            joblib.dump({"vectorizer": vec, "model": model, "threshold": thr, "meta_fields": 8},
                        Path(args.models_dir) / "recruitment_logreg.joblib")
    # Text-only variant: the web form only supplies the job description, so this is the deployed model.
    Xt = vec.transform(texts)
    m_txt = make_models(True)["tfidf+logreg"]
    m_txt.fit(Xt[tr_i], y_all[tr_i])
    thr_t = best_threshold(y_all[va_i], m_txt.decision_function(Xt[va_i]))
    result["models"]["tfidf+logreg (text only, deployed)"] = {
        "test": metrics(y_all[te_i], m_txt.decision_function(Xt[te_i]), thr=thr_t), "val_threshold": round(thr_t, 4)}
    joblib.dump({"vectorizer": vec, "model": m_txt, "threshold": thr_t},
                Path(args.models_dir) / "recruitment_text_logreg.joblib")
    t = result["models"]["tfidf+logreg (text only, deployed)"]["test"]
    print(f"[tfidf+logreg text-only] test f1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f} pr_auc={t['pr_auc']:.3f}", flush=True)
    (out / "results_recruitment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, tops


# ----------------------------------------------------------------------------- report

def write_report(out, fn, rc):
    md = []
    if fn:
        md += ["## Fake news: in-domain vs cross-dataset (fake = positive class)\n",
               "| Model | Train set | Test set | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
        for key, res in fn["runs"].items():
            m, tr = [s.strip() for s in key.split("|")]
            for te, v in res.items():
                if isinstance(v, dict):
                    kind = "in-domain" if tr.endswith(te) else "cross"
                    md.append(f"| {m} | {tr.replace('train=', '')} | {te} ({kind}) | {v['accuracy']:.4f} | {v['precision']:.4f} | "
                              f"{v['recall']:.4f} | {v['f1']:.4f} | {v['roc_auc']:.4f} |")
        md.append("")
    if rc:
        md += ["\n## Recruitment scam detection (EMSCAD, scam = positive class)\n",
               "| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | CV5 F1 (mean±std) |", "|---|---|---|---|---|---|---|"]
        for name, e in rc["models"].items():
            t = e["test"]
            cv = f"{e['cv5_f1_mean']:.3f}±{e['cv5_f1_std']:.3f}" if "cv5_f1_mean" in e else "-"
            md.append(f"| {name} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} | {t['pr_auc']:.4f} | {t['roc_auc']:.4f} | {cv} |")
    (out / "results_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fake_news", "recruitment", "all"], default="all")
    ap.add_argument("--out", default=str(ROOT / "text_benchmark_out"))
    ap.add_argument("--models-dir", default=str(ROOT / "trained_models"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    fn = rc = None
    tops = {}
    if args.task in ("fake_news", "all"):
        fn, t = run_fake_news(args, out); tops.update(t)
    if args.task in ("recruitment", "all"):
        rc, t = run_recruitment(args, out); tops.update(t)
    (out / "top_terms.json").write_text(json.dumps(tops, indent=2), encoding="utf-8")
    write_report(out, fn, rc)
    print(f"\nAll outputs in {out}")


if __name__ == "__main__":
    main()
