"""Extra text experiments that make the study journal-grade (CPU only, minutes):

Fake news
  * bootstrap 95% CIs for every cell of the cross-dataset matrix (TF-IDF + logistic regression)
  * leave-one-corpus-out: train on two corpora pooled, test on the third (remedy: more diverse sources)
  * pooled training on all three corpora, tested on each
Recruitment (EMSCAD)
  * repeated stratified hold-out (5 seeds) -> mean +/- std
  * GROUPED cross-validation by employer (company_profile): does the model memorise employers?
  * bootstrap CIs for the deployed text-only model on the fixed test split

Outputs: text_benchmark_out/extra_results.json and extra_tables.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import train_text_benchmark as ttb  # noqa: E402

OUT = ROOT / "text_benchmark_out"
SEED = 42


# ------------------------------------------------------------------------------------------ helpers
def boot_ci(y, score, thr=0.0, n_boot=1000, seed=SEED):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    rng = np.random.RandomState(seed)
    y, score = np.asarray(y), np.asarray(score)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        yy, ss = y[i], score[i]
        if len(set(yy)) < 2:
            continue
        accs.append(accuracy_score(yy, ss >= thr)); f1s.append(f1_score(yy, ss >= thr, zero_division=0)); aucs.append(roc_auc_score(yy, ss))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"acc_ci95": q(accs), "f1_ci95": q(f1s), "auc_ci95": q(aucs)}


def fit_tfidf_lr(train_rows, balanced=True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True, stop_words="english")
    X = vec.fit_transform([t for t, _ in train_rows])
    y = np.array([l for _, l in train_rows])
    m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced" if balanced else None, solver="liblinear").fit(X, y)
    return vec, m


def eval_rows(vec, m, rows):
    X = vec.transform([t for t, _ in rows])
    y = np.array([l for _, l in rows])
    s = m.decision_function(X)
    r = ttb.metrics(y, s)
    r.update(boot_ci(y, s))
    return r


# ------------------------------------------------------------------------------------------ fake news
def fake_news():
    print("== fake news ==", flush=True)
    corp = {"ISOT": ttb.load_isot(ROOT), "WELFake": ttb.load_welfake(ROOT)}
    sp = {k: ttb.stratified_split(v) for k, v in corp.items()}
    liar = {s: ttb.load_liar(ROOT, s) for s in ("train", "valid", "test")}
    train = {"ISOT": sp["ISOT"][0], "WELFake": sp["WELFake"][0], "LIAR": liar["train"]}
    test = {"ISOT": sp["ISOT"][2], "WELFake": sp["WELFake"][2], "LIAR": liar["test"]}
    res = {"matrix_ci": {}, "leave_one_out": {}, "pooled": {}}

    for tr in train:  # 3x3 matrix with bootstrap CIs
        vec, m = fit_tfidf_lr(train[tr])
        for te in test:
            res["matrix_ci"][f"train={tr}|test={te}"] = eval_rows(vec, m, test[te])
        print("matrix row", tr, flush=True)

    for held in train:  # leave-one-corpus-out
        pool = [r for k, v in train.items() if k != held for r in v]
        vec, m = fit_tfidf_lr(pool)
        r = eval_rows(vec, m, test[held])
        res["leave_one_out"][f"heldout={held}"] = {"train_corpora": [k for k in train if k != held], "n_train": len(pool), **r}
        print("LOCO", held, r["accuracy"], r["f1"], r["roc_auc"], flush=True)

    pool = [r for v in train.values() for r in v]  # pooled on all three
    vec, m = fit_tfidf_lr(pool)
    for te in test:
        res["pooled"][f"test={te}"] = {"n_train": len(pool), **eval_rows(vec, m, test[te])}
        print("pooled ->", te, res["pooled"][f"test={te}"]["accuracy"], flush=True)
    return res


# ------------------------------------------------------------------------------------------ recruitment
_TEXT_FIELDS = ("title", "company_profile", "description", "requirements", "benefits")


def load_emscad_groups():
    rows, seen = [], set()
    with open(ROOT / "datasets/text/emscad/fake_job_postings.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            text = re.sub(r"\s+", " ", " . ".join(r[f] for f in _TEXT_FIELDS if r[f])).strip()
            if text in seen or len(text.split()) < 5:
                continue
            seen.add(text)
            prof = re.sub(r"\s+", " ", r["company_profile"]).strip().lower()
            group = prof if prof else f"__noprofile_{r['job_id']}"
            rows.append((text, int(r["fraudulent"]), group))
    return rows


def best_thr(y, s):
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
    return float(t[int(np.argmax(f))])


def text_lr(texts_tr, y_tr):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(texts_tr)
    return vec, LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", solver="liblinear").fit(X, y_tr)


def recruitment():
    from sklearn.model_selection import GroupKFold, train_test_split

    print("== recruitment ==", flush=True)
    data = load_emscad_groups()
    texts = np.array([d[0] for d in data], dtype=object)
    y = np.array([d[1] for d in data])
    groups = np.array([d[2] for d in data], dtype=object)
    n_groups = len(set(groups))
    print(f"{len(data)} postings, {int(y.sum())} scams, {n_groups} distinct employer groups", flush=True)
    res = {"n": len(data), "scams": int(y.sum()), "employer_groups": n_groups}

    def run(tr_i, va_i, te_i):
        vec, m = text_lr(texts[tr_i], y[tr_i])
        sv = m.decision_function(vec.transform(texts[va_i]))
        thr = best_thr(y[va_i], sv)
        st = m.decision_function(vec.transform(texts[te_i]))
        r = ttb.metrics(y[te_i], st, thr=thr)
        return r, st, thr

    # repeated stratified hold-out (70/15/15), 5 seeds
    reps = []
    for seed in range(5):
        idx = np.arange(len(y))
        tr_i, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va_i, te_i = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
        r, _, _ = run(tr_i, va_i, te_i)
        reps.append(r)
        print(f"  random split {seed}: F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}", flush=True)
    res["repeated_random_holdout"] = {k: [round(float(np.mean([r[k] for r in reps])), 4), round(float(np.std([r[k] for r in reps], ddof=1)), 4)]
                                      for k in ("precision", "recall", "f1", "pr_auc", "roc_auc")}

    # grouped 5-fold by employer: validation threshold taken from a random 15% of the training fold's *groups*
    gk = GroupKFold(n_splits=5)
    greps = []
    for fold, (tr_all, te_i) in enumerate(gk.split(texts, y, groups)):
        ug = np.array(sorted(set(groups[tr_all])), dtype=object)
        rng = np.random.RandomState(fold)
        val_g = set(rng.choice(ug, size=max(1, int(0.15 * len(ug))), replace=False))
        va_mask = np.array([g in val_g for g in groups[tr_all]])
        tr_i, va_i = tr_all[~va_mask], tr_all[va_mask]
        if y[va_i].sum() == 0 or y[te_i].sum() == 0:
            continue
        r, _, _ = run(tr_i, va_i, te_i)
        greps.append(r)
        print(f"  grouped fold {fold}: F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} (scams in test fold: {int(y[te_i].sum())})", flush=True)
    res["grouped_by_employer_cv"] = {"folds": len(greps), **{k: [round(float(np.mean([r[k] for r in greps])), 4), round(float(np.std([r[k] for r in greps], ddof=1)), 4)]
                                                           for k in ("precision", "recall", "f1", "pr_auc", "roc_auc")}}

    # bootstrap CI for the fixed split used elsewhere (same as the deployed model's split)
    rows = [(i, int(v)) for i, v in enumerate(y)]
    tr, va, te = ttb.stratified_split(rows)
    tr_i, va_i, te_i = [np.array([i for i, _ in part]) for part in (tr, va, te)]
    r, st, thr = run(tr_i, va_i, te_i)
    rng = np.random.RandomState(SEED)
    from sklearn.metrics import f1_score, precision_score, recall_score

    f1s, ps, rs = [], [], []
    yt = y[te_i]
    for _ in range(1000):
        b = rng.randint(0, len(yt), len(yt))
        pred = (st[b] >= thr).astype(int)
        f1s.append(f1_score(yt[b], pred, zero_division=0)); ps.append(precision_score(yt[b], pred, zero_division=0)); rs.append(recall_score(yt[b], pred, zero_division=0))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    res["fixed_split_text_only"] = {**r, "f1_ci95": q(f1s), "precision_ci95": q(ps), "recall_ci95": q(rs)}
    print("fixed split text-only:", r["f1"], q(f1s), flush=True)
    return res


def write_md(fn, rc):
    md = ["## Fake news: bootstrap 95% CIs (logistic regression)\n", "| Train | Test | Accuracy [95% CI] | F1 | AUC |", "|---|---|---|---|---|"]
    for k, v in fn["matrix_ci"].items():
        tr, te = [p.split("=")[1] for p in k.split("|")]
        md.append(f"| {tr} | {te} | {v['accuracy']:.3f} {v['acc_ci95']} | {v['f1']:.3f} | {v['roc_auc']:.3f} |")
    md += ["\n## Fake news: leave-one-corpus-out (train on the other two, pooled)\n", "| Held-out corpus | Train corpora | Accuracy [95% CI] | F1 | AUC |", "|---|---|---|---|---|"]
    for k, v in fn["leave_one_out"].items():
        md.append(f"| {k.split('=')[1]} | {' + '.join(v['train_corpora'])} | {v['accuracy']:.3f} {v['acc_ci95']} | {v['f1']:.3f} | {v['roc_auc']:.3f} |")
    md += ["\n## Fake news: pooled training on all three corpora\n", "| Test | Accuracy [95% CI] | F1 | AUC |", "|---|---|---|---|"]
    for k, v in fn["pooled"].items():
        md.append(f"| {k.split('=')[1]} | {v['accuracy']:.3f} {v['acc_ci95']} | {v['f1']:.3f} | {v['roc_auc']:.3f} |")
    md += ["\n## Recruitment (text-only logistic regression)\n", f"{rc['n']} postings, {rc['scams']} scams, {rc['employer_groups']} employer groups.\n",
           "| Protocol | Precision | Recall | F1 | PR-AUC |", "|---|---|---|---|---|"]
    fmt = lambda d, k: f"{d[k][0]:.3f} ± {d[k][1]:.3f}"
    r1, r2 = rc["repeated_random_holdout"], rc["grouped_by_employer_cv"]
    md.append(f"| Random hold-out, 5 seeds | {fmt(r1,'precision')} | {fmt(r1,'recall')} | {fmt(r1,'f1')} | {fmt(r1,'pr_auc')} |")
    md.append(f"| Grouped by employer, {r2['folds']}-fold CV | {fmt(r2,'precision')} | {fmt(r2,'recall')} | {fmt(r2,'f1')} | {fmt(r2,'pr_auc')} |")
    f = rc["fixed_split_text_only"]
    md.append(f"\nFixed split (deployed): F1 {f['f1']:.3f} {f['f1_ci95']}, precision {f['precision']:.3f} {f['precision_ci95']}, recall {f['recall']:.3f} {f['recall_ci95']}.")
    (OUT / "extra_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    fn = fake_news()
    (OUT / "extra_results_fake_news.json").write_text(json.dumps(fn, indent=2), encoding="utf-8")
    rc = recruitment()
    (OUT / "extra_results_recruitment.json").write_text(json.dumps(rc, indent=2), encoding="utf-8")
    write_md(fn, rc)
    print(f"All outputs in {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
