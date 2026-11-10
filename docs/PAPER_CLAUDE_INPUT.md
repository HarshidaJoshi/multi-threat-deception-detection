# XAI-Verify — Complete experiment pack for paper writing (Claude / thesis)

**Project:** A Multi-Threat Explainable AI Framework for Digital Deception Detection  
**Authors (from draft):** Harshida Joshi, Swati Sharma, Ashwini Kumar Jha — Parul University  
**Generated:** 2026-10-08 (local CPU experiments + aggregated benchmarks)

Use this file as the **single paste** into Claude when drafting Methods, Results, Discussion, and Limitations. All numbers below come from saved JSON/MD in this repository unless marked *pending*.

---

## 1. One-paragraph system summary (for Abstract / Introduction)

XAI-Verify is a Django web prototype that detects three deception modalities: (1) real vs AI/deepfake **images** via CNN classifiers with **Grad-CAM** heatmaps; (2) **fake news** text via TF-IDF + logistic regression (and optional BERT in the UI) with **LIME/SHAP**; (3) **recruitment scams** on EMSCAD via the same text pipeline plus rule-based forensics. Each prediction returns a label, confidence, and an explanation artefact for human review.

---

## 2. Datasets (cite these in Methods)

| Modality | Dataset | Split / size | Notes |
|----------|---------|--------------|--------|
| Images | 140k Real and Fake Faces (`real-vs-fake`) | train 100k, valid 10k, test 20k (50/50 real/fake) | Primary in-domain benchmark |
| Images | CIFAKE | train 10k sample (5k/class), test 3k (1.5k/class) | Cross-generator / domain shift |
| Images | Local collection (`datasets/Deepfake image detection`) | 999 images (563 fake) | **Never used for training** in mixed-source protocol |
| Fake news | ISOT Fake News | ~38.8k after cleaning | In-domain + cross-corpus |
| Fake news | WELFake | ~63.1k after cleaning | Cross-corpus |
| Fake news | LIAR | train/valid/test TSV | Political claims; hard transfer |
| Recruitment | EMSCAD | 15,879 postings, 713 scams, 4,767 employer groups | Severe class imbalance |

---

## 3. Image experiments

### 3.1 CNN benchmark (subset run, seed 42)

**Script:** `scripts/train_benchmark_gpu.py`  
**Config:** max 5000 images/class train, 3000/class eval; 4 head epochs + 2 fine-tune epochs; batch 16; augment + two-stage fine-tune.

| Model | Accuracy | Precision | Recall | F1 | AUC | Train time (s) |
|-------|----------|-----------|--------|-----|-----|----------------|
| ResNet50 | 0.8988 | 0.9247 | 0.8683 | 0.8957 | 0.9670 | 9005 |
| EfficientNetB0 | 0.8387 | 0.8543 | 0.8167 | 0.8350 | 0.9228 | 3548 |
| MobileNetV2 | 0.8268 | 0.8132 | 0.8487 | 0.8305 | 0.9096 | 2512 |
| SimpleCNN | 0.5898 | 0.5718 | 0.7157 | 0.6357 | 0.6326 | 5103 |

**EfficientNetB0 ablation (same subset):**

| Setting | Accuracy | F1 | AUC |
|---------|----------|-----|-----|
| Full (aug + fine-tune) | 0.8387 | 0.8350 | 0.9228 |
| No fine-tune | 0.8092 | 0.8046 | 0.8941 |
| No augmentation | 0.8607 | 0.8628 | 0.9380 |

**Source file:** `benchmark_out/results_table.md`

### 3.2 Full training set — frozen ResNet50 + heads (CPU, all 100k train)

**Script:** `scripts/train_full_frozen_features.py`  
**Test:** full 20,000 held-out images.

| Head | Test accuracy [±95% CI] | F1 | AUC | n |
|------|-------------------------|-----|-----|---|
| Logistic regression | 0.9002 ± 0.0042 | 0.901 | 0.9646 | 20000 |
| MLP (512→128→sigmoid) | **0.9536 ± 0.0029** | **0.9535** | **0.9904** | 20000 |

**Deployed image model:** `trained_models/deepfake_resnet50_full.keras` (ResNet50 frozen + MLP pipeline equivalent).

**Source:** `benchmark_out/full_frozen/results_full_frozen.json`

### 3.3 Robustness: in-domain CI, cross-domain, McNemar

**Script:** `scripts/evaluate_robustness.py`  
**In-domain (faces test, bootstrap 95% CI):**

| Model | Accuracy [95% CI] | F1 [95% CI] | AUC [95% CI] |
|-------|-------------------|-------------|--------------|
| resnet50_full_mlp | 0.954 [0.9485, 0.9593] | 0.954 [0.948, 0.9591] | 0.991 [0.9898, 0.9928] |
| resnet50 | 0.905 [0.8972, 0.9118] | 0.903 [0.8946, 0.9095] | 0.971 [0.9671, 0.9739] |
| efficientnetb0 | 0.847 [0.8377, 0.8555] | 0.845 [0.8348, 0.8538] | 0.931 [0.9244, 0.936] |
| mobilenetv2 | 0.835 [0.826, 0.8442] | 0.840 [0.8303, 0.8492] | 0.918 [0.911, 0.9243] |
| simplecnn | 0.595 [0.5825, 0.6062] | 0.642 [0.6288, 0.6541] | 0.638 [0.624, 0.6505] |

**Cross-domain (accuracy / F1 / AUC) — models trained on faces, tested without retraining:**

| Model | CIFAKE test | Local own images |
|-------|-------------|------------------|
| resnet50_full_mlp | 0.499 / 0.001 / 0.395 | 0.438 / 0.028 / 0.441 |
| resnet50 | 0.498 / 0.000 / 0.419 | 0.431 / 0.069 / 0.401 |
| efficientnetb0 | 0.499 / 0.001 / 0.419 | 0.429 / 0.069 / 0.417 |

**Key finding for Discussion:** Strong in-domain performance (~95%) does **not** transfer to CIFAKE or the local 999-image set (~50% accuracy, F1 near zero for best models on CIFAKE).

**McNemar:** resnet50_full_mlp significantly outperforms all other models on the faces test set (p ≪ 0.05 for all pairs listed in `benchmark_out/robustness_tables.md`).

### 3.4 Mixed-source training ablation (FINAL EXPERIMENT)

**Script:** `scripts/train_mixed_source.py`  
**Design:** Frozen ResNet50 features; MLP head; 3 random seeds; train under four data regimes; test on faces test, CIFAKE test, and local unseen images.

| Training data | Faces test (20k) | CIFAKE test (3k) | Local unseen (999) |
|---------------|------------------|------------------|---------------------|
| Faces only (100k) | 0.958 ± 0.004 / 0.992 ± 0.001 | 0.499 / 0.407 | 0.435 / 0.431 |
| CIFAKE only (9k) | 0.480 ± 0.009 / 0.430 ± 0.006 | 0.934 ± 0.004 / 0.983 ± 0.001 | **0.585 ± 0.005 / 0.592 ± 0.006** |
| Faces + CIFAKE (all faces) (109k) | 0.960 ± 0.003 / 0.992 ± 0.001 | 0.929 ± 0.001 / 0.981 ± 0.001 | 0.423 / 0.411 |
| Faces + CIFAKE (balanced) (18k) | 0.884 ± 0.006 / 0.955 ± 0.003 | 0.934 ± 0.002 / 0.982 ± 0.001 | 0.479 / 0.477 |

**Source:** `benchmark_out/mixed/results_mixed.md` (final; also in journal `manuscript.tex` §Mixed training sources).

### 3.5 Grad-CAM quantitative check

**Sample:** 400 test images (200 real, 200 fake); model resnet50_full_mlp; accuracy on sample 95%.

| Split | Central mass mean | Ratio vs uniform |
|-------|-------------------|------------------|
| All | 0.367 | 1.468 |
| Correct predictions | 0.365 | 1.461 |
| Incorrect | 0.397 | 1.589 |

**Source:** `benchmark_out/gradcam_attention_full_model.json`

### 3.6 Optional local holdout (app weights)

**Script:** `scripts/evaluate_deepfake.py` on local test set (499 images) → **Accuracy 74.75%, Precision 92.83%, Recall 73.26%, F1 81.90%, AUC 0.8585** (`data/deepfake_eval/deepfake_results.md`). Use to discuss domain mismatch vs faces benchmark (not main 20k test).

---

## 4. Text experiments

### 4.1 Fake news — TF-IDF + logistic regression

**Script:** `scripts/train_text_benchmark.py` + `scripts/extra_text_experiments.py`

**Cross-corpus (bootstrap 95% CI on accuracy):**

| Train → Test | Accuracy [95% CI] | F1 | AUC |
|--------------|-------------------|-----|-----|
| ISOT → ISOT | 0.984 [0.981, 0.987] | 0.982 | 0.999 |
| ISOT → WELFake | 0.822 [0.815, 0.830] | 0.827 | 0.911 |
| ISOT → LIAR | 0.484 [0.457, 0.511] | 0.547 | 0.540 |
| WELFake → WELFake | 0.957 [0.953, 0.961] | 0.952 | 0.992 |
| Pooled (all 3) → LIAR | 0.599 [0.571, 0.624] | 0.566 | 0.632 |

**Leave-one-corpus-out:** held-out ISOT → 0.989 acc; held-out WELFake → 0.820 acc; held-out LIAR → 0.458 acc.

**Source:** `text_benchmark_out/extra_tables.md`, `results_fake_news.json`

### 4.2 Recruitment scams — EMSCAD

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|-----|--------|---------|
| Rule-based (deployed heuristics) | 0.056 | 0.065 | 0.060 | 0.046 | 0.506 |
| TF-IDF + logreg (deployed) | 0.895 | 0.720 | 0.798 | 0.837 | 0.963 |
| TF-IDF + linear SVM | 0.907 | 0.729 | 0.808 | 0.875 | 0.987 |

**Grouped by employer (5-fold CV):** F1 **0.600 ± 0.158**, PR-AUC **0.694 ± 0.128** (vs random hold-out F1 **0.849 ± 0.033**).

**Source:** `text_benchmark_out/results_tables.md`, `extra_tables.md`

---

## 5. Suggested paper storyline (copy to Claude as instructions)

1. **Problem:** Multi-modal deception; need evidence-aware decisions, not black-box scores.  
2. **Method:** XAI-Verify architecture + public datasets + frozen-feature / CNN training protocols above.  
3. **Results:** Strong in-domain image and news metrics; **document domain shift** (CIFAKE, local images, LIAR); mixed-source table shows whether multi-dataset training helps local/CIFAKE.  
4. **XAI:** Grad-CAM central-mass statistic; LIME/SHAP as audit trail (qualitative in UI).  
5. **Limitations:** No multi-seed GPU full fine-tune on 100k yet (optional future work — see `docs/GPU_RUN_GUIDE.md`); recruitment grouped CV much harder than random split; LIAR label semantics differ from news corpora.

---

## 6. Figures and files to attach in supplementary material

| Asset | Path |
|-------|------|
| ROC curves (subset benchmark) | `benchmark_out/roc_curves.png` |
| Confusion matrices | `benchmark_out/confusion_*.png` |
| Taxonomy figure (survey) | `docs/figures/sicacs2026_taxonomy_figure.png` |
| Robustness tables | `benchmark_out/robustness_tables.md` |
| Mixed-source results | `benchmark_out/mixed/results_mixed.md` |

---

## 7. Claude prompt template

```
You are helping write a journal article for Taylor & Francis "Cybernetics and Systems" on XAI-Verify.
Emphasise intelligent systems, applied AI, human–machine explainability, and cross-environment generalisation.
Use ONLY the numbers in the attached PAPER_CLAUDE_INPUT.md. Do not invent metrics.
Structure: Abstract, Introduction, Related Work (brief), Methods, Experimental Setup,
Results (tables as in section 3–4), Discussion (domain shift + XAI), Limitations, Conclusion.
Tone: formal, third person, past tense for experiments.
```

---

## 8. Not yet run (optional upgrade)

- **`scripts/gpu_run_all.py`** on Kaggle/Colab: 3 seeds, **full** 100k fine-tuning, DistilBERT/BERT text baselines → `aggregated_tables.md`.  
  See `docs/GPU_RUN_GUIDE.md`. Current paper pack is complete **without** this for a solid thesis/conference draft.
