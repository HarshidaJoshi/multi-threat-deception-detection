# Submitting to *Cybernetics and Systems* (Taylor & Francis)

**Journal:** [Cybernetics and Systems: An International Journal](https://www.tandfonline.com/journals/ucbs20)  
**ISSN area:** Cybernetics, systems theory, applied AI/ML, intelligent systems, human–machine interaction, decision-making under uncertainty.

**Your manuscript file:** `docs/journal_manuscript/manuscript.tex` (compile with `pdflatex` twice, or Overleaf).

**Submission portal:** https://rp.tandfonline.com/submission/create?journalCode=UCBS  

**Instructions for authors:** https://www.tandfonline.com/action/authorSubmission?show=instructions&journalCode=ucbs20  

**Review:** Double-anonymous peer review (remove author names from the PDF you upload; use a separate title page if required).

**Open access:** Hybrid (*Open Select*) — you can publish subscription-only (no APC) or pay APC for gold OA if your institution has no agreement.

---

## 1. Why this journal fits your paper

The journal explicitly lists:

- **Applied Artificial Intelligence and Machine Learning**
- **Intelligent Systems and Deep Learning**
- **Human–Machine Interaction**
- **Decision Making** (uncertainty, fuzzy systems)

Frame **XAI-Verify** as an **intelligent multi-module system** for deception sensing with **human-auditable feedback** (Grad-CAM, exact word contributions), not as “yet another deepfake CNN.” Your **cross-dataset generalisation** results are a **systems / cybernetics** story: performance depends on the **environment** (corpus, generator, employer group), not only on in-domain accuracy.

**Suggested subtitle angle (optional title tweak):**  
*“A Cross-Dataset Evaluation of an Explainable Multi-Module System for Digital Deception Detection”*

---

## 2. Novelty for *Cybernetics and Systems* reviewers

| Claim | Evidence in repo |
|--------|------------------|
| Unified **protocol** for three deception channels with **cross-domain** tests | `manuscript.tex`, `evaluate_robustness.py`, text benchmarks |
| **Generalisation gap** quantified (images ~95% → ~50%; news ~98% → ~48% on LIAR) | `robustness_tables.md`, `extra_tables.md` |
| **Mixed-source control experiment** (faces vs CIFAKE vs local) | `benchmark_out/mixed/results_mixed.md`, §\ref{sec:mixed} in manuscript |
| **Human–machine audit trail** (model-native explanations, not decorative) | Grad-CAM + central-mass metric; TF-IDF exact contributions |
| **Grouped evaluation** (employer hold-out) vs random split | EMSCAD tables in `text_benchmark_out/` |
| Deployable **prototype** (Django XAI-Verify) | Secondary; one subsection or supplementary |

**Avoid claiming:** state-of-the-art deepfake accuracy; “first fake news detector”; novelty of ResNet/EfficientNet architectures.

---

## 3. Commands — refresh all metrics before submission

Run from `F:\deepfake_detection_project`:

```powershell
cd F:\deepfake_detection_project

.\venv\Scripts\python.exe scripts\train_mixed_source.py
.\venv\Scripts\python.exe scripts\evaluate_robustness.py --cifake-per-class 1500
.\venv\Scripts\python.exe scripts\gradcam_attention_analysis.py --per-class 250
.\venv\Scripts\python.exe scripts\make_gradcam_figure.py
.\venv\Scripts\python.exe scripts\evaluate_deepfake.py
.\venv\Scripts\python.exe scripts\train_text_benchmark.py
.\venv\Scripts\python.exe scripts\extra_text_experiments.py
.\venv\Scripts\python.exe scripts\generate_results_tables.py
```

**Optional (strengthens “Applied AI” for reviewers):** GPU run → `docs/GPU_RUN_GUIDE.md` → `gpu_run_all.py` → download `aggregated_tables.md`.

**Build PDF:**

```powershell
cd docs\journal_manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

---

## 4. Files to zip for Claude / co-author revision

Attach only these (numbers must match manuscript):

- `docs/journal_manuscript/manuscript.tex`
- `benchmark_out/robustness_tables.md`
- `benchmark_out/mixed/results_mixed.md`
- `benchmark_out/results_table.md`
- `text_benchmark_out/extra_tables.md`
- `data/deepfake_eval/deepfake_results.md`
- `docs/PAPER_CLAUDE_INPUT.md` (update §3.4–3.6 before use)

**Claude prompt (journal-specific):**

```text
Revise the attached manuscript for Taylor & Francis journal "Cybernetics and Systems".
Emphasise: intelligent multi-module system, human–machine explainability, decision-making under
dataset shift / open environments. Use ONLY attached table numbers. Keep double-anonymous tone
(no author names in body). Shorten if over ~12,000 words. Strengthen Discussion with cybernetics
language (feedback, environment, sensor generalisation) without jargon overload.
```

---

## 5. Taylor & Francis submission checklist

- [ ] **Anonymous manuscript PDF** (no author names, affiliations, or acknowledgments identifying you)
- [ ] **Title page** (separate file if portal asks): names, affiliations, ORCID, corresponding email
- [x] **Abstract** (journal requirement: unstructured, 100 words)
- [x] **Keywords** (journal requirement: no more than 5)
- [x] **Figures** at publication quality: `figs/system_architecture.png` and `figs/gradcam_examples.png` (300 dpi)
- [ ] **References** in journal style (author–date or numbered per T&F template — check Instructions)
- [ ] **Data / code availability** statement (GitHub or supplementary ZIP; no private paths)
- [ ] **Ethics / consent** if using human-facing demo (brief statement)
- [ ] **Conflict of interest** and **funding** declarations
- [ ] **Cover letter** (1 page): fit to *Applied AI* + *intelligent systems* scope; highlight cross-domain + XAI audit trail

---

## 6. Cover letter bullets (copy and edit)

1. We submit an empirical study of an **explainable multi-module system** for three deception modalities (synthetic faces, fake news, recruitment scams).

2. We contribute a **reproducible cross-dataset protocol** with confidence intervals and significance tests, showing that **in-domain accuracy is a poor indicator** of behaviour on new generators, corpora, or employers.

3. A **mixed-source training experiment** shows that combining controlled training environments improves transfer between those environments but **not** to an independently collected image set.

4. Explanations are **computed from the deployed models** (Grad-CAM; exact linear word contributions) to support **human–machine review**, aligned with the journal’s interest in applied AI and intelligent systems.

5. We state **limitations** explicitly (CPU training, frozen backbone at full scale, corpus-specific cues).

---

## 7. After acceptance

- Respond to reviewers by pointing to **Table cross-domain**, **Table mixed-source**, and **grouped EMSCAD** — these address “real-world generalisation” comments.
- If asked for **transformer baselines**, cite optional `gpu_run_all.py` results or add one DistilBERT row from a short GPU run.

---

*Last updated for target journal: Cybernetics and Systems (ucbs20), Taylor & Francis.*
