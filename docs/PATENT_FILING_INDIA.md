# Patent filing guide (India) — XAI-Verify

**Purpose:** You chose to pursue a patent. This document is a **practical checklist** for filing with **IP India**. It is **not legal advice**. A **Registered Patent Agent** must draft claims and the complete specification.

**Working title (adjust with agent):**  
*System and method for evidence-aware multi-threat digital authenticity verification comprising deepfake image analysis, misinformation text analysis, and recruitment fraud screening with unified audit records.*

---

## 1. Final decision (your choice)

| Question | Answer |
|----------|--------|
| Do you want to try for a patent? | **Yes** (your decision) |
| Is grant guaranteed? | **No** — Section **3(k)** and prior art may block broad claims |
| Must you file before thesis/paper? | **Yes** — file **provisional** (or complete) **before** public thesis upload, journal submission, or public GitHub |
| Are you “registered” after building the app? | **No** — registration starts only after **e-filing** on IP India |

---

## 2. What to patent (narrow angles — discuss with agent)

Do **not** claim only: “A web app using EfficientNet and BERT.”  
Prefer **one primary invention** plus dependent claims, grounded in **your implementation**:

1. **Unified evidence-aware authenticity system**  
   - Module routing: image → classifier + Grad-CAM + forensics fallback (`utils/orchestration.py`, `utils/deepfake_models.py`).  
   - Text → transformer / TF-IDF + reliability flags for short input (`utils/text_models.py`).  
   - Recruitment → structured inputs (job text, company, URL) + weighted rule engine (`utils/recruitment.py`, `docs/recruitment_heuristic_spec.md`).  
   - **Single audit store** (`AnalysisRecord`) + PDF report bundling label, scores, and explanation artifacts (`apps/reports/`, `utils/reports.py`).

2. **Recruitment fraud scoring method (if prior art search is clear)**  
   - Neutral prior 50, clamped score, compound rule X1 (payment + unregistered employer), whitelist + TLD heuristics — **if** not anticipated by Fraud-BERT / ORF papers alone as identical method.

3. **Evaluation / deployment methods (from your manuscript — if filing as method claims)**  
   - Grouped-by-employer recruitment validation; cross-dataset protocol for three threat types in one study — stronger for **paper** than product patent unless tied to a **specific system architecture**.

Your agent will pick **one main story** and drop the rest to avoid a bloated, rejectable application.

---

## 3. Step-by-step process (India)

### Phase A — Prepare (1–3 weeks)

1. **Confirm ownership** with university **IP / research cell**:  
   - Who is **applicant** (you, institute, or joint)?  
   - Who are **inventors** (you, supervisor — only humans who contributed to the **inventive concept**)?  
   - Does the institute require **Form 1** in their name?

2. **Hire a Registered Patent Agent**  
   - List: [IP India — Patent Agents](https://ipindia.gov.in/)  
   - Give them: this file, `docs/ARCHITECTURE.md`, `docs/MODELS.md`, `docs/recruitment_heuristic_spec.md`, and (private) evaluation results.

3. **Prior art search** (agent or professional search firm)  
   - Keywords: multi-threat deception, authenticity verification, recruitment ORF, Grad-CAM audit log, multimodal misinformation framework.  
   - Compare to MDAM³, Moyo fake news + deepfake framework, Fraud-BERT, VeriSafe / job fraud systems.

4. **Keep disclosure confidential until filing**  
   - No public GitHub, no thesis portal, no journal upload, no conference proceedings.

### Phase B — Provisional filing (recommended first step)

5. **Draft provisional specification** (agent) including for **each ML module** (IPO AI/ML guidance):
   - Architecture (layers, inputs, outputs).  
   - Training data **characteristics** (size, labeling, domain).  
   - Preprocessing (`text_ml.clean_news`, image forensics steps in `utils/ml_utils.py`).  
   - Training (loss, optimizer, epochs) where applicable.  
   - **Validation metrics** (from your eval datasets / manuscript tables).  
   - System diagrams: web layer → orchestration → models → DB → PDF.

6. **File online** — [IP India e-filing](https://ipindia.gov.in/)

   | Form | Content |
   |------|---------|
   | **Form 1** | Application, applicant, inventors, address |
   | **Form 2** | **Provisional** specification (no formal claims required in provisional, but describe invention fully) |
   | **Form 3** | Foreign filing declaration (if no foreign apps yet, still filed) |
   | **Form 5** | Declaration as to inventorship |
   | **Form 26** | Power of attorney for patent agent |
   | **Form 28** | Fee concession — startup / small entity / educational institute if eligible |

7. **Pay official fee** — check current schedule on IP India (individual / institute rates differ).

8. **Save:** application number, **filing date** (priority date). This is your “patent pending” date.

### Phase C — Within 12 months of provisional

9. **File complete specification** (Form 2 complete) with **claims** (independent + dependent).  
10. Optionally file **Form 9** (request for early publication) if you want faster publication.  
11. **Request for examination (RFE)** — within **48 months** from priority (confirm current rule on IP India).  
12. Respond to **FER** (First Examination Report); amend claims if examiner cites **3(k)** or lack of novelty.

### Phase D — After grant (if any)

13. Pay **renewal fees** yearly.  
14. Mark product **“Patent applied for”** or **“Patent No. …”** only as permitted by law after filing/grant.

---

## 4. Documents to hand your patent agent

| Document | Location in repo |
|----------|------------------|
| System architecture | `docs/ARCHITECTURE.md` |
| Model training / inference | `docs/MODELS.md`, `utils/deepfake_classifier.py`, `utils/text_ml.py` |
| Recruitment rules | `docs/recruitment_heuristic_spec.md`, `utils/recruitment.py` |
| Orchestration API | `utils/orchestration.py` |
| Persistence / evidence | `apps/reports/models.py`, `utils/reports.py` |
| Deployment (for “industrial applicability”) | `docs/DEPLOYMENT.md` |
| Evaluation evidence | `data/deepfake_eval/`, `data/fake_news_eval/`, manuscript results (share privately) |

Agent needs **block diagrams**, **flowcharts**, and **worked examples** (sample image → scores → heatmap → PDF section list).

---

## 5. Fees (indicative — verify on IP India)

- **Official fees:** lower for natural person / startup / institute; higher for complete + examination + renewal.  
- **Agent fees:** often roughly **₹25,000–₹1,50,000+** for provisional + complete drafting (varies by city and complexity).  
- **Institute:** some universities **reimburse** or **mandate** their IP cell to file — ask first.

---

## 6. Timeline vs your thesis

```text
NOW          → Agent + prior art + draft provisional
Week 2–4     → File provisional on IP India  ← YOU ARE "PATENT APPLIED FOR" AFTER THIS
Then         → Submit thesis / paper (if institute allows)
≤ 12 months  → Complete specification + claims
Later        → RFE → examination → grant or refusal (often years)
```

---

## 7. Realistic expectations

- **Provisional filed** = you **started** registration; **not** the same as **granted patent**.  
- Examiners may reject **broad software + ML** claims under **Section 3(k)** unless claims recite **technical effect** (e.g. reduced transcription error, auditable fraud decision pipeline with defined steps) and **enablement** under **Section 10**.  
- Your **strongest academic work** (cross-dataset evaluation, grouped employer test) supports **papers**; the agent must **translate** only **implementable system/method** features into claims.

---

## 8. Email template for supervisor / IP cell

```text
Subject: Provisional patent filing — XAI-Verify (before thesis submission)

Dear [Supervisor / IP Cell],

I intend to file a provisional patent application in India for my M.Tech project
"XAI-Verify" (multi-threat explainable authenticity verification: deepfake images,
fake news, recruitment scam), before submitting the thesis or journal paper.

Please confirm:
1) Applicant name (student / institute / joint)?
2) Inventor list and institute IP policy?
3) Whether the institute provides patent agent support or fee reimbursement?
4) Any required invention disclosure form?

Thesis and paper are not yet publicly submitted.

Regards,
[Your name]
[Enrollment / department]
```

---

## 9. Checklist — “Am I registered?”

- [ ] Patent agent engaged  
- [ ] Prior art search reviewed  
- [ ] Provisional specification drafted  
- [ ] Form 1, 2, 3, 5, 26 (and 28 if applicable) filed on IP India  
- [ ] Filing receipt and **application number** received  
- [ ] Only **after** filing: thesis / paper / public repo as allowed by institute  

**When all filing boxes are checked:** you have **applied** for a patent. **Grant** is a separate step after examination.

---

## 10. References

- [IP India](https://ipindia.gov.in/) — forms, fees, e-filing  
- Patents Act, 1970 — Sections **3(k)**, **6** (inventors), **10** (specification)  
- IPO examination guidelines on **AI/ML/DL** (Sufficiency of disclosure, Scenarios 1–7) — use same structure in your specification  

---

*Last updated for project: XAI-Verify / deepfake_detection_project. Update invention title and claim strategy after agent prior art report.*
