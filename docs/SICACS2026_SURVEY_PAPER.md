# SICACS / ICACS 2026 — Survey Paper (Draft)

**Conference:** 2nd International Conference on Artificial Intelligence, Communication Technologies & Smart Cities  
**Submission portal:** [Microsoft CMT — SICACS2026](https://cmt3.research.microsoft.com/SICACS2026/Submission/Index)  
**Type:** Full-length **Review / Survey** paper (conference asks for **minimum 10 pages**; adapt into their Word/LaTeX template before upload)  
**Source topic:** Multi-threat explainable AI for digital deception detection (DeepFake + fake news + recruitment scam)

---

## CHOOSE YOUR CASE (reply with A / B / C / D / E)

Pick one angle before final formatting. The **full draft below is written for Case A** (recommended). If you choose another case, say which one and the abstract/introduction will be rewritten to that focus.

| Case | Suggested title | Focus | Best if you want… |
|------|-----------------|-------|-------------------|
| **A (recommended)** | A Survey on Multi-Threat Explainable AI for Digital Deception Detection: DeepFakes, Fake News, and Recruitment Fraud | Joint survey of three threats + XAI | Closest to your dissertation; strongest novelty story |
| **B** | Explainable DeepFake Detection with CNN and EfficientNet Models: A Survey Oriented to Evidence-Aware Authentication | Image + Grad-CAM deep dive | Shorter path; sharper technical claims |
| **C** | BERT, LIME, and SHAP for Fake News and Job Scam Detection: A Survey of Explainable Text Authenticity Methods | Text deception only | Fast to finish; clean BERT/XAI narrative |
| **D** | From Smart Platforms to Trusted Content: A Survey of AI Methods for Authenticity Verification in Digital Cities | Smart-city / platform trust framing | Stronger match to conference theme wording |
| **E** | Comparative Survey of EfficientNet–Grad-CAM and BERT–LIME/SHAP Pipelines for Deception Detection | Method comparison survey | Good for tables and viva-style clarity |

**Status:** Case **A** selected and finalised → see `docs/SICACS2026_CASE_A_FINAL.md` + figure `docs/figures/sicacs2026_taxonomy_figure.png`.

---

# CASE A — FULL SURVEY DRAFT (superseded by CASE_A_FINAL)

**Title:** A Survey on Multi-Threat Explainable AI for Digital Deception Detection: DeepFakes, Fake News, and Recruitment Fraud

**Authors:** *[Add your name, affiliation, email]*  
**Affiliation:** *[Department / Institute]*

---

## Abstract

Online deception now arrives in several forms at once. A forged face can circulate as a profile photograph, a misleading news post can reshape public discussion within minutes, and a false job advertisement can steal money or personal data from applicants. Each of these problems has been studied extensively on its own, yet many deployed systems still treat them as separate pipelines. This survey reviews recent machine learning and deep learning work on DeepFake image detection, fake news classification, and recruitment scam detection, with special attention to methods that also explain their decisions. We organise the literature around five practical families of technique that appear repeatedly in strong systems: EfficientNet-style convolutional backbones, Grad-CAM visual evidence, BERT-oriented text encoders, and LIME/SHAP word-level explanations. Across the surveyed studies we observe three recurring gaps. First, high benchmark scores often drop under cross-dataset or real-user conditions. Second, explanation is frequently added after training rather than designed as part of the decision process. Third, recruitment fraud remains under-represented compared with fake news and DeepFake research, even though it shares many linguistic cues with other text scams. From these observations we outline an evidence-aware multi-threat view of authenticity verification and list open research questions that matter for safer communication platforms and smarter digital services.

**Keywords:** DeepFake detection, fake news, recruitment fraud, EfficientNet, Grad-CAM, BERT, LIME, SHAP, explainable AI, digital authenticity

---

## 1. Introduction

Digital platforms have made content creation cheap and content checking expensive. A smartphone and a free editing tool are enough to publish a face that never existed, rewrite a headline that looks official, or post a hiring advert that impersonates a known company. For ordinary users the three threats feel different—images look “real,” news feels “shared,” and jobs feel “urgent”—but from a security point of view they belong to the same family of problems: synthetic or deceptive digital content that aims to transfer trust from an unreliable message to a human decision.

Early research answered each threat separately. DeepFake detection grew around convolutional networks and, later, transformers trained on FaceForensics++, Celeb-DF, and similar corpora. Fake news detection moved from shallow bag-of-words models to BERT-style contextual classifiers. Recruitment scam studies arrived later and remain fewer; many of them still depend on classical machine learning or on social science interviews rather than large labelled job-ad corpora. As accuracy numbers rose, another issue became hard to ignore. Black-box scores give little help to a moderator, auditor, or job seeker who asks a simple question: *why was this sample marked fake?*

Explainable artificial intelligence (XAI) tried to answer that question. Grad-CAM heatmaps highlight facial regions used by image classifiers. LIME and SHAP allocate importance to words and phrases in text models. These tools do not automatically make a system trustworthy, but they turn a scalar prediction into something closer to usable evidence. Recent work therefore links detection and explanation more tightly, especially for DeepFake and fake news tasks.

This survey has three aims. First, it summarizes the main technical routes for the three deception types listed above. Second, it places EfficientNet-family image models, Grad-CAM, BERT, LIME, and SHAP inside that map, because these methods are currently among the most practical choices for evidence-aware systems. Third, it identifies open gaps that suggest a multi-threat framework rather than a set of isolated detectors. The discussion is written with communication platforms and smart digital services in mind: authenticity checking is becoming a civic and economic requirement, not only a research demo.

The rest of the paper is organised as follows. Section 2 describes earlier surveys and the taxonomy used here. Sections 3–5 review DeepFake, fake news, and recruitment fraud studies. Section 6 focuses on explainability methods. Section 7 compares representative systems in tabular form. Section 8 discusses challenges and research directions. Section 9 concludes.

---

## 2. Related Surveys and Taxonomy

### 2.1 Earlier survey work

Several surveys already map parts of this space. Mirsky and Lee give a long-standing account of how DeepFakes are created and detected [1]. Later reviews discuss modern detectors, datasets, and evaluation pitfalls, including domain shift across forgery methods and compression settings [2], [3]. On the text side, Alghamdi et al. and related reviews classify fake news methods into content-based, propagation-based, and multimodal lines [4], [5]. Recruitment fraud has fewer dedicated technical surveys; most coverage appears as sections inside broader online fraud or labour-market studies, or as individual system papers such as Fraud-BERT [6].

A more recent systematic review by Moyo et al. treats fake news and deepfakes together and argues that technical success alone is not enough without social and governance context [7]. That point matters for this survey: platforms in digital cities host all three threat types, and users rarely see them as separate research communities.

### 2.2 Taxonomy used in this paper

We group work along four axes:

1. **Threat modality** — image (DeepFake), text (fake news), text in hiring context (recruitment scam).  
2. **Learning family** — classical ML, CNN/EfficientNet, transformers/BERT, hybrid/multimodal.  
3. **Explanation style** — none, visual (Grad-CAM and relatives), local text (LIME/SHAP), design-level explanations.  
4. **Deployment concern** — accuracy only, cross-dataset robustness, imbalance handling, human interpretability.

This taxonomy keeps the survey close to engineering choices rather than to publisher categories alone.

---

## 3. DeepFake Image Detection

### 3.1 Problem setting

DeepFake systems synthesise or edit faces through generative models. Common attacks include face swapping, reenactment, attribute editing, and fully generated portraits. Early detectors looked for blending artefacts and local texture breaks. As generators improved, those cues became weaker, and detectors had to learn more abstract inconsistencies.

### 3.2 CNN and EfficientNet approaches

Convolutional networks remain a practical starting point for still-image analysis. EfficientNet variants are particularly popular because they balance depth, width, and resolution under a smaller parameter budget than older heavy CNNs. Kumar et al. combine spatial and frequency cues with an EfficientNet backbone and use attention plus Grad-CAM for interpretation, reporting very strong AUC on controlled tests [8]. Deepa et al. compare classical texture pipelines with EfficientNet-B3 and InceptionV3; deep models win by a clear margin, with EfficientNet-B3 near 98% accuracy in their setting [9]. Compact designs and synthetic data expansion have also been explored when labelled forgery data are scarce [10].

For many applications the choice between EfficientNet-B0 and heavier siblings is a systems trade-off. B0 is lighter for prototypes and edge-friendly services; larger variants often win pure accuracy contests. What matters for an authenticity platform is not only which letter appears after “EfficientNet,” but whether the model can be paired with an explanation that humans can inspect.

### 3.3 Transformers and hybrid vision models

Vision transformers and hybrid CNN–transformer designs try to catch long-range inconsistencies that local convolutions miss. Studies based on convolutional vision transformers and CSWin-style backbones report high accuracy on FaceForensics++ / Celeb-DF-like data [11], [12]. Survey comparisons place some hybrid systems around the high 90s for accuracy on standard splits [3]. The recurring warning is the same: when compression level, generator family, or identity distribution changes, scores often fall. Real uploads from social platforms rarely match the clean laboratory condition of a leaderboard table.

### 3.4 Explainable DeepFake detection

Moderators and investigators need more than a probability bar. DeepDect-style platforms attach Grad-CAM heatmaps and short notes to face-swap and face-generation decisions [13]. Other CNN studies analyse what layers attend to and report F1 scores roughly in the 0.8–0.9 range depending on architecture and data [14]. Broader reviews of multimodal explainable DeepFake detection argue that explanation quality should itself be evaluated, not assumed [15], [16]. Quantitative comparison work has even suggested that LIME-style region probes can be useful when validating which facial areas a detector depends on [16].

**Takeaway for practice:** CNN/EfficientNet detectors plus Grad-CAM currently offer a workable balance between accuracy, latency, and visual evidence for image authenticity services.

---

## 4. Fake News and Online Misinformation Detection

### 4.1 Scope and classical baselines

Fake news research covers fabricated articles, misleading headlines, rumours, and low-resource or multilingual variants. Older detectors used TF-IDF, linguistic counts, SVM, Random Forest, and stacking ensembles [4], [17]. These remain useful when data are small or when a transparent linear baseline is needed, but they struggle when the same claim is paraphrased.

### 4.2 BERT and transformer-based detectors

Contextual transformers changed the field. LSTM–BERT hybrids, dual-stream BERT–GNN models, graph-augmented ensembles, and transfer-learning pipelines repeatedly report accuracies in the mid-to-high 90s on popular English corpora [18], [19], [20]. Graph-aware designs try to encode author–claim or post–user structure when such signals exist. Data scarcity remains a practical limiter; LLM-based augmentation has been used to enlarge multilingual training sets, with caution that synthetic text can also inject bias [21].

### 4.3 Explainable text classification

BiLSTM-LIME style systems show that strong accuracy and local word explanations can coexist, with reported accuracy near 97% in one recent study [22]. Distilled BERT pipelines with SHAP explanations have been applied to COVID-19 claims and related health misinformation [23]. Design-science evaluations of explainable disinformation tools remind us that presenting an explanation does not always raise user trust; sometimes it increases doubt [24]. Therefore LIME and SHAP should be treated as decision support for analysts, not as automatic proof.

**Takeaway for practice:** BERT-family encoders are the default text backbone; LIME/SHAP are the most common local evidence layers for word-level justification.

---

## 5. Recruitment Scam and Employment Fraud Detection

### 5.1 Why this threat deserves separate attention

A fraudulent job post can look linguistically similar to spam or fake news, but the stakes differ. Victims may send fees, bank details, or identity documents under time pressure. Platforms that manage hiring in digital labour markets therefore need specialised fraud signals: unrealistic pay claims, vague company details, urgency language, unusual contact channels, and imbalance between legitimate and fake ads.

### 5.2 Machine learning and BERT-based detectors

Naudé et al. apply machine learning to fraudulent job-type detection and discuss social implications of algorithmic screening [25]. Fraud-BERT fine-tunes BERT on EMSCAD-style data where fake posts are a minority class, reporting high accuracy and strong AUC while handling imbalance better than older TF-IDF pipelines [6]. Outside pure computer science, criminology and sociological studies examine scam industries and victim pathways; they are useful for threat modelling even when they do not propose classifiers [26], [27].

### 5.3 Gaps relative to fake news research

Compared with fake news, labelled recruitment datasets are fewer, public benchmarks are less standardised, and explainability is rarely the main contribution. Many papers stop at accuracy. That leaves room for transferring BERT + LIME/SHAP pipelines from news to job ads, while adding domain heuristics that fake-news detectors do not need.

**Takeaway for practice:** Recruitment fraud is still a thin but high-impact area; text transformers and imbalance-aware training are currently the strongest algorithmic directions.

---

## 6. Explainable AI Techniques Relevant to Authenticity Systems

### 6.1 Grad-CAM for images

Grad-CAM aggregates gradients flowing into late convolutional maps to produce a coarse heatmap over the input. For face forgery, a useful heatmap should land on blending seams, eye/mouth inconsistency, or forged texture regions rather than on irrelevant background. EfficientNet and other CNNs integrate Grad-CAM with little extra cost at inference time, which is one reason the combination appears so often in forensic prototypes [8], [13], [14].

### 6.2 LIME and SHAP for text

LIME approximates the local decision boundary with an interpretable model around one sample. SHAP attributes prediction change to features using cooperative-game values. In practice both highlight tokens that push a label toward fake or real. They can disagree, and both can be gamed by correlated wording, so they should be logged as *supporting evidence*, not as ground truth [22], [23], [24].

### 6.3 Evidence-aware authenticity

We use the phrase *evidence-aware* for systems that return (i) a threat-specific label, (ii) a confidence score, and (iii) an explanation artefact usable by a human reviewer. That artefact may be a Grad-CAM overlay, a ranked word list, or both. Multi-threat frameworks become natural once each modality has its own detector and a shared evidence log.

---

## 7. Comparative Summary

### Table 1. Representative technique families

| Family | Typical task | Strength | Common weakness |
|--------|--------------|----------|-----------------|
| EfficientNet / CNN | DeepFake images | Strong accuracy–cost trade-off | Cross-dataset drop |
| Vision transformer / hybrid | DeepFake images/video | Captures long-range cues | Compute cost; data hunger |
| Classical ML (TF-IDF, SVM) | News / job text | Simple and inspectable | Weak on paraphrase |
| BERT / graph–BERT | Fake news, recruitment | Strong contextual modelling | Needs labelled data; opaque alone |
| Grad-CAM | Image explanation | Cheap visual evidence | Coarse localisation |
| LIME / SHAP | Text explanation | Local word rationale | Unstable / not causal proof |

### Table 2. Threat coverage in current literature

| Threat | Maturity of detectors | Maturity of XAI | Notes |
|--------|-----------------------|-----------------|-------|
| DeepFake images | High | Medium–high (Grad-CAM common) | Robustness still fragile |
| Fake news | High | Medium (LIME/SHAP rising) | Multimodal work growing |
| Recruitment scam | Low–medium | Low | Method transfer from fake news useful |

### Table 3. Alignment with an evidence-aware multi-threat design

| Module | Suggested detector | Suggested evidence | Supporting literature themes |
|--------|--------------------|--------------------|------------------------------|
| Image authenticity | EfficientNetB0 / EfficientNet family | Grad-CAM | [8], [9], [13], [14] |
| Fake news | BERT-style classifier | LIME / SHAP | [18], [22], [23] |
| Recruitment scam | BERT fine-tune + heuristics | LIME / SHAP | [6], [25] |

---

## 8. Open Challenges and Research Directions

**1. Cross-domain robustness.** Detectors trained on one forgery family or one news corpus often fail outside that distribution. Smart platforms receive noisy, compressed, multilingual content. Future surveys and systems should treat cross-domain evaluation as a first-class requirement.

**2. Explanation fidelity.** Heatmaps and token scores can look convincing while highlighting the wrong reason. Quantitative evaluation of explanations [16] deserves wider adoption.

**3. Multi-threat fusion.** Most papers optimise one task. Real incidents may mix a forged profile photo with a deceptive job description. Joint pipelines and shared audit logs remain rare.

**4. Recruitment-data scarcity.** Public job-fraud corpora are limited. Careful augmentation, semi-supervised learning, and privacy-preserving sharing of labels are needed.

**5. Human factors.** Adding XAI does not automatically increase trust [24]. Interface design and analyst workflow should be studied together with model metrics.

**6. Smart-service integration.** In communication systems and digital-city services, authenticity checks must run under latency, cost, and privacy constraints. Lightweight EfficientNet variants and selective explanation (explain only borderline cases) are practical research targets.

---

## 9. Conclusion

This survey reviewed DeepFake image detection, fake news classification, and recruitment scam detection through the lens of explainable authenticity verification. CNN and EfficientNet models remain strong for images when paired with Grad-CAM. BERT-family models dominate recent text work, and LIME/SHAP offer the most practical local explanations. Recruitment fraud still lags in volume of technical studies, yet it is operationally urgent. The main research opportunity is not another isolated high-accuracy model, but an evidence-aware multi-threat framework that connects detection scores to audit-ready explanations across image and text threats. Such frameworks are increasingly relevant for trustworthy communication platforms and safer digital city services.

---

## Acknowledgments

*[Optional: funding, lab, guide name. Declare any AI-assisted editing if the conference requires disclosure.]*

---

## References

[1] Y. Mirsky and W. Lee, “The Creation and Detection of Deepfakes: A Survey,” *ACM Computing Surveys*, vol. 54, no. 1, pp. 1–41, 2021, doi: 10.1145/3425780.

[2] B. C. Soundarya and H. L. Gururaj, “Deepfake detection: critical review of state-of-the-art approaches and future perspectives,” *Discover Applied Sciences*, vol. 8, no. 2, 2026, doi: 10.1007/s42452-025-08174-9.

[3] N. A. Bhat and K. J. Giri, “DeepFake detection in images and videos: a survey on models, datasets, and evaluation metrics,” *Multimedia Tools and Applications*, vol. 85, no. 3, 2026, doi: 10.1007/s11042-026-21365-9.

[4] J. Alghamdi, S. Luo, and Y. Lin, “A comprehensive survey on machine learning approaches for fake news detection,” *Multimedia Tools and Applications*, vol. 83, no. 17, pp. 51009–51067, 2023, doi: 10.1007/s11042-023-17470-8.

[5] F. A. Alshuwaier and F. A. Alsulaiman, “Fake News Detection Using Machine Learning and Deep Learning Algorithms: A Comprehensive Review and Future Perspectives,” *Computers*, vol. 14, no. 9, p. 394, 2025, doi: 10.3390/computers14090394.

[6] K. Taneja, J. Vashishtha, and S. Ratnoo, “Fraud-BERT: transformer based context aware online recruitment fraud detection,” *Discover Computing*, vol. 28, no. 1, 2025, doi: 10.1007/s10791-025-09502-8.

[7] B. V. Moyo, T. Tuyikeze, F. Matsebula, et al., “An AI-driven conceptual framework for detecting fake news and deepfake content: a systematic review,” *Frontiers in Artificial Intelligence*, vol. 9, 2026, doi: 10.3389/frai.2026.1737790.

[8] M. Kumar, A. Kumar, V. Yadav, et al., “A hybrid spatial–frequency attention-based algorithm using efficientnet for robust and interpretable deepfake detection,” *Scientific Reports*, vol. 16, no. 1, 2026, doi: 10.1038/s41598-026-46086-9.

[9] B. G. Deepa, C. K. Lokesh, D. Umamaheswari, et al., “An enhanced deep learning framework for DeepFake detection using EfficientNet-B3…,” *Discover Computing*, vol. 29, no. 1, 2026, doi: 10.1007/s10791-025-09890-x.

[10] D. Javiya, G. Jethava, H. Parmar, et al., “Enhancing deepfake detection with Adaptive-DCGAN and Lite-CNN…,” *Discover Applied Sciences*, vol. 7, no. 11, 2025, doi: 10.1007/s42452-025-07690-y.

[11] A. H. Soudy, O. Sayed, H. Tag-Elser, et al., “Deepfake detection using convolutional vision transformers and convolutional neural networks,” *Neural Computing and Applications*, vol. 36, no. 31, pp. 19759–19775, 2024, doi: 10.1007/s00521-024-10181-7.

[12] A. P. Magesh, S. S. M. Ramakrishnan, R. Arumuga Arun, et al., “Building an efficient Deep Fake detection system using… CNNs and transformers,” *Discover Computing*, vol. 28, no. 1, 2025, doi: 10.1007/s10791-025-09586-2.

[13] F. Castro, V. Gattulli, D. Impedovo, et al., “DeepDect: an explainable AI platform for face swapping and face generation DeepFake detection,” *Multimedia Tools and Applications*, vol. 85, no. 5, 2026, doi: 10.1007/s11042-026-21672-1.

[14] N. Mansoor and A. I. Iliev, “Explainable AI for DeepFake Detection,” *Applied Sciences*, vol. 15, no. 2, p. 725, 2025, doi: 10.3390/app15020725.

[15] M. S. Momin, A. Sufian, D. Barman, et al., “Explainable deepfake detection across different modalities…,” *Image and Vision Computing*, vol. 163, p. 105738, 2025, doi: 10.1016/j.imavis.2025.105738.

[16] K. Tsigos, E. Apostolidis, S. Baxevanakis, et al., “Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection,” in *MAD ’24*, 2024, pp. 37–45, doi: 10.1145/3643491.3660292.

[17] I. Ni’mah, R. Wijayanti, A. Santosa, et al., “A simple contrastive embedding framework for low-resource fake news detection,” *Neural Computing and Applications*, vol. 37, no. 26, pp. 21407–21433, 2025, doi: 10.1007/s00521-025-11467-0.

[18] H. Rama Moorthy, N. J. Avinash, N. S. Krishnaraj Rao, et al., “Dual stream graph augmented transformer model integrating BERT and GNNs…,” *Scientific Reports*, vol. 15, no. 1, 2025, doi: 10.1038/s41598-025-05586-w.

[19] C. Kumar, M. Bansal, M. A. Khan, et al., “Graph-augmented transformer ensemble framework for robust and scalable fake news detection…,” *Scientific Reports*, vol. 16, no. 1, 2025, doi: 10.1038/s41598-025-31653-3.

[20] B. S. Alqadi, S. A. Alsuhibany, S. N. Yousafzai, et al., “Transfer learning driven fake news detection… using large language models,” *Scientific Reports*, vol. 15, no. 1, 2025, doi: 10.1038/s41598-025-10670-2.

[21] R. Chalehchaleh, R. Farahbakhsh, and N. Crespi, “Addressing data scarcity in multilingual fake news detection…,” *Social Network Analysis and Mining*, vol. 15, no. 1, 2025, doi: 10.1007/s13278-025-01505-z.

[22] S. G. Sneha, A. Sen, S. Malik, et al., “BiLSTM-LIME: integrating NLP and advanced machine learning models for fake news detection,” *Discover Artificial Intelligence*, vol. 6, no. 1, 2026, doi: 10.1007/s44163-026-00852-w.

[23] X. Men and V. Y. Mariano, “Explainable Fake News Detection Based on BERT and SHAP Applied to COVID-19,” *IJMECS*, vol. 16, no. 1, pp. 11–22, 2024, doi: 10.5815/ijmecs.2024.01.02.

[24] I. Bezzaoui, C. Stein, C. Weinhardt, et al., “Explainable AI for online disinformation detection…,” *Electronic Markets*, vol. 35, no. 1, 2025, doi: 10.1007/s12525-025-00799-3.

[25] M. Naudé, K. J. Adebayo, and R. Nanda, “A machine learning approach to detecting fraudulent job types,” *AI & SOCIETY*, vol. 38, no. 2, pp. 1013–1024, 2023, doi: 10.1007/s00146-022-01469-0.

[26] I. Franceschini, C. Wyndham, L. Le, et al., “Imperfect Victims… Online Scam Industry,” *Asian Journal of Criminology*, vol. 21, no. 1, 2026, doi: 10.1007/s11417-026-09495-3.

[27] A. A. Asiama, “Pathways of Intention… Cyber Fraud Among Ghanaian Youth,” *Journal of Developmental and Life-Course Criminology*, vol. 12, no. 1, 2026, doi: 10.1007/s40865-026-00306-y.

---

## Submission checklist (SICACS / ICACS 2026)

1. Download the official **Manuscript Template** from [icacs-2026.charusat.ac.in](https://icacs-2026.charusat.ac.in/) and paste this content into it.  
2. Aim for **≥ 10 pages** in the template (expand tables / add 1 figure of taxonomy if needed).  
3. Remove Markdown headers; no embedded web links, headers, or footers in the camera-ready file (per conference guidance).  
4. Screen for plagiarism before CMT upload.  
5. Submit only through [CMT SICACS2026](https://cmt3.research.microsoft.com/SICACS2026/Submission/Index).  
6. Add **author names, affiliation, emails**, and (if required) AI-use disclosure.

---

## How to reply so I continue correctly

Send one line, for example:

- `Select Case A — finalise Word structure + add taxonomy figure`  
- `Select Case D — rewrite for smart cities theme`  
- `Select Case B — DeepFake-only survey, expand Grad-CAM section`  
- `Select Case C — text-only survey`  
- `Select Case E — comparison-focused survey`
