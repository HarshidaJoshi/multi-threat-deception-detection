# A Survey on Multi-Threat Explainable AI for Digital Deception Detection: DeepFakes, Fake News, and Recruitment Fraud

**Conference draft — Case A (selected)**  
**Target venue:** ICACS / SICACS 2026 (Springer LNNS proceedings)  
**Submit via:** Microsoft CMT — SICACS2026  
**Length target:** ≥ 10 pages after paste into Springer Word/LaTeX template  

**Authors:** Harshida Joshi$^{1}$ (Student, AI & Data Science; ORCID: 0009-0008-7301-4733), Swati Sharma$^{1}$ (Assistant Professor / Guide), Ashwini Kumar Jha$^{1}$ (Co-guide; ORCID: 0000-0002-6607-2168)  
**Affiliation:** $^{1}$Department of Artificial Intelligence & Data Science, Parul Institute of Engineering & Technology, Faculty of Engineering & Technology, Parul University, Waghodia, Vadodara, Gujarat, 391760, India  
**Email:** harshida.joshi@paruluniversity.ac.in, swati.sharma@paruluniversity.ac.in *(placeholders — replace with real IDs)*, ashwinibjha@gmail.com

> Paste this text into the official Springer manuscript template from icacs-2026.charusat.ac.in. Remove this instruction box. Do not keep Markdown `#` symbols or web hyperlinks in the final PDF. Insert Figure 1 from `docs/figures/sicacs2026_taxonomy_figure.png`.

---

## Abstract

Online deception rarely appears as a single neat problem. A synthetic face can sit on a profile page, a rewritten news claim can travel through messaging apps, and a false job post can pressure applicants into sharing documents or money. Research communities have studied each of these threats in depth, but deployed authenticity tools still tend to treat them as disconnected classifiers. This survey reviews recent machine learning and deep learning literature on DeepFake image detection, fake news classification, and recruitment scam detection, with particular attention to systems that expose *why* a decision was made. We organise the material around techniques that keep appearing in strong practical designs: EfficientNet-family convolutional backbones, Grad-CAM visual evidence, BERT-style text encoders, and LIME/SHAP local explanations. From the surveyed work we highlight three stubborn gaps—cross-dataset brittleness, explanation treated as an afterthought, and thin coverage of recruitment fraud—and we argue for an evidence-aware multi-threat view of authenticity verification that fits communication platforms and digital-city services.

**Keywords:** DeepFake detection · Fake news · Recruitment fraud · EfficientNet · Grad-CAM · BERT · LIME · SHAP · Explainable AI · Digital authenticity

---

## 1 Introduction

Digital platforms lowered the cost of publishing and raised the cost of verifying. With common tools, a user can post a face that never lived, recirculate a headline that sounds official, or advertise a job that exists only to harvest identity documents. End users experience these as different social situations—looking, reading, applying—yet each event moves trust from a weak digital cue toward a human action.

Historically the research response split along the same lines. DeepFake detection grew around convolutional networks trained on FaceForensics++, Celeb-DF, and related corpora, later joined by transformer hybrids. Fake news detection migrated from bag-of-words and TF–IDF baselines to contextual transformers such as BERT. Recruitment scam research arrived later, with fewer public datasets and fewer dedicated system papers. While reported accuracies climbed, another shortcoming became obvious in practice. A score such as “0.94 fake” rarely helps a moderator or applicant who must act under time pressure.

Explainable artificial intelligence (XAI) tries to shorten that gap. Grad-CAM overlays indicate which facial areas influenced an image model. LIME and SHAP allocate importance to words in a news item or job advert. Explanations are imperfect; they can be coarse, unstable, or misleading. Still, they turn a bare label into something closer to usable evidence. That is the central theme of this survey: *detection without evidence is incomplete for authenticity work*.

### 1.1 Contributions

This paper makes four contributions:

1. A multi-threat reading of the literature that places DeepFake, fake news, and recruitment scam research under one authenticity framing.  
2. A compact taxonomy that separates threat modality, learning family, explanation style, and deployment concern (Figure 1).  
3. Comparative tables that connect technique families to strengths, weaknesses, and an evidence-aware module design.  
4. A list of open problems that matter for communication systems and smart digital services, including robustness, explanation fidelity, and data scarcity in hiring fraud.

### 1.2 Paper organisation

Section 2 states how the literature was organised and reviews earlier surveys. Section 3 covers DeepFake image detection. Section 4 covers fake news. Section 5 covers recruitment fraud. Section 6 discusses Grad-CAM, LIME, and SHAP in authenticity settings. Section 7 summarises datasets and metrics. Section 8 presents comparative tables and a reference multi-threat pipeline. Section 9 lists open challenges. Section 10 concludes.

---

## 2 Survey Method, Related Reviews, and Taxonomy

### 2.1 Scope and selection logic

The survey is narrative rather than a full PRISMA-style meta-analysis. We privileged peer-reviewed journal and conference papers (roughly 2020–2026) that (i) propose or review detectors for one of the three threats, or (ii) study explainability for those detectors. Special weight was given to EfficientNet / CNN image work, BERT text work, and Grad-CAM / LIME / SHAP explanations, because these families currently offer a realistic accuracy–cost–interpretability balance for prototype platforms. Purely sociological scam studies appear where they clarify threat context, even when they do not introduce a classifier.

### 2.2 Earlier survey work

Mirsky and Lee provide a foundational account of DeepFake creation and detection [1]. Later reviews detail modern models, datasets, and evaluation pitfalls, including domain shift under new generators and compression [2], [3], [28]. Fake news surveys map content-based, propagation-based, source-based, and multimodal lines [4], [5], [29]. Cross-cutting reviews now place fake news and deepfakes in one social–technical frame [7]. Recruitment fraud still lacks a dense survey literature; technical progress is carried mainly by individual system papers such as Fraud-BERT [6] and related job-fraud classifiers [25].

### 2.3 Taxonomy

Figure 1 summarises the taxonomy used throughout the paper. Work is read along four axes:

- **Threat modality:** DeepFake image, fake news text, recruitment scam text.  
- **Learning family:** classical ML, CNN / EfficientNet, vision transformers / hybrids, BERT / graph–BERT.  
- **Explanation style:** none, Grad-CAM (visual), LIME / SHAP (local text), broader design explanations.  
- **Deployment concern:** in-domain accuracy, cross-domain robustness, class imbalance, human review cost.

**Fig. 1.** Taxonomy of multi-threat digital deception detection feeding evidence-aware authenticity verification.  
*(Insert image: `docs/figures/sicacs2026_taxonomy_figure.png`)*

This taxonomy is intentionally engineering-oriented. It asks what a platform builder should choose, not only how publishers group keywords.

---

## 3 DeepFake Image Detection

### 3.1 Threat setting

DeepFake systems synthesise or edit faces. Typical attacks include face swap, expression reenactment, attribute editing, and fully generated portraits. Early detectors searched for blending boundaries and texture irregularities. As generators improved, those cues weakened, and models had to learn subtler inconsistencies in colour, frequency response, and identity coherence.

### 3.2 CNN and EfficientNet approaches

Convolutional networks remain attractive for still-image screening because they are mature, relatively cheap at inference, and easy to couple with Grad-CAM. EfficientNet variants are common because compound scaling of depth, width, and resolution often yields a better accuracy–parameter trade-off than older heavy CNNs. Kumar et al. combine spatial and frequency cues around an EfficientNet backbone with attention and Grad-CAM, reporting very high AUC in controlled tests [8]. Deepa et al. compare classical texture pipelines (for example GLCM / HOG with SVM or Random Forest) against EfficientNet-B3 and InceptionV3; deep models win clearly, with EfficientNet-B3 near 98% accuracy in their setting [9]. Lite CNNs and synthetic-data expansion appear when labelled forgery sets are small [10]. Ensemble and multi-stream designs further try to reduce overfitting to one forgery style [30], [31].

For prototypes and edge-friendly services, EfficientNet-B0 is often preferred over heavier siblings. Larger variants may win leaderboard accuracy, but authenticity platforms also care about latency and the clarity of the accompanying heatmap.

### 3.3 Transformers and hybrid vision models

Vision transformers and CNN–transformer hybrids aim to capture long-range cues that local convolutions miss. Convolutional vision transformers and CSWin-style systems report high accuracy on FaceForensics++ / Celeb-DF-style splits [11], [12]. Survey comparisons place some hybrids in the high 90s for accuracy on standard partitions [3]. The repeated caution is cross-condition fragility: change the generator, the compression setting, or the identity distribution, and scores often fall [2], [3]. Social uploads rarely resemble a clean benchmark cut.

### 3.4 Explainable DeepFake detection

Operational review needs more than a probability. Platform-style systems attach Grad-CAM heatmaps and short notes for face-swap and face-generation cases [13]. Other CNN studies report F1 roughly in the 0.8–0.9 band depending on architecture and data while analysing internal decisions [14]. Reviews of multimodal explainable DeepFake detection argue that explanation quality must be measured, not assumed [15]. Quantitative frameworks that stress test explanation regions have found LIME competitive among several XAI methods for a studied detector [16].

**Section takeaway.** For image authenticity services, EfficientNet-family CNNs plus Grad-CAM currently offer one of the most practical balances of accuracy, cost, and visual evidence.

---

## 4 Fake News and Online Misinformation Detection

### 4.1 Scope and classical baselines

Fake news work covers fabricated articles, misleading headlines, rumours, health claims, and low-resource or multilingual settings. Early detectors used TF–IDF, linguistic markers, SVM, Random Forest, and stacking ensembles [4], [17]. These remain useful as transparent baselines or when labelled data are scarce, but they weaken when the same claim is paraphrased or translated.

### 4.2 BERT and transformer-based detectors

Contextual transformers reshaped the field. LSTM–BERT hybrids, dual-stream BERT–GNN models, graph-augmented ensembles, reinforced transformer designs, and transfer-learning pipelines report mid-to-high 90s accuracy on popular English corpora [18], [19], [20], [32]. Graph components help when interaction structure (user–post, claim–evidence) is available. Data scarcity still limits many languages; LLM-based augmentation can enlarge training sets, though synthetic text may inject its own bias [21]. Ensemble deep models further push in-domain performance [33].

### 4.3 Explainable text classification

High accuracy alone does not satisfy an analyst who must justify a flag. BiLSTM-LIME designs show that strong scores and local word highlighting can coexist, with reported accuracy near 97% in one recent study [22]. BERT-oriented pipelines with SHAP have been applied to COVID-19 related claims [23]. Design-science work on explainable disinformation tools warns that explanations do not automatically raise trust; they can also create hesitation [24]. LIME and SHAP are therefore best treated as decision support, not as automatic proof of intent.

**Section takeaway.** BERT-family encoders are the default text backbone; LIME and SHAP are the most common local evidence layers for word-level justification.

---

## 5 Recruitment Scam and Employment Fraud Detection

### 5.1 Why recruitment fraud is not just another spam task

A fake job can share lexical patterns with spam or fake news, but the user journey differs. Victims may pay fees, share bank details, or send identity documents under urgency. Useful signals include unrealistic salary promises, thin company identity, odd contact channels, pressure language, and severe class imbalance between genuine and fraudulent posts.

### 5.2 Learning-based detectors

Naudé et al. apply machine learning to fraudulent job-type detection and discuss social implications of algorithmic screening [25]. Fraud-BERT fine-tunes BERT on EMSCAD-style data where fake ads are a minority class, reporting high accuracy and strong AUC while handling imbalance better than older TF–IDF pipelines [6]. Criminology and developmental studies of scam labour markets help with threat modelling even when they do not train classifiers [26], [27].

### 5.3 Relative immaturity

Compared with fake news, recruitment fraud offers fewer public benchmarks, less standardised evaluation, and little systematic XAI. Method transfer is therefore attractive: reuse BERT + LIME/SHAP from news detection, then add hiring-specific heuristics and imbalance-aware training.

**Section takeaway.** Technical coverage is thin relative to impact; transformer fine-tuning plus explanation transfer from fake news is currently the most credible route.

---

## 6 Explainable AI for Evidence-Aware Authenticity

### 6.1 Grad-CAM for images

Grad-CAM aggregates class gradients into late convolutional feature maps to yield a coarse heatmap. For face forgery, useful heat should concentrate on blending seams, facial inconsistency, or anomalous texture rather than irrelevant background. CNNs such as EfficientNet integrate Grad-CAM with modest runtime cost [8], [13], [14].

### 6.2 LIME and SHAP for text

LIME fits a simple local surrogate around one sample. SHAP attributes prediction change to features under a cooperative-game interpretation. In practice both surface tokens that push a label toward fake or real. They can disagree with each other and can latch onto correlated wording, so systems should store them as *supporting evidence*, not as causal ground truth [22]–[24].

### 6.3 What “evidence-aware” means here

We call a system *evidence-aware* when each prediction returns: (i) a threat-specific label, (ii) a confidence score, and (iii) an explanation artefact that a human reviewer can inspect—Grad-CAM overlay, ranked token list, or both. Multi-threat frameworks become natural once each modality has a specialised detector and a shared audit log.

---

## 7 Datasets and Evaluation Practices

### 7.1 Common DeepFake resources

FaceForensics++, Celeb-DF, and related public sets dominate published image results [1]–[3]. Many papers also add private or freshly generated samples to test newer synthesis methods [9], [13]. Reporting only in-domain accuracy without cross-dataset checks overstates readiness for platform use.

### 7.2 Fake news and recruitment corpora

Fake news studies use multiple text and multimodal collections; survey papers document the fragmentation of definitions and labels [4], [5], [29]. Recruitment work often depends on EMSCAD-style job corpora or restricted institutional data [6], [25]. Imbalance is the normal case: fraudulent posts are fewer, so accuracy alone can mislead.

### 7.3 Metrics that matter

Accuracy, precision, recall, F1, and AUC appear almost everywhere. For authenticity platforms one should also track false-alarm rate under class skew, cross-dataset drop, latency for interactive review, and—when XAI is claimed—*whether the explanation changes reviewer decisions* [16], [24]. That last metric is still rare.

---

## 8 Comparative Summary and Reference Multi-Threat View

### Table 1. Technique families

| Family | Typical task | Main strength | Common weakness |
|---|---|---|---|
| EfficientNet / CNN | DeepFake images | Accuracy–cost trade-off | Cross-dataset drop |
| Vision transformer / hybrid | DeepFake image/video | Long-range cues | Compute and data hunger |
| Classical ML (TF–IDF, SVM) | News / job text | Simple, inspectable | Weak under paraphrase |
| BERT / graph–BERT | Fake news, recruitment | Contextual modelling | Opaque without XAI; needs labels |
| Grad-CAM | Image explanation | Cheap visual evidence | Coarse localisation |
| LIME / SHAP | Text explanation | Local word rationale | Unstable; not causal proof |

### Table 2. Literature maturity by threat

| Threat | Detector maturity | XAI maturity | Remarks |
|---|---|---|---|
| DeepFake images | High | Medium–high | Grad-CAM common; robustness fragile |
| Fake news | High | Medium | LIME/SHAP rising; multimodal growing |
| Recruitment scam | Low–medium | Low | Method transfer from fake news useful |

### Table 3. Evidence-aware module alignment

| Module | Detector suggestion | Evidence suggestion | Literature anchors |
|---|---|---|---|
| Image authenticity | EfficientNetB0 / EfficientNet family | Grad-CAM | [8], [9], [13], [14] |
| Fake news | BERT-style classifier | LIME / SHAP | [18], [22], [23] |
| Recruitment scam | BERT fine-tune + heuristics | LIME / SHAP | [6], [25] |

A practical multi-threat layout follows Table 3: route images to an EfficientNet detector with Grad-CAM; route news and job text to a shared BERT text stack with LIME/SHAP; write every decision and artefact into one evidence log for later audit. This is not claimed as a finished product architecture; it is a reading of where the literature already points.

---

## 9 Open Challenges and Research Directions

1. **Cross-domain robustness.** Models strong on one generator or one news corpus often weaken elsewhere. Platform content is compressed, multilingual, and noisy. Cross-domain reporting should be mandatory for claims of deployability.  
2. **Explanation fidelity.** Heatmaps and token ranks can look credible while reflecting the wrong reason. Quantitative stress tests of explanations [16] deserve wider use.  
3. **Multi-threat fusion.** Real incidents may combine a forged profile photograph with a deceptive job description. Joint pipelines and shared logs remain uncommon.  
4. **Recruitment data scarcity.** Public hiring-fraud corpora are limited. Semi-supervised learning, careful augmentation, and privacy-aware label sharing are needed.  
5. **Human factors.** Adding XAI does not guarantee higher trust [24]. Interface design and analyst workflow belong in the evaluation plan.  
6. **Smart-service constraints.** Communication systems and digital-city services face latency, cost, and privacy limits. Lightweight EfficientNet variants and *selective* explanation (explain borderline cases first) are practical research targets.

---

## 10 Conclusion

DeepFake, fake news, and recruitment fraud are usually studied apart, yet users meet them on the same platforms. This survey linked the three threats through an evidence-aware authenticity lens. EfficientNet-family CNNs with Grad-CAM form a strong image path. BERT-family models with LIME/SHAP form a strong text path for both news and job ads. Recruitment fraud remains thinner than the other two areas but is operationally urgent. The clearest research opportunity is not another isolated high-accuracy classifier; it is a multi-threat framework that returns labels, confidence, and inspectable evidence together. Such frameworks are increasingly relevant for trustworthy communication platforms and safer digital services.

---

## Acknowledgments

The authors thank [guide / lab / funding agency]. *[If required by the conference policy, disclose limited AI-assisted language editing here.]*

---

## References

[1] Y. Mirsky and W. Lee, “The Creation and Detection of Deepfakes: A Survey,” ACM Computing Surveys, vol. 54, no. 1, pp. 1–41, 2021, doi: 10.1145/3425780.

[2] B. C. Soundarya and H. L. Gururaj, “Deepfake detection: critical review of state-of-the-art approaches and future perspectives,” Discover Applied Sciences, vol. 8, no. 2, 2026, doi: 10.1007/s42452-025-08174-9.

[3] N. A. Bhat and K. J. Giri, “DeepFake detection in images and videos: a survey on models, datasets, and evaluation metrics,” Multimedia Tools and Applications, vol. 85, no. 3, 2026, doi: 10.1007/s11042-026-21365-9.

[4] J. Alghamdi, S. Luo, and Y. Lin, “A comprehensive survey on machine learning approaches for fake news detection,” Multimedia Tools and Applications, vol. 83, no. 17, pp. 51009–51067, 2023, doi: 10.1007/s11042-023-17470-8.

[5] F. A. Alshuwaier and F. A. Alsulaiman, “Fake News Detection Using Machine Learning and Deep Learning Algorithms: A Comprehensive Review and Future Perspectives,” Computers, vol. 14, no. 9, p. 394, 2025, doi: 10.3390/computers14090394.

[6] K. Taneja, J. Vashishtha, and S. Ratnoo, “Fraud-BERT: transformer based context aware online recruitment fraud detection,” Discover Computing, vol. 28, no. 1, 2025, doi: 10.1007/s10791-025-09502-8.

[7] B. V. Moyo, T. Tuyikeze, F. Matsebula, et al., “An AI-driven conceptual framework for detecting fake news and deepfake content: a systematic review,” Frontiers in Artificial Intelligence, vol. 9, 2026, doi: 10.3389/frai.2026.1737790.

[8] M. Kumar, A. Kumar, V. Yadav, et al., “A hybrid spatial–frequency attention-based algorithm using efficientnet for robust and interpretable deepfake detection,” Scientific Reports, vol. 16, no. 1, 2026, doi: 10.1038/s41598-026-46086-9.

[9] B. G. Deepa, C. K. Lokesh, D. Umamaheswari, et al., “An enhanced deep learning framework for DeepFake detection using EfficientNet-B3 comparative evaluation of deep and machine learning techniques,” Discover Computing, vol. 29, no. 1, 2026, doi: 10.1007/s10791-025-09890-x.

[10] D. Javiya, G. Jethava, H. Parmar, et al., “Enhancing deepfake detection with Adaptive-DCGAN and Lite-CNN: a novel approach to image classification,” Discover Applied Sciences, vol. 7, no. 11, 2025, doi: 10.1007/s42452-025-07690-y.

[11] A. H. Soudy, O. Sayed, H. Tag-Elser, et al., “Deepfake detection using convolutional vision transformers and convolutional neural networks,” Neural Computing and Applications, vol. 36, no. 31, pp. 19759–19775, 2024, doi: 10.1007/s00521-024-10181-7.

[12] A. P. Magesh, S. S. M. Ramakrishnan, R. Arumuga Arun, et al., “Building an efficient Deep Fake detection system using the recognition capabilities of convolutional neural networks and transformers,” Discover Computing, vol. 28, no. 1, 2025, doi: 10.1007/s10791-025-09586-2.

[13] F. Castro, V. Gattulli, D. Impedovo, et al., “DeepDect: an explainable AI platform for face swapping and face generation DeepFake detection,” Multimedia Tools and Applications, vol. 85, no. 5, 2026, doi: 10.1007/s11042-026-21672-1.

[14] N. Mansoor and A. I. Iliev, “Explainable AI for DeepFake Detection,” Applied Sciences, vol. 15, no. 2, p. 725, 2025, doi: 10.3390/app15020725.

[15] M. S. Momin, A. Sufian, D. Barman, et al., “Explainable deepfake detection across different modalities: An overview of methods and challenges,” Image and Vision Computing, vol. 163, p. 105738, 2025, doi: 10.1016/j.imavis.2025.105738.

[16] K. Tsigos, E. Apostolidis, S. Baxevanakis, et al., “Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection,” in Proc. MAD ’24, 2024, pp. 37–45, doi: 10.1145/3643491.3660292.

[17] I. Ni’mah, R. Wijayanti, A. Santosa, et al., “A simple contrastive embedding framework for low-resource fake news detection,” Neural Computing and Applications, vol. 37, no. 26, pp. 21407–21433, 2025, doi: 10.1007/s00521-025-11467-0.

[18] H. Rama Moorthy, N. J. Avinash, N. S. Krishnaraj Rao, et al., “Dual stream graph augmented transformer model integrating BERT and GNNs for context aware fake news detection,” Scientific Reports, vol. 15, no. 1, 2025, doi: 10.1038/s41598-025-05586-w.

[19] C. Kumar, M. Bansal, M. A. Khan, et al., “Graph-augmented transformer ensemble framework for robust and scalable fake news detection in social media ecosystems,” Scientific Reports, vol. 16, no. 1, 2025, doi: 10.1038/s41598-025-31653-3.

[20] B. S. Alqadi, S. A. Alsuhibany, S. N. Yousafzai, et al., “Transfer learning driven fake news detection and classification using large language models,” Scientific Reports, vol. 15, no. 1, 2025, doi: 10.1038/s41598-025-10670-2.

[21] R. Chalehchaleh, R. Farahbakhsh, and N. Crespi, “Addressing data scarcity in multilingual fake news detection: an LLM-based dataset augmentation approach,” Social Network Analysis and Mining, vol. 15, no. 1, 2025, doi: 10.1007/s13278-025-01505-z.

[22] S. G. Sneha, A. Sen, S. Malik, et al., “BiLSTM-LIME: integrating NLP and advanced machine learning models for fake news detection,” Discover Artificial Intelligence, vol. 6, no. 1, 2026, doi: 10.1007/s44163-026-00852-w.

[23] X. Men and V. Y. Mariano, “Explainable Fake News Detection Based on BERT and SHAP Applied to COVID-19,” International Journal of Modern Education and Computer Science, vol. 16, no. 1, pp. 11–22, 2024, doi: 10.5815/ijmecs.2024.01.02.

[24] I. Bezzaoui, C. Stein, C. Weinhardt, et al., “Explainable AI for online disinformation detection: Insights from a design science research project,” Electronic Markets, vol. 35, no. 1, 2025, doi: 10.1007/s12525-025-00799-3.

[25] M. Naudé, K. J. Adebayo, and R. Nanda, “A machine learning approach to detecting fraudulent job types,” AI & SOCIETY, vol. 38, no. 2, pp. 1013–1024, 2023, doi: 10.1007/s00146-022-01469-0.

[26] I. Franceschini, C. Wyndham, L. Le, et al., “Imperfect Victims: Comparing Chinese and Vietnamese Survivors of Forced Criminality in Southeast Asia’s Online Scam Industry,” Asian Journal of Criminology, vol. 21, no. 1, 2026, doi: 10.1007/s11417-026-09495-3.

[27] A. A. Asiama, “Pathways of Intention: Narratives of Future Withdrawal from Cyber Fraud Among Ghanaian Youth,” Journal of Developmental and Life-Course Criminology, vol. 12, no. 1, 2026, doi: 10.1007/s40865-026-00306-y.

[28] F. Abbas and A. Taeihagh, “Unmasking deepfakes: A systematic review of deepfake detection and generation techniques using artificial intelligence,” Expert Systems with Applications, vol. 252, p. 124260, 2024, doi: 10.1016/j.eswa.2024.124260.

[29] J. Lv, Y. Gao, L. Li, et al., “Multi-modal fake news detection: A comprehensive survey on deep learning technology, advances, and challenges,” Journal of King Saud University Computer and Information Sciences, vol. 37, no. 9, 2025, doi: 10.1007/s44443-025-00317-7.

[30] K. Omar, R. H. Sakr, and M. F. Alrahmawy, “An ensemble of CNNs with self-attention mechanism for DeepFake video detection,” Neural Computing and Applications, vol. 36, no. 6, pp. 2749–2765, 2024, doi: 10.1007/s00521-023-09196-3.

[31] H. Duan, Q. Jiang, X. Jin, et al., “Mf-net: multi-feature fusion network based on two-stream extraction and multi-scale enhancement for face forgery detection,” Complex & Intelligent Systems, vol. 11, no. 1, 2025, doi: 10.1007/s40747-024-01634-6.

[32] G. K. M, K. S. Faizz Ahmad, S. G. Pamidimukkala, et al., “Hybrid optimization driven fake news detection using reinforced transformer models,” Scientific Reports, vol. 15, no. 1, 2025, doi: 10.1038/s41598-025-99936-3.

[33] M. E. Almandouh, M. F. Alrahmawy, M. Eisa, et al., “Ensemble based high performance deep learning models for fake news detection,” Scientific Reports, vol. 14, no. 1, 2024, doi: 10.1038/s41598-024-76286-0.

---

## Author checklist before CMT upload

1. Download Springer template from the conference site and paste sections 1–10 + references.  
2. Insert **Fig. 1** from `docs/figures/sicacs2026_taxonomy_figure.png` (exported as a clean PNG/TIFF; not a screenshot with URL).  
3. Fill author names, ORCID if available, and affiliation.  
4. Expand lightly if page count is under 10 in the template (add one more comparison row table or dataset bullet list).  
5. Run plagiarism check; remove links/headers/footers from the PDF.  
6. Upload PDF only through CMT: https://cmt3.research.microsoft.com/SICACS2026/Submission/Index
