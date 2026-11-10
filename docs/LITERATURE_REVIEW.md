# Chapter 2: Literature Review

**Dissertation Title:** A Multi-Threat Explainable AI Framework for Digital Deception Detection and Evidence-Aware Authenticity Verification

**Literature base:** Papers collected from `f:\MTECH\Literature Review`, plus two added technique-matched papers [49]–[50] (50 papers in total)

**Link to present work:** Image module uses EfficientNetB0 with Grad-CAM. Text modules for fake news and recruitment scam use BERT-style classification with LIME and SHAP explanations.

---

## 2.1 Introduction

People now meet deception online in more than one shape. A face that never belonged to a person, a news post written to mislead readers, and a job advert designed to steal money or personal data may all appear on the same platform. Each of these harms trust differently: visual evidence can no longer be taken at face value, public discussion becomes noisy, and job seekers become easy targets. That is why this chapter does not review only one topic. It covers DeepFake image detection, fake news detection, and recruitment scam detection, and also looks at work that tries to make model decisions easier to understand.

Fifty papers were studied for this review after adding two further technique-matched works. Roughly twenty-two focus on DeepFake detection, twenty-three on fake news or related misinformation problems, five on recruitment fraud or online scam contexts, and one systematic review that discusses fake news and deepfakes together [1]. The writing is organised by theme rather than by listing papers one after another. Careful attention is given to EfficientNetB0, Grad-CAM, BERT, LIME, and SHAP, because these are the methods used in the present dissertation system.

---

## 2.2 DeepFake Image Detection

### 2.2.1 How the problem grew

DeepFake tools use generative models—mainly GAN-based methods in earlier years, and newer synthesis methods more recently—to create or alter faces. Survey papers by Mirsky and colleagues [21], critical reviews such as [9] and [24], and later overview studies explain common forms of attack: face swapping, expression transfer, attribute change, and fully generated faces. As the quality of fakes improved, ordinary visual checking became unreliable. Automated detection therefore became necessary.

Moyo and co-authors [1] surveyed the field using a structured review process and observed that research has moved from simple CNN detectors toward transformer-based and CLIP-style models. At the same time, they point out that cross-dataset testing, multimodal detection, and explainability are still weak. In other words, the area looks mature in terms of papers published, but it is still uneven when one asks what works outside a fixed laboratory setting.

### 2.2.2 CNN models and EfficientNet-based work

For still images, convolutional networks remain common and practical. Many authors try to combine colour-space features with frequency-domain signals, because compression often leaves traces that are easy to miss in raw pixels. Kumar et al. [3] build such a hybrid system around EfficientNet, add attention, and use Grad-CAM so that important regions can be shown to the user. They report a very high AUC (around 0.997). This kind of result supports choosing EfficientNet as a backbone when both accuracy and explanation are needed.

Deepa et al. [5] compare classical machine learning with deep networks on DeepFake images. They use SVM and Random Forest with texture features such as GLCM and HOG, and then compare these with EfficientNet-B3 and InceptionV3. Deep learning wins clearly. EfficientNet-B3 reaches about 97.95% accuracy and also performs well on later AI-generated samples. The present dissertation uses EfficientNetB0 instead of B3, mainly because B0 is lighter for a prototype. The important point from [5] is not the exact variant, but that the EfficientNet family is strong for this task.

A closely related explainable study evaluates XceptionNet, EfficientNet, and MesoNet on Celeb-DF v2 and an internal set of manipulated images, and then attaches Grad-CAM heatmaps so that the decision can be inspected visually [49]. EfficientNet performs among the stronger detectors in that comparison, which again supports keeping an EfficientNet backbone for the image module while insisting on Grad-CAM evidence.

Other groups try to make models more stable across datasets. Ensemble CoAtNet work [6], MF-Net [19], and texture-based dual-stream designs [20] all show that a single feature stream often overfits one style of forgery. Compact CNN work with synthetic data generation [13] and attention-based EfficientNet studies [18] also show that both data quality and model design continue to matter.

### 2.2.3 Transformers and hybrid vision models

Transformers were brought into DeepFake detection because CNNs mainly look at local neighbourhoods, while forgery cues may appear across a wider area of the face. Systems based on CSWin transformers [7] and convolutional vision transformers [11], as well as EfficientNet–Swin hybrids discussed in survey literature [10], report high scores on FaceForensics++ and Celeb-DF type data. Survey papers [10], [28] place leading results near 97.82% accuracy on FaceForensics++ in some comparisons.

Still, many surveys give the same warning: when the training set and the test set come from different forgery methods or different compression settings, accuracy often falls [9], [10], [19]. That warning is important for this dissertation, because uploaded images from real users will not match a clean benchmark exactly.

### 2.2.4 Explainable DeepFake detection

A detection score by itself is not enough when someone must defend a decision or understand a false alarm. DeepDect [8] therefore builds a platform that can handle both face-swap and fully generated faces, and attaches Grad-CAM heatmaps plus short text notes. Another study [14] trains ResNet-50, InceptionV3, and VGG-16 and then analyses what the network has learned. Their F1-scores sit roughly between 0.8 and 0.9. Reviews on explainable DeepFake detection [15] and work on measuring XAI quality [22] argue that explanations should also be checked carefully, not accepted blindly.

In this dissertation, Grad-CAM is used with EfficientNetB0 for the same reason: the user should see which facial areas influenced the prediction. Similar thinking appears in [3], [8], and [15].

### 2.2.5 Short summary of DeepFake work

Overall, CNN and EfficientNet models remain useful for images. Hybrid spatial–frequency designs help under compression. Transformers can raise accuracy, but they are heavier. Grad-CAM has become an expected part of forensic discussion. The open problem is less about inventing one more detector and more about placing an image detector inside a wider authenticity system that can also handle text threats.

---

## 2.3 Fake News and Misinformation Detection

### 2.3.1 Background and surveys

Fake news spreads quickly through social platforms and can disturb politics, health advice, and public debate. Survey papers [25], [27], [36], and [39] describe how researchers define the problem and how they group methods: content-based, propagation-based, source-based, and multimodal. Alghamdi and Luo [25] give a broad map of machine learning and deep learning solutions. Multimodal surveys [39] remind us that today’s false posts are often mixtures of text, image, and sometimes video.

Moyo et al. [1] discuss fake news and deepfakes in one place and argue that technical methods cannot be separated from social and governance questions. That view fits the present dissertation, because the system is designed as a multi-threat framework rather than a single classifier.

### 2.3.2 Classical and early learning methods

Many older systems still use TF-IDF, bag-of-words, SVM, Random Forest, and stacking ensembles [2], [27], [29]. Contrastive embedding work for low-resource settings [29] shows that careful representation learning can raise precision when fake and real classes are unbalanced. Classical methods are useful when data are small, but they struggle when the same false claim is rewritten in different words.

### 2.3.3 BERT and transformer-based detection

Recent papers mostly rely on BERT, RoBERTa, or similar transformers. Examples include LSTM–BERT hybrids [30], graph-augmented ensembles [32], [37], reinforcement-style transformer designs [38], and transfer learning setups [43]. Dual-stream BERT–GNN models [32] report very high accuracy, near 99%, with AUC near 0.985. These numbers show how powerful contextual text models can be when the dataset is rich enough.

Data shortage remains a real obstacle. Chalehchaleh et al. [4] use Llama 3 to create synthetic news samples and then train BERT classifiers. Performance improves in few-shot settings, but synthetic samples may also introduce new bias if they are not checked carefully. Ensemble studies [33] and multimodal reviews [35] also show that model choice depends heavily on data size and content type.

### 2.3.4 LIME, SHAP, and explainable text detection

High accuracy is not enough if no one can see why a news item was marked fake. BiLSTM-LIME [31] reports about 97.21% accuracy and an F1 near 0.97 while highlighting important words with LIME. That paper is valuable because it shows that transparency and performance can exist together. Men and Mariano [50] follow a related path for COVID-19 claims: BERT-based distillation strengthens BiLSTM classification, and SHAP is then used to show which words pushed the decision toward fake or real. A design-science study of explainable disinformation tools [34] is more cautious. Adding explanations does not always increase trust. Sometimes users become less sure after reading an explanation. So LIME and SHAP should be designed as decision support, not as decoration.

Other explainable multimodal studies [26], [42] ask for clearer reasons behind predictions, especially in languages with fewer labelled corpora. These findings support the use of LIME and SHAP in the fake news and recruitment text modules of the present work.

### 2.3.5 Short summary of fake news work

The field has moved from word counts to contextual transformers. BERT-based models are now the usual choice for strong accuracy. Explainable methods such as LIME and SHAP are important for trust. Remaining difficulties include class imbalance, language diversity, multimodal fusion, and careful explanation design. The present system follows this direction by using BERT-style classification with LIME and SHAP attribution.

---

## 2.4 Recruitment Scam and Online Job Fraud

### 2.4.1 Limited technical literature

Recruitment scam papers are fewer than DeepFake or fake news papers. In the present collection, only Fraud-BERT [46] and the fraudulent job-type study [44] are mainly technical. The other three papers [45], [47], [48] are social or criminological studies. They help in understanding why scams succeed, but they do not propose detection models. This thin technical base is itself a research gap.

### 2.4.2 Computational approaches

Naudé et al. [44] try to classify types of fraudulent job posts, such as identity theft and multi-level marketing schemes. They compare rule-based features, bag-of-words, embeddings, and transformer features. A Gradient Boosting model with mixed features reaches an F1 near 0.88. Transformer features do better than simple lexical rules, which supports the use of contextual language models for job text.

Taneja et al. [46] present Fraud-BERT. They fine-tune BERT on EMSCAD, a public set of real and fake job ads where fake posts are the minority. Reported results are about 99% accuracy, F1 near 0.93, and AUC near 0.97. The paper shows that BERT handles imbalance better than older TF-IDF pipelines. Among all recruitment papers reviewed here, Fraud-BERT is the closest match to the present dissertation’s recruitment module.

### 2.4.3 Wider social evidence

Research on Ponzi victims [45], scam-compound recruitment [47], and youth involved in cyber fraud [48] shows that fake job offers are often the first contact point for larger online crime. These papers do not give algorithms, but they justify why a detection module for recruitment posts is socially important.

### 2.4.4 Short summary of recruitment work

Algorithmic recruitment-fraud research is still young. Fraud-BERT [46] and fraud-type classification [44] give useful technical direction, mainly around BERT and contextual features. Explainable multi-threat systems remain rare. This dissertation therefore applies the same text analysis and explanation approach used for fake news to fraudulent job posts.

---

## 2.5 Techniques Used in the Present Framework

### 2.5.1 EfficientNetB0

EfficientNet models scale network depth, width, and input size together, which usually gives good accuracy for fewer parameters. Previous DeepFake studies that use EfficientNet-B0 or EfficientNet-B3 [3], [5], [10] support choosing B0 for a research system that must remain practical. The goal is a solid image classifier, not the heaviest possible network.

### 2.5.2 Grad-CAM

Grad-CAM produces heatmaps of the regions that most influenced a CNN decision. DeepFake explainability papers use it to point toward eyes, mouth edges, blending lines, and other suspicious areas [3], [8], [14], [15]. In the present system, Grad-CAM serves as visual evidence for authenticity checking.

### 2.5.3 BERT

BERT encodes text using both left and right context. Fake news papers [31], [32], [43] and Fraud-BERT [46] show that this style of model works well for deceptive writing, including unbalanced scam data. That is why BERT-oriented classification is adopted for text modules.

### 2.5.4 LIME and SHAP

LIME explains a single prediction by highlighting local token importance [31], [34]. SHAP estimates how each feature contributed to the output [15], [34]. Used together, they turn a text score into evidence that a user can read. This is required both for fake news and for recruitment scam analysis.

---

## 2.6 Comparative Analysis

The tables below compare important papers by technique, focus, reported results, and practical weaknesses. Metrics are taken as reported in the reviewed sources; where a paper is a survey or qualitative study, outcome cells are noted accordingly.

### Table 2.1 — DeepFake Image Detection (selected papers)

| Paper Title | Ref No | Techniques | Objective / Focus | Outcomes (metrics) | Difficulties | Limitations | Benefits |
|-------------|--------|------------|-------------------|--------------------|--------------|-------------|----------|
| Enhanced DL framework using EfficientNet-B3 | [5] | EfficientNet-B3, InceptionV3, SVM, RF, GLCM/HOG | Compare ML and DL for DeepFake image classification | Acc about 97.95% (EfficientNet-B3) | Result changes with dataset size; weak handcrafted features | Mainly one dataset; still images only | Strong support for EfficientNet family |
| Hybrid spatial–frequency EfficientNet detector | [3] | EfficientNet, DCT, attention, Grad-CAM | Robust and explainable image detection | AUC about 0.997 | Harder fusion of spatial and frequency cues | Cross-generator testing still limited | Good accuracy with Grad-CAM evidence |
| DeepDect XAI platform | [8] | ResNet-50, CapsNet, CNN, Grad-CAM | Platform for face-swap and generated faces | Engine chosen through benchmarking | Several models to maintain | Not EfficientNet-only | Useful real-world design with heatmaps |
| Explainable AI for DeepFake Detection | [14] | ResNet, Inception, VGG | Interpretable forged-image detection | F1 about 0.8 to 0.9 | Cost of training multiple CNNs | Older backbones | Connects accuracy with interpretability |
| DeepFake Detection survey (models/datasets) | [10] | CNN, GAN, ViT, EfficientNetB0 | Compare models, datasets, and metrics | Acc up to 97.82% on FF++ | Hard to compare uneven benchmarks | Survey only | Helps select models and datasets |
| Critical review of SOTA approaches | [9] | GAN-related generation analysis | Review methods and open issues | Qualitative synthesis | Generators change quickly | No new model proposed | Clear picture of remaining challenges |
| Ensemble CNN with self-attention (CoAtNet) | [6] | CoAtNet, bagging | DeepFake video face detection | Strong ensemble behaviour reported | Video pipeline is heavier | Mainly video-focused | Local and global cues together |
| Hybrid CNN forged facial detection | [17] | LBP, Gabor, CNN, ResNet-152 | Detect morphing and face swap | ACER 2.82% on DEFACTO | Texture feature design effort | May need retuning for other data | Low error on morphing attacks |
| MF-Net multi-feature fusion | [19] | Two-stream CNN, multi-scale fusion | Improve cross-dataset face forgery detection | High in-dataset AUC | Cross-dataset drop still exists | Complex design | Richer feature combination |
| CSWin transformer detector | [7] | CNN + CSWin transformer | Efficient DeepFake image detection | Strong F1 reported | Higher transformer cost | Needs careful tuning | Captures both local and long-range cues |
| DeepFake detection with CNN models and XAI | [49] | EfficientNet, XceptionNet, MesoNet, Grad-CAM | Accurate and interpretable DeepFake detection | EfficientNet and XceptionNet strong on Celeb-DF v2 | Multiple CNNs to compare | Internal + public data mix | Direct support for EfficientNet + Grad-CAM design |

### Table 2.2 — Fake News / Misinformation Detection (selected papers)

| Paper Title | Ref No | Techniques | Objective / Focus | Outcomes (metrics) | Difficulties | Limitations | Benefits |
|-------------|--------|------------|-------------------|--------------------|--------------|-------------|----------|
| BiLSTM-LIME interpretable detector | [31] | BiLSTM, GloVe, LIME | Accurate and transparent fake news detection | Acc 97.21%; F1 about 0.97 | Training and explanation pipeline | Mostly English data | Strong accuracy with word-level reasons |
| Dual-stream BERT + GNN | [32] | BERT, GNN, LIME | Context-aware fake news detection | Acc about 99%; AUC about 0.985 | Graph building cost | Depends on graph quality | Combines meaning with network context |
| LLM-based multilingual augmentation | [4] | Llama 3, BERT | Reduce lack of labelled news data | Better F1 in few-shot tests | Synthetic samples may add bias | Quality of generated text varies | Useful when real labels are scarce |
| Comprehensive ML survey for FND | [25] | ML/DL taxonomy | Map definitions and methods | Survey discussion | Definitions keep changing | Not an implemented system | Good theoretical base |
| Explainable disinformation (design science) | [34] | LIME, SHAP, Grad-CAM | Study XAI tools for user trust | Mixed trust results in user study | Explanations can confuse users | Adoption depends on interface | Shows XAI must be tested with people |
| Multimodal fake news survey | [39] | BERT, RoBERTa, multimodal DL | Review text–image–video fake news | Survey discussion | Fusion across modalities is hard | No one best model | Directions for multimodal systems |
| Contrastive embedding (low-resource) | [29] | Contrast-BERT, ensembles | Fake news detection with little data | Precision improved by about 26.64% | Heavily imbalanced data | Language-specific setup | Helpful for skewed real data |
| Ensemble DL fake news models | [33] | BERT, RoBERTa, ensembles | Raise performance through ensembles | Acc about 96.73% | Slower inference | Harder to maintain many models | More stable accuracy |
| Towards Smart FND through XAI | [42] | BERT, CNN, LSTM | Move toward explainable multimodal detection | Acc about 90% reported | Modality alignment problems | Still early research | Argues for explanation-ready systems |
| Automatic fake news review | [27] | BERT, LLM, ML/DL overview | Review the progress of automatic detection | Some older baselines around 65% Acc | LLMs can both create and detect fakes | Field changes quickly | Shows challenges of the LLM period |
| Explainable fake news detection using BERT and SHAP | [50] | BERT distillation, BiLSTM, SHAP | Accurate and explainable COVID-19 fake news detection | Improved BiLSTM with BERT distillation; SHAP explanations | COVID-focused domain | Text-only setup | Strong match for BERT + SHAP text module |

### Table 2.3 — Recruitment Scam / Job Fraud

| Paper Title | Ref No | Techniques | Objective / Focus | Outcomes (metrics) | Difficulties | Limitations | Benefits |
|-------------|--------|------------|-------------------|--------------------|--------------|-------------|----------|
| Fraud-BERT ORF detection | [46] | BERT, Bi-LSTM baselines, TF-IDF | Detect fake job posts on EMSCAD | Acc about 99%; F1 about 0.93; AUC about 0.97 | Very unbalanced classes | BERT needs more compute | Closest technical match to present work |
| ML approach to fraudulent job types | [44] | Embeddings, transformers, Gradient Boosting, LIME | Classify types of fraudulent job ads | F1 about 0.88 | Labelling many fraud types | Not a full web platform | Interpretable fraud-type clues |
| Ponzi victim motivations (India) | [45] | Qualitative study | Understand why victims fall for schemes | Not a detection metric | No algorithm proposed | Social science only | Explains social harm |
| Imperfect Victims (forced cyber fraud) | [47] | Qualitative / legal analysis | Study recruitment into scam compounds | Not a detection metric | Victims often misunderstood | Not technical | Links job fraud to forced crime |
| Pathways of Intention (Ghana cyber fraud) | [48] | Qualitative interviews | Study plans to leave cyber fraud | Not a detection metric | Not predictive | No ML model | Broader view of fraud economies |

### Table 2.4 — Technique alignment with the present dissertation

| Technique | Representative refs | Role in literature | Role in present prototype |
|-----------|---------------------|--------------------|---------------------------|
| EfficientNetB0 | [3], [5], [10], [49] | Efficient CNN backbone for DeepFake images | Main image classifier |
| Grad-CAM | [3], [8], [14], [15], [49] | Spatial explanation for CNN decisions | Visual evidence on predicted images |
| BERT | [31], [32], [46], [50] | Contextual text classification | Scoring of news and job text |
| LIME | [31], [34] | Local token explanation | Word-level text evidence |
| SHAP | [15], [34], [50] | Feature contribution analysis | Feature importance view for text |

---

## 2.7 Research Gaps

After reading these papers, several gaps become clear.

First, DeepFake, fake news, and recruitment fraud are usually studied as separate topics [1], [9], [25], [39]. Very few systems put image and text deception inside one authenticity workflow.

Second, explanation is still treated as an add-on. Many models report high accuracy but give little help to the user [14], [26], [42]. Grad-CAM, LIME, and SHAP appear often in papers, yet they are not always built into complete working systems [34].

Third, recruitment scam detection has a small technical base. Fraud-BERT [46] and fraud-type classification [44] are useful, but work on explainable job-post screening is still limited.

Fourth, models often perform well on the dataset they were trained for and then weaken on new generators, new compression, or new languages [4], [10], [19], [29].

Fifth, large transformer systems may give better scores, but they also need more time and memory [7], [32]. EfficientNetB0 remains a practical choice for image work in a research prototype [5], [10].

Sixth, many systems stop at a label and a confidence value. Fewer systems force that label to come with visual or textual evidence [8], [31], [34].

These gaps shape the present dissertation. The proposed system joins EfficientNetB0 and Grad-CAM for images with BERT-oriented text analysis and LIME/SHAP explanations for fake news and recruitment scam content, inside one multi-threat authenticity framework.

---

## 2.8 Chapter Summary

This chapter reviewed fifty papers linked to DeepFake images, fake news, and recruitment scams. Image research supports CNN and EfficientNet models and shows rising interest in Grad-CAM. Fake news research strongly supports BERT-style models and explainable methods such as LIME and SHAP. Recruitment scam research is smaller, but Fraud-BERT and related machine learning work give a clear direction for applying the same text pipeline to fake job posts. The comparative tables summarise techniques, reported results, and remaining weak points. The next chapter uses these findings to present the methodology of the proposed framework.

---

## References

[1] B. V. Moyo, T. Tuyikeze, F. Matsebula, et al. (2026) An AI-driven conceptual framework for detecting fake news and deepfake content: a systematic review. *Frontiers in Artificial Intelligence*, 9. doi: https://doi.org/10.3389/frai.2026.1737790

[2] R. K. Ayyasamy, C. Ponnusamy, K. N. Bhargavi, et al. (2025) A hybrid deep learning framework for fake news detection using LSTM-CGPNN and metaheuristic optimization. *Scientific Reports*, 15(1). doi: https://doi.org/10.1038/s41598-025-25311-x

[3] M. Kumar, A. Kumar, V. Yadav, et al. (2026) A hybrid spatial–frequency attention-based algorithm using efficientnet for robust and interpretable deepfake detection. *Scientific Reports*, 16(1). doi: https://doi.org/10.1038/s41598-026-46086-9

[4] R. Chalehchaleh, R. Farahbakhsh, and N. Crespi (2025) Addressing data scarcity in multilingual fake news detection: an LLM-based dataset augmentation approach. *Social Network Analysis and Mining*, 15(1). doi: https://doi.org/10.1007/s13278-025-01505-z

[5] B. G. Deepa, C. K. Lokesh, D. Umamaheswari, et al. (2026) An enhanced deep learning framework for DeepFake detection using EfficientNet-B3 comparative evaluation of deep and machine learning techniques. *Discover Computing*, 29(1). doi: https://doi.org/10.1007/s10791-025-09890-x

[6] K. Omar, R. H. Sakr, and M. F. Alrahmawy (2024) An ensemble of CNNs with self-attention mechanism for DeepFake video detection. *Neural Computing and Applications*, 36(6), 2749–2765. doi: https://doi.org/10.1007/s00521-023-09196-3

[7] A. P. Magesh, S. S. M. Ramakrishnan, R. Arumuga Arun, et al. (2025) Building an efficient Deep Fake detection system using the recognition capabilities of convolutional neural networks and transformers. *Discover Computing*, 28(1). doi: https://doi.org/10.1007/s10791-025-09586-2

[8] F. Castro, V. Gattulli, D. Impedovo, et al. (2026) DeepDect: an explainable AI platform for face swapping and face generation DeepFake detection. *Multimedia Tools and Applications*, 85(5). doi: https://doi.org/10.1007/s11042-026-21672-1

[9] B. C. Soundarya and H. L. Gururaj (2026) Deepfake detection: critical review of state-of-the-art approaches and future perspectives. *Discover Applied Sciences*, 8(2). doi: https://doi.org/10.1007/s42452-025-08174-9

[10] N. A. Bhat and K. J. Giri (2026) DeepFake detection in images and videos: a survey on models, datasets, and evaluation metrics. *Multimedia Tools and Applications*, 85(3). doi: https://doi.org/10.1007/s11042-026-21365-9

[11] A. H. Soudy, O. Sayed, H. Tag-Elser, et al. (2024) Deepfake detection using convolutional vision transformers and convolutional neural networks. *Neural Computing and Applications*, 36(31), 19759–19775. doi: https://doi.org/10.1007/s00521-024-10181-7

[12] O. Pontorno, L. Guarnera, and S. Battiato (2025) DeepFeatureX-SN: Generalization of deepfake detection via contrastive learning. *Multimedia Tools and Applications*, 84(39), 47721–47740. doi: https://doi.org/10.1007/s11042-025-21060-1

[13] D. Javiya, G. Jethava, H. Parmar, et al. (2025) Enhancing deepfake detection with Adaptive-DCGAN and Lite-CNN: a novel approach to image classification. *Discover Applied Sciences*, 7(11). doi: https://doi.org/10.1007/s42452-025-07690-y

[14] N. Mansoor and A. I. Iliev (2025) Explainable AI for DeepFake Detection. *Applied Sciences*, 15(2), 725. doi: https://doi.org/10.3390/app15020725

[15] M. S. Momin, A. Sufian, D. Barman, et al. (2025) Explainable deepfake detection across different modalities: An overview of methods and challenges. *Image and Vision Computing*, 163, 105738. doi: https://doi.org/10.1016/j.imavis.2025.105738

[16] K. Taha (2026) Generative AI for multimodal content: a survey with empirical and experimental evaluations. *Artificial Intelligence Review*, 59(6). doi: https://doi.org/10.1007/s10462-026-11525-6

[17] S. Kaman, S. B. Javeriya, and A. Makandar (2025) Hybrid CNN-based detection of forged facial images using Gabor filters and local binary patterns. *Discover Computing*, 28(1). doi: https://doi.org/10.1007/s10791-025-09731-x

[18] Y. Sheng, Z. Zou, Z. Yu, et al. (2025) ID-insensitive deepfake detection model based on multi-attention mechanism. *Scientific Reports*, 15(1). doi: https://doi.org/10.1038/s41598-025-96254-6

[19] H. Duan, Q. Jiang, X. Jin, et al. (2025) Mf-net: multi-feature fusion network based on two-stream extraction and multi-scale enhancement for face forgery detection. *Complex & Intelligent Systems*, 11(1). doi: https://doi.org/10.1007/s40747-024-01634-6

[20] R. Gong, R. He, D. Zhang, et al. (2025) Robust face forgery detection integrating local texture and global texture information. *EURASIP Journal on Information Security*, 2025(1). doi: https://doi.org/10.1186/s13635-025-00189-4

[21] Y. Mirsky and W. Lee (2021) The Creation and Detection of Deepfakes: A Survey. *ACM Computing Surveys*, 54(1), 1–41. doi: https://doi.org/10.1145/3425780

[22] K. Tsigos, E. Apostolidis, S. Baxevanakis, et al. (2024) Towards Quantitative Evaluation of Explainable AI Methods for Deepfake Detection. In *Proceedings of the 3rd ACM International Workshop on Multimedia AI against Disinformation (MAD ’24)*, pp. 37–45. doi: https://doi.org/10.1145/3643491.3660292

[23] T. T. Aurpa (2026) Transparent AI for mathematics: transformer-based large language models for mathematical entity relationship extraction with XAI. *Scientific Reports*, 16(1). doi: https://doi.org/10.1038/s41598-026-43507-7

[24] F. Abbas and A. Taeihagh (2024) Unmasking deepfakes: A systematic review of deepfake detection and generation techniques using artificial intelligence. *Expert Systems with Applications*, 252, 124260. doi: https://doi.org/10.1016/j.eswa.2024.124260

[25] J. Alghamdi, S. Luo, and Y. Lin (2023) A comprehensive survey on machine learning approaches for fake news detection. *Multimedia Tools and Applications*, 83(17), 51009–51067. doi: https://doi.org/10.1007/s11042-023-17470-8

[26] H. R. LekshmiAmmal and A. K. Madasamy (2025) A reasoning based explainable multimodal fake news detection for low resource language using large language models and transformers. *Journal of Big Data*, 12(1). doi: https://doi.org/10.1186/s40537-025-01093-x

[27] R. Ș. Emil and B. Remus (2025) A Review of Automatic Fake News Detection: From Traditional Methods to Large Language Models. *Future Internet*, 17(10), 435. doi: https://doi.org/10.3390/fi17100435

[28] L. A. Passos, D. Jodas, K. A. P. Costa, et al. (2024) A review of deep learning‐based approaches for deepfake content detection. *Expert Systems*, 41(8). doi: https://doi.org/10.1111/exsy.13570

[29] I. Ni’mah, R. Wijayanti, A. Santosa, et al. (2025) A simple contrastive embedding framework for low-resource fake news detection. *Neural Computing and Applications*, 37(26), 21407–21433. doi: https://doi.org/10.1007/s00521-025-11467-0

[30] S. Muthukumar, J. G. Fernandez, B. J. Ambika, et al. (2026) AI Based LSTM-BERT Model for False News Detection. In *Data Science and Exploration in Artificial Intelligence (CODE-AI 2025)*, *Communications in Computer and Information Science*, vol. 2689, pp. 124–134. Springer. doi: https://doi.org/10.1007/978-3-032-19318-6_12

[31] S. G. Sneha, A. Sen, S. Malik, et al. (2026) BiLSTM-LIME: integrating NLP and advanced machine learning models for fake news detection. *Discover Artificial Intelligence*, 6(1). doi: https://doi.org/10.1007/s44163-026-00852-w

[32] H. Rama Moorthy, N. J. Avinash, N. S. Krishnaraj Rao, et al. (2025) Dual stream graph augmented transformer model integrating BERT and GNNs for context aware fake news detection. *Scientific Reports*, 15(1). doi: https://doi.org/10.1038/s41598-025-05586-w

[33] M. E. Almandouh, M. F. Alrahmawy, M. Eisa, et al. (2024) Ensemble based high performance deep learning models for fake news detection. *Scientific Reports*, 14(1). doi: https://doi.org/10.1038/s41598-024-76286-0

[34] I. Bezzaoui, C. Stein, C. Weinhardt, et al. (2025) Explainable AI for online disinformation detection: Insights from a design science research project. *Electronic Markets*, 35(1). doi: https://doi.org/10.1007/s12525-025-00799-3

[35] S. Yakkundi, R. Patil, S. Sangani, et al. (2025) Exploring machine learning for fake news detection: techniques, tools, challenges, and future research directions. *Discover Applied Sciences*, 7(8). doi: https://doi.org/10.1007/s42452-025-07548-3

[36] F. A. Alshuwaier and F. A. Alsulaiman (2025) Fake News Detection Using Machine Learning and Deep Learning Algorithms: A Comprehensive Review and Future Perspectives. *Computers*, 14(9), 394. doi: https://doi.org/10.3390/computers14090394

[37] C. Kumar, M. Bansal, M. A. Khan, et al. (2025) Graph-augmented transformer ensemble framework for robust and scalable fake news detection in social media ecosystems. *Scientific Reports*, 16(1). doi: https://doi.org/10.1038/s41598-025-31653-3

[38] G. K. M, K. S. Faizz Ahmad, S. G. Pamidimukkala, et al. (2025) Hybrid optimization driven fake news detection using reinforced transformer models. *Scientific Reports*, 15(1). doi: https://doi.org/10.1038/s41598-025-99936-3

[39] J. Lv, Y. Gao, L. Li, et al. (2025) Multi-modal fake news detection: A comprehensive survey on deep learning technology, advances, and challenges. *Journal of King Saud University Computer and Information Sciences*, 37(9). doi: https://doi.org/10.1007/s44443-025-00317-7

[40] A. D. Geto, E. D. Emiru, N. E. Seid, et al. (2025) Multimodal based Amharic fake news detection using CNN and attention-based BiLSTM. *Scientific Reports*, 15(1). doi: https://doi.org/10.1038/s41598-025-17579-w

[41] Y. Zhang, X. Ma, C. Song, et al. (2026) Multimodal graph contrastive learning for fake news video detection. *Journal of King Saud University Computer and Information Sciences*, 38(1). doi: https://doi.org/10.1007/s44443-025-00400-z

[42] A. B. Athira, S. D. Madhu Kumar, and A. M. Chacko (2022) Towards Smart Fake News Detection Through Explainable AI. arXiv:2207.11490. doi: https://doi.org/10.48550/arXiv.2207.11490

[43] B. S. Alqadi, S. A. Alsuhibany, S. N. Yousafzai, et al. (2025) Transfer learning driven fake news detection and classification using large language models. *Scientific Reports*, 15(1). doi: https://doi.org/10.1038/s41598-025-10670-2

[44] M. Naudé, K. J. Adebayo, and R. Nanda (2023) A machine learning approach to detecting fraudulent job types. *AI & SOCIETY*, 38(2), 1013–1024. doi: https://doi.org/10.1007/s00146-022-01469-0

[45] N. Samanta (2025) Analysing motivations of Ponzi victims in West Bengal, India. *Humanities and Social Sciences Communications*, 12(1). doi: https://doi.org/10.1057/s41599-025-04737-8

[46] K. Taneja, J. Vashishtha, and S. Ratnoo (2025) Fraud-BERT: transformer based context aware online recruitment fraud detection. *Discover Computing*, 28(1). doi: https://doi.org/10.1007/s10791-025-09502-8

[47] I. Franceschini, C. Wyndham, L. Le, et al. (2026) Imperfect Victims: Comparing Chinese and Vietnamese Survivors of Forced Criminality in Southeast Asia’s Online Scam Industry. *Asian Journal of Criminology*, 21(1). doi: https://doi.org/10.1007/s11417-026-09495-3

[48] A. A. Asiama (2026) Pathways of Intention: Narratives of Future Withdrawal from Cyber Fraud Among Ghanaian Youth. *Journal of Developmental and Life-Course Criminology*, 12(1). doi: https://doi.org/10.1007/s40865-026-00306-y

[49] DeepFake Detection Using CNN Models with Explainable AI Techniques (2025). *IEEE Access*. doi: https://doi.org/10.1109/ACCESS.2025.11135663

[50] X. Men and V. Y. Mariano (2024) Explainable Fake News Detection Based on BERT and SHAP Applied to COVID-19. *International Journal of Modern Education and Computer Science*, 16(1), 11–22. doi: https://doi.org/10.5815/ijmecs.2024.01.02
