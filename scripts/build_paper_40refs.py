"""Build paper.tex: revised prose + 40 refs from literature_refs_formatted.txt."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FMT = (ROOT / "docs" / "literature_refs_formatted.txt").read_text(encoding="utf-8")
OUT = ROOT / "docs" / "sicacs2026_llncs" / "paper.tex"
NOTE = ROOT / "docs" / "sicacs2026_llncs" / "SELECTED_40_REFS.txt"

# 40 tech-aligned refs for multi-threat XAI survey (skip 10 peripheral/social/off-topic)
KEEP = [
    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22,
    24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 46, 49, 50,
]
assert len(KEEP) == 40, len(KEEP)
SKIPPED = sorted(set(range(1, 51)) - set(KEEP))

raw = {}
for line in FMT.splitlines():
    m = re.match(r"\[(\d+)\]\s+(.*)", line.strip())
    if m:
        raw[int(m.group(1))] = m.group(2)

lr_to_new = {lr: i for i, lr in enumerate(KEEP, 1)}


def C(*lrs):
    keys = [f"r{lr_to_new[lr]:02d}" for lr in lrs]
    return "\\cite{" + ",".join(keys) + "}"


def to_lncs(n: int, text: str) -> str:
    t = text.replace("*", "")
    doi_m = re.search(r"10\.\d{4,9}/\S+", t)
    doi = doi_m.group(0).rstrip(".,;)") if doi_m else ""
    t = re.sub(r"\s*doi:\s*https://doi\.org/\S+", "", t).strip().rstrip(".")
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s+(.*)$", t)
    if not m:
        authors, year, rest = "Anon.", "2024", t
    else:
        authors, year, rest = m.group(1), m.group(2), m.group(3)
    authors = authors.replace(" and ", ", ").rstrip(".") + ":"
    rest = rest.strip().rstrip(".")
    line = f"{authors} {rest} ({year})."
    if doi:
        line += f" \\doi{{{doi}}}"
    return f"\\bibitem{{r{n:02d}}}\n{line}\n"


bib = "\n".join(to_lncs(n, raw[lr]) for n, lr in enumerate(KEEP, 1))

# Plain template; replace <<Cn>> markers
body = r"""% SICACS / ICACS 2026 survey paper in Springer LLNCS format
% 40 selected references from docs/literature_refs_formatted.txt
% Compile: pdflatex paper.tex
%
\documentclass[runningheads]{llncs}
%
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
%
\begin{document}
%
\title{A Survey on Multi-Threat Explainable AI for Digital Deception Detection: DeepFakes, Fake News, and Recruitment Fraud}
%
\titlerunning{Multi-Threat Explainable AI for Digital Deception Detection}
%
\author{Harshida Joshi\inst{1}\orcidID{0009-0008-7301-4733} \and
Swati Sharma\inst{1} \and
Ashwini Kumar Jha\inst{1}\orcidID{0000-0002-6607-2168}}
%
\authorrunning{H. Joshi et al.}
%
\institute{Department of Artificial Intelligence \& Data Science,
Parul Institute of Engineering \& Technology,
Faculty of Engineering \& Technology,
Parul University, Waghodia, Vadodara, Gujarat, 391760, India\\
Harshida Joshi: Student, Artificial Intelligence \& Data Science\\
Swati Sharma: Assistant Professor (Guide)\\
Ashwini Kumar Jha: Co-Guide\\
\email{harshidhijoshi@gmail.com, swati.sharma@paruluniversity.ac.in, ashwinibjha@gmail.com}}
%
\maketitle
%
\begin{abstract}
Online deception rarely appears as a single, neat problem. A synthetic face can sit on a profile page, a rewritten news claim can travel through messaging apps, and a false job post can pressure applicants into sharing documents or money. Research communities have studied each of these threats in depth, but deployed authenticity tools still tend to treat them as separate classifiers. This survey reviews recent machine learning and deep learning literature on deepfake image detection, fake news classification, and recruitment scam detection, with particular attention to systems that expose why a decision was made. We organize the material around techniques that keep appearing in strong practical designs: EfficientNet-family convolutional backbones, Grad-CAM visual evidence, BERT-style text encoders, and LIME/SHAP local explanations. From the surveyed work, we highlight three stubborn gaps---cross-dataset brittleness, explanation treated as an afterthought, and thin coverage of recruitment fraud---and we argue for an evidence-aware multi-threat view of authenticity verification that fits communication platforms and digital-city services.

\keywords{DeepFake detection \and Fake news \and Recruitment fraud \and EfficientNet \and Grad-CAM \and BERT \and LIME \and SHAP \and Explainable AI \and Digital authenticity}
\end{abstract}
%
\section{Introduction}
Digital platforms lower the cost of publishing and raise the cost of verification. With common tools, a user can post a face that never lived, recirculate a headline that sounds official, or advertise a job that exists only to harvest identity documents. End users experience these as different social situations---looking, reading, applying---yet each event moves trust from a weak digital cue toward a human action.

Historically, the research response has been split along the same lines. DeepFake detection grew around convolutional networks trained on FaceForensics++, Celeb-DF, and related corpora, later joined by transformer hybrids. Fake news detection has migrated from bag-of-words and TF--IDF baselines to contextual transformers such as BERT. Recruitment scam research arrived later, with fewer public datasets and fewer dedicated system studies. Although the reported accuracies increased, another shortcoming became evident in practice. A score such as ``0.94 fake'' rarely helps a moderator or applicant who must act under time constraints.

Explainable artificial intelligence (XAI) tries to bridge this gap. The Grad-CAM overlays indicate which facial areas influenced the image model. LIME and SHAP allocate importance to words in news items or job advertisements. Explanations are imperfect; they can be coarse, unstable or misleading. However, they turn a bare label into something closer to usable evidence. This is the central theme of this survey: detection without evidence is incomplete for authenticity work.

\subsection{Contributions}
This study makes four contributions.
\begin{enumerate}
\item A multi-threat reading of the literature that places DeepFake, fake news, and recruitment scam research under one authenticity frame.
\item A compact taxonomy that separates threat modality, learning family, explanation style, and deployment concern (Fig.~\ref{fig1}).
\item Comparative tables connecting technique families to strengths, weaknesses, and evidence-aware module design.
\item A list of open problems that matter for communication systems and smart digital services, including robustness, explanation fidelity, and data scarcity in hiring fraud.
\end{enumerate}

\subsection{Paper Organisation}
Section~\ref{sec:method} states how the literature is organized and reviews earlier surveys. Section~\ref{sec:deepfake} covers the detection of DeepFake images. Section~\ref{sec:fakenews} covers fake news. Section~\ref{sec:recruit} discusses recruitment fraud. Section~\ref{sec:xai} discusses Grad-CAM, LIME, and SHAP in authenticity settings. Section~\ref{sec:data} summarizes the datasets and metrics. Section~\ref{sec:compare} presents comparative tables and a reference multi-threat pipeline. Section~\ref{sec:open} lists the open challenges. Section~\ref{sec:concl} concludes.

\section{Survey Method, Related Reviews, and Taxonomy}
\label{sec:method}

\subsection{Scope and Selection Logic}
The survey is narrative rather than a full PRISMA-style meta-analysis. We privileged peer-reviewed journal and conference papers (roughly 2020--2026) that (i)~propose or review detectors for one of the three threats, or (ii)~study the explainability of those detectors. Special weight was given to EfficientNet/CNN image work, BERT text work, and Grad-CAM/LIME/SHAP explanations because these families currently offer a realistic accuracy--cost--interpretability balance for prototype platforms. From the conducted literature collection we retain \textbf{40} technique-aligned works for this conference survey. Purely sociological scam studies are used only where they clarify threat context.

\subsection{Earlier Survey Work}
Mirsky and Lee provided a foundational account of DeepFake creation and detection~<<C21>>. Later reviews have detailed modern models, datasets, and evaluation pitfalls, including domain shifts under new generators and compression~<<C9,10,24,28>>. Fake news surveys map content-based, propagation-based, source-based, and multimodal lines~<<C25,36,39>>. Cross-cutting reviews now place fake news and deepfakes in one socio-technical frame~<<C1>>. Recruitment fraud still lacks a dense survey literature; technical progress is mainly carried by system papers such as Fraud-BERT~<<C46>> and related job-fraud classifiers~<<C44>>.

\subsection{Taxonomy}
Figure~\ref{fig1} summarizes the taxonomy used in this study. The work is read along four axes:
\begin{itemize}
\item \textbf{Threat modality:} DeepFake images, fake news text, and recruitment scam text.
\item \textbf{Learning family:} classical ML, CNN/EfficientNet, vision transformers/hybrids, BERT/graph--BERT.
\item \textbf{Explanation style:} none, Grad-CAM (visual), LIME/SHAP (local text), and broader design explanations.
\item \textbf{Deployment concerns:} in-domain accuracy, cross-domain robustness, class imbalance, and human review costs.
\end{itemize}

\begin{figure}
\includegraphics[width=\textwidth]{fig1.png}
\caption{Taxonomy of multi-threat digital deception detection feeding evidence-aware authenticity verification.}
\label{fig1}
\end{figure}

\noindent This taxonomy is intentionally engineering-oriented. It asks what a platform builder should choose, not only how publishers group the keywords.

\section{DeepFake Image Detection}
\label{sec:deepfake}

\subsection{Threat Setting}
DeepFake systems synthesize or edit facial images. Typical attacks include face swapping, expression reenactment, attribute editing, and fully generated portraits. Early detectors searched for blending boundaries and texture irregularities. As generators improved, these cues weakened, and models had to learn subtler inconsistencies in color, frequency response, and identity coherence.

\subsection{CNN and EfficientNet Approaches}
Convolutional networks remain attractive for still-image screening because they are mature, relatively inexpensive at inference, and easy to couple with Grad-CAM. EfficientNet variants are common because the compound scaling of depth, width, and resolution often yields a better accuracy--parameter trade-off than older heavy CNNs. Kumar et al.\ combine spatial and frequency cues around an EfficientNet backbone with attention and Grad-CAM, reporting very high AUC in controlled tests~<<C3>>. Deepa et al.\ compared classical texture pipelines against EfficientNet-B3 and InceptionV3; deep models win clearly, with EfficientNet-B3 near 98\% accuracy in their setting~<<C5>>. Lite CNNs and synthetic-data expansion appear when labelled forgery sets are small~<<C13>>. Ensemble and multi-stream designs further attempt to reduce overfitting to one forgery style~<<C6,19>>. CNN--Grad-CAM comparisons across EfficientNet-class backbones further support explainable image screening~<<C49>>.

For prototypes and edge-friendly services, EfficientNet-B0 is often preferred over its heavier counterparts. Larger variants may win leaderboard accuracy, but authenticity platforms also care about latency and the clarity of the accompanying heatmap.

\subsection{Transformers and Hybrid Vision Models}
Vision transformers and CNN--transformer hybrids aim to capture long-range cues that local convolutional layers miss. Convolutional vision transformers and CSWin-style systems report high accuracy on FaceForensics++/Celeb-DF-style splits~<<C11,7>>. Survey comparisons place some hybrids in the high 90s for accuracy on standard partitions~<<C10>>. Contrastive and generalization-oriented DeepFake work also stresses robustness beyond a single forgery family~<<C12,18>>. Repeated caution is cross-condition fragility: change the generator, compression setting, or identity distribution, and the scores often fall~<<C9,10>>. Social uploads rarely resemble clean benchmark cuts.

\subsection{Explainable DeepFake Detection}
An operational review requires more than a probability. Platform-style systems attach Grad-CAM heatmaps and short notes for face-swap and face-generation cases~<<C8>>. Other CNN studies reported F1 scores of approximately 0.8--0.9, depending on the architecture and data, while analyzing internal decisions~<<C14>>. Reviews of multimodal explainable DeepFake detection argue that explanation quality must be measured, not assumed~<<C15>>. Hybrid forged-face CNN work and quantitative XAI stress tests further motivate careful explanation validation~<<C17,22>>.

\paragraph{Section takeaway.}
For image authenticity services, EfficientNet-family CNNs plus Grad-CAM currently offer one of the most practical balances of accuracy, cost, and visual evidence.

\section{Fake News and Online Misinformation Detection}
\label{sec:fakenews}

\subsection{Scope and Classical Baselines}
Fake news work covers fabricated articles, misleading headlines, rumors, health claims, and low-resource or multilingual settings. Early detectors used TF--IDF, linguistic markers, SVM, Random Forest, and stacking ensembles~<<C25,29>>. These remain useful as transparent baselines or when labelled data are scarce, but they weaken when the same claim is paraphrased or translated.

\subsection{BERT and Transformer-Based Detectors}
Contextual transformers have reshaped this field. LSTM--BERT hybrids, dual-stream BERT--GNN models, graph-augmented ensembles, reinforced transformer designs, and transfer-learning pipelines report mid-to-high 90s accuracy on popular English corpora~<<C30,32,37,38,43>>. Graph components are useful when interaction structure is available. Data scarcity still limits many languages; LLM-based augmentation can enlarge training sets, although synthetic text may inject its own bias~<<C4>>. Ensemble deep models further improve in-domain performance~<<C33>>. Broader machine-learning surveys continue to map technique and tooling choices~<<C35>>.

\subsection{Explainable Text Classification}
High accuracy alone does not satisfy an analyst who must justify flags. BiLSTM-LIME designs show that strong scores and local word highlighting can coexist, with reported accuracy near 97\% in one recent study~<<C31>>. BERT-oriented pipelines with SHAP have been applied to COVID-19 related claims~<<C50>>. Reasoning-based multimodal explainable detectors and design-science evaluations of disinformation tools warn that explanations do not automatically raise trust; they can also create hesitancy~<<C26,34,42>>. Therefore, LIME and SHAP are best treated as decision support, not as automatic proof of intent.

\paragraph{Section takeaway.}
BERT-family encoders are the default text backbone, and LIME and SHAP are the most common local evidence layers for word-level justification.

\section{Recruitment Scam and Employment Fraud Detection}
\label{sec:recruit}

\subsection{Why Recruitment Fraud Is Not Just Another Spam Task}
A fake job can share lexical patterns with spam or fake news, but the user journey differs. Victims may pay fees, share bank details, or send identity documents urgently. Useful signals include unrealistic salary promises, vague company identity, odd contact channels, pressure language, and severe class imbalance between genuine and fraudulent posts.

\subsection{Learning-Based Detectors}
Naud\'{e} et al.\ apply machine learning to fraudulent job-type detection and discuss social implications of algorithmic screening~<<C44>>. Fraud-BERT fine-tunes BERT on EMSCAD-style data, where fake ads are a minority class, reporting high accuracy and strong AUC while handling imbalance better than older TF--IDF pipelines~<<C46>>.

\subsection{Relative Immaturity}
Compared with fake news, recruitment fraud offers fewer public benchmarks, less standardized evaluation, and little systematic XAI. Therefore, method transfer is attractive: reuse BERT + LIME/SHAP from news detection, then add hiring-specific heuristics and imbalance-aware training.

\paragraph{Section takeaway.}
Technical coverage is thin relative to impact; transformer fine-tuning plus explanation transfer from fake news is currently the most credible route.

\section{Explainable AI for Evidence-Aware Authenticity}
\label{sec:xai}

\subsection{Grad-CAM for Images}
Grad-CAM aggregates class gradients into late convolutional feature maps to generate a coarse heatmap. For face forgery, useful heat should concentrate on blending seams, facial inconsistency, or anomalous texture rather than irrelevant background. CNNs such as EfficientNet integrate Grad-CAM with modest runtime cost~<<C3,8,14,49>>.

\subsection{LIME and SHAP for Text}
LIME fits a simple local surrogate around a sample. SHAP attributes the prediction change to features under a cooperative-game interpretation. In practice, both surface tokens that push a label toward fake or real. They can disagree with each other and latch onto correlated wording; therefore, systems should store them as supporting evidence, not as causal ground truth~<<C31,50,34>>.

\subsection{What Evidence-Aware Means Here}
We call a system evidence-aware when each prediction returns (i)~a threat-specific label, (ii)~a confidence score, and (iii)~an explanation artifact that a human reviewer can inspect---a Grad-CAM overlay, ranked token list, or both. Multi-threat frameworks become natural once each modality has a specialized detector and a shared audit log.

\section{Datasets and Evaluation Practices}
\label{sec:data}

\subsection{Common DeepFake Resources}
FaceForensics++, Celeb-DF, and related public sets dominate published image results~<<C21,9,10>>. Many studies also add private or freshly generated samples to test newer synthesis methods~<<C5,8>>. Reporting only in-domain accuracy without cross-dataset checks overstates readiness for platform use.

\subsection{Fake News and Recruitment Corpora}
Fake news studies use multiple text and multimodal collections, and survey papers document the fragmentation of definitions and labels~<<C25,36,39>>. Recruitment work often depends on EMSCAD-style job corpora or restricted institutional data~<<C46,44>>. Imbalance is the normal case: fraudulent posts are fewer, so accuracy alone can mislead.

\subsection{Metrics That Matter}
Accuracy, precision, recall, F1, and AUC appear almost everywhere. For authenticity platforms, one should also track the false-alarm rate under class skew, cross-dataset drop, latency for interactive review, and---when XAI is claimed---whether the explanation changes reviewer decisions~<<C22,34>>. The last metric is still rare.

\section{Comparative Summary and Reference Multi-Threat View}
\label{sec:compare}
Table~\ref{tab1} summarizes the technique families. Table~\ref{tab2} compares literature maturity by threat. Table~\ref{tab3} aligns modules with detectors and evidence methods.

\begin{table}
\caption{Representative technique families for digital deception detection.}
\label{tab1}
\begin{tabular}{|p{2.4cm}|p{2.3cm}|p{2.6cm}|p{2.6cm}|}
\hline
Family & Typical task & Main strength & Common weakness\\
\hline
EfficientNet / CNN & DeepFake images & Accuracy--cost trade-off & Cross-dataset drop\\
\hline
Vision transformer / hybrid & DeepFake image/video & Long-range cues & Compute and data hunger\\
\hline
Classical ML (TF--IDF, SVM) & News / job text & Simple, inspectable & Weak under paraphrase\\
\hline
BERT / graph--BERT & Fake news, recruitment & Contextual modelling & Opaque without XAI\\
\hline
Grad-CAM & Image explanation & Cheap visual evidence & Coarse localisation\\
\hline
LIME / SHAP & Text explanation & Local word rationale & Unstable; not causal proof\\
\hline
\end{tabular}
\end{table}

\begin{table}
\caption{Literature maturity by threat type.}
\label{tab2}
\begin{tabular}{|l|l|l|p{3.6cm}|}
\hline
Threat & Detector maturity & XAI maturity & Remarks\\
\hline
DeepFake images & High & Medium--high & Grad-CAM common; robustness fragile\\
\hline
Fake news & High & Medium & LIME/SHAP rising; multimodal growing\\
\hline
Recruitment scam & Low--medium & Low & Method transfer from fake news useful\\
\hline
\end{tabular}
\end{table}

\begin{table}
\caption{Evidence-aware module alignment suggested by the surveyed literature.}
\label{tab3}
\begin{tabular}{|p{2.3cm}|p{3.1cm}|p{2.4cm}|p{2.6cm}|}
\hline
Module & Detector suggestion & Evidence & Literature\\
\hline
Image authenticity & EfficientNetB0 / EfficientNet family & Grad-CAM & <<C3,5,8,14,49>>\\
\hline
Fake news & BERT-style classifier & LIME / SHAP & <<C32,31,50>>\\
\hline
Recruitment scam & BERT fine-tune + heuristics & LIME / SHAP & <<C46,44>>\\
\hline
\end{tabular}
\end{table}

\noindent A practical multi-threat layout follows Table~\ref{tab3}: route images to an EfficientNet detector with Grad-CAM; route news and job text to a shared BERT text stack with LIME/SHAP; write every decision and artifact into one evidence log for later auditing. This is not claimed as a finished product architecture; it is a reading of where the literature already points.

\section{Open Challenges and Research Directions}
\label{sec:open}
\begin{enumerate}
\item \textbf{Cross-domain robustness.} Models strong on one generator or one news corpus often weaken elsewhere. Platform content is compressed, multilingual, and noisy. Cross-domain reporting should be mandatory for claims of deployability.
\item \textbf{Explanation fidelity.} Heatmaps and token ranks can appear credible while reflecting the wrong reasons. Quantitative stress tests of explanations~<<C22>> deserve wider application.
\item \textbf{Multi-threat fusion.} Real incidents may combine a forged profile photograph with a deceptive job description. Joint pipelines and shared logs are still uncommon.
\item \textbf{Recruitment data scarcity.} Public hiring-fraud corpora are limited. Semi-supervised learning, careful augmentation, and privacy-aware label sharing are required.
\item \textbf{Human factors.} Adding XAI does not guarantee higher trust~<<C34>>. Interface design and analyst workflow belong in the evaluation plan.
\item \textbf{Smart-service constraints.} Communication systems and digital-city services face latency, cost, and privacy limits. Lightweight EfficientNet variants and selective explanations are practical research targets.
\end{enumerate}

\section{Conclusion}
\label{sec:concl}
DeepFake, fake news, and recruitment fraud are usually studied separately, yet users encounter them on the same platforms. This survey linked the three threats through an evidence-aware authenticity lens. EfficientNet-family CNNs with Grad-CAM form a strong image pathway. BERT-family models with LIME/SHAP form a strong text path for both news and job advertisements. Recruitment fraud remains less prevalent than the other two areas but is operationally urgent. The clearest research opportunity is not another isolated high-accuracy classifier but a multi-threat framework that returns labels, confidence, and inspectable evidence together. Such frameworks are increasingly relevant for trustworthy communication platforms and safer digital services.

\begin{credits}
\subsubsection{\ackname}
The authors thank the Department of Artificial Intelligence \& Data Science, Parul Institute of Engineering \& Technology, Parul University, for support during this survey work.

\subsubsection{\discintname}
The authors have no competing interests to declare that are relevant to the content of this article.
\end{credits}
%
\begin{thebibliography}{99}

<<BIB>>
\end{thebibliography}
\end{document}
"""


def expand_cite(match: re.Match) -> str:
    nums = [int(x) for x in match.group(1).split(",")]
    return C(*nums)


body = re.sub(r"<<C([0-9,]+)>>", expand_cite, body)
body = body.replace("<<BIB>>", bib)
OUT.write_text(body, encoding="utf-8")

lines = ["SELECTED 40 references for paper.tex", "Source: docs/literature_refs_formatted.txt", ""]
lines.append("Paper# | LR# | Citation start")
lines.append("-" * 80)
for n, lr in enumerate(KEEP, 1):
    lines.append(f"{n:02d} | [{lr:02d}] | {raw[lr][:100]}")
lines.append("")
lines.append("SKIPPED (10): " + ", ".join(f"[{s}]" for s in SKIPPED))
lines.append("Skipped titles: Ayyasamy LSTM-CGPNN; Taha generative multimodal; Gong texture; Aurpa math XAI;")
lines.append("Emil fake-news review; Geto Amharic; Zhang video GCL; Samanta Ponzi; Franceschini; Asiama.")
NOTE.write_text("\n".join(lines), encoding="utf-8")

print("OK", OUT)
print("bibitems", body.count("\\bibitem{"))
print("skipped", SKIPPED)
