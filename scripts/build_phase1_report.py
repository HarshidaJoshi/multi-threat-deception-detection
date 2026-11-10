"""Generate fuller Phase-1 dissertation draft (Ch.1–3 summary) as Word."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

TITLE = (
    "A Multi-Threat Explainable AI Framework for Digital Deception Detection "
    "and Evidence-Aware Authenticity Verification"
)


def font(run, size=12, bold=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    font(r, 14, True)


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    font(r, 12, True)


def body(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    r = p.add_run(text)
    font(r, 12, False)


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.clear()
    r = p.add_run(text)
    font(r, 12, False)


def main():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1)
        s.bottom_margin = Inches(1)
        s.left_margin = Inches(1.25)
        s.right_margin = Inches(1)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("M.TECH DISSERTATION – PHASE 1 REPORT")
    font(r, 14, True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE)
    font(r, 12, True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        "Joshi Harshida Mukeshbhai (2503032340019)\n"
        "Guide: Dr. Swati Sharma | Co-Guide: Dr. Ashwini Kumar Jha\n"
        "M.Tech (Artificial Intelligence & Data Science), Parul University"
    )
    font(r, 11, False)

    h1(doc, "Chapter 1: Introduction")
    h2(doc, "1.1 Background")
    body(
        doc,
        "Digital platforms have lowered the cost of publishing and raised the cost of verification. "
        "With common tools, a user can post a face that never lived, recirculate a headline that sounds "
        "official, or advertise a job that exists only to harvest identity documents. These threats "
        "look different to end users—looking, reading, applying—yet each event moves trust from a weak "
        "digital cue toward a human action.",
    )
    body(
        doc,
        "DeepFake detection research grew around convolutional networks and later transformers. Fake news "
        "detection moved from bag-of-words and TF–IDF baselines to BERT-style contextual models. "
        "Recruitment scam research arrived later and still has fewer dedicated system papers. While "
        "reported accuracies climbed, a practical shortcoming remains: a score such as “0.94 fake” rarely "
        "helps a moderator or applicant who must act under time pressure.",
    )

    h2(doc, "1.2 Problem Statement")
    body(
        doc,
        "Most authenticity tools still treat DeepFake images, fake news, and recruitment fraud as "
        "separate pipelines and often behave as black boxes. There is a need for a unified, evidence-aware "
        "framework that detects multiple deception threats and explains why a decision was made.",
    )

    h2(doc, "1.3 Research Objectives")
    bullet(doc, "Design a multi-threat explainable AI framework for digital deception detection.")
    bullet(doc, "Detect DeepFake images using EfficientNetB0 with Grad-CAM visual evidence.")
    bullet(doc, "Detect fake news and recruitment scam text using BERT-style models with LIME/SHAP.")
    bullet(doc, "Implement a Phase-1 web prototype that returns label, confidence, and explanation.")
    bullet(doc, "Evaluate performance using standard classification metrics and explanation usefulness.")

    h2(doc, "1.4 Scope")
    body(
        doc,
        "Phase 1 covers still-image DeepFake screening, English text fake-news analysis, recruitment "
        "job-post screening by method transfer, and a Django-based prototype with reports. Large-scale "
        "video streaming detection and multilingual production deployment are deferred to later phases.",
    )

    h1(doc, "Chapter 2: Literature Review (Summary)")
    body(
        doc,
        "This chapter is based on the conducted literature survey and dissertation literature review on "
        "multi-threat explainable AI for digital deception detection (DeepFakes, fake news, and recruitment "
        "fraud). About fifty papers were studied thematically, with special attention to EfficientNet, "
        "Grad-CAM, BERT, LIME, and SHAP.",
    )
    h2(doc, "2.1 DeepFake Image Detection")
    body(
        doc,
        "CNN and EfficientNet models remain practical for still-image DeepFake detection. Studies report "
        "strong accuracy for EfficientNet-family backbones and increasingly attach Grad-CAM heatmaps so "
        "that suspicious facial regions can be inspected. Hybrid spatial–frequency designs and "
        "transformer hybrids can improve scores, but cross-dataset and cross-generator drops remain common.",
    )
    h2(doc, "2.2 Fake News Detection")
    body(
        doc,
        "Recent fake news systems rely heavily on BERT and related transformers. Explainable designs using "
        "LIME and SHAP show that high accuracy and local word-level evidence can coexist. Surveys also warn "
        "that explanation interfaces must be tested carefully because explanations do not always increase trust.",
    )
    h2(doc, "2.3 Recruitment Scam Detection")
    body(
        doc,
        "Technical recruitment-fraud literature is thinner than fake news research. Fraud-BERT and related "
        "job-fraud classifiers support BERT fine-tuning on imbalanced job-ad data. Social studies of scam "
        "recruitment justify why job-post screening matters, even when they do not propose algorithms.",
    )
    h2(doc, "2.4 Research Gaps")
    bullet(doc, "Threats are usually studied separately rather than in one authenticity workflow.")
    bullet(doc, "Explanation is often added after training instead of designed as evidence.")
    bullet(doc, "Recruitment scam has limited technical and explainable system coverage.")
    bullet(doc, "Cross-dataset robustness remains weak for many high-scoring models.")
    body(
        doc,
        "These gaps motivate an evidence-aware multi-threat framework joining EfficientNetB0 + Grad-CAM "
        "for images with BERT + LIME/SHAP for fake news and recruitment text.",
    )

    h1(doc, "Chapter 3: Proposed Methodology")
    h2(doc, "3.1 Framework Overview")
    body(
        doc,
        "The proposed system is organised as a multi-threat authenticity service. Users submit an image "
        "or text sample. The orchestration layer routes the sample to the correct detector, runs "
        "inference, generates an explanation artifact, and stores an analysis record for later review or PDF export.",
    )
    h2(doc, "3.2 Module Design")
    bullet(doc, "DeepFake module: EfficientNetB0 binary classifier + Grad-CAM heatmap.")
    bullet(doc, "Fake news module: BERT-style classifier + LIME token ranks + SHAP-style feature bars.")
    bullet(doc, "Recruitment module: shared text pipeline + job heuristics (urgency, contact channel, employer cues).")
    bullet(doc, "Evidence layer: label + confidence + explanation artifact in one report.")
    h2(doc, "3.3 Evaluation Plan")
    body(
        doc,
        "Phase-1 and Phase-2 evaluation will use Accuracy, Precision, Recall, and F1-score on held-out "
        "image and text samples. Explanation quality will be reviewed qualitatively (heatmap focus and "
        "token relevance). Cross-condition checks will be planned for Phase 2.",
    )

    h1(doc, "Chapter 4: Phase-1 Work Completed / Planned")
    bullet(doc, "Title finalization and Annexure-I proposal preparation.")
    bullet(doc, "Literature survey / review completed for DeepFake, fake news, recruitment, and XAI.")
    bullet(doc, "Architecture defined for multi-threat Django prototype.")
    bullet(doc, "Implementation path: EfficientNetB0 + Grad-CAM; BERT + LIME/SHAP; recruitment heuristics.")
    bullet(doc, "Dataset organization and training/evaluation workflow planned for continued Phase work.")

    h1(doc, "Chapter 5: Conclusion and Phase-2 Plan")
    body(
        doc,
        "Phase 1 establishes the research problem, literature foundation, and proposed multi-threat "
        "evidence-aware framework. Phase 2 will focus on complete training/evaluation, robustness testing, "
        "improved explanation presentation, and dissertation finalization.",
    )

    h1(doc, "Selected References (from literature base)")
    refs = [
        "Mirsky, Y., Lee, W. The Creation and Detection of Deepfakes: A Survey. ACM Computing Surveys, 2021.",
        "Kumar, M., et al. Hybrid spatial–frequency EfficientNet deepfake detection. Scientific Reports, 2026.",
        "Deepa, B.G., et al. EfficientNet-B3 DeepFake detection framework. Discover Computing, 2026.",
        "Castro, F., et al. DeepDect: explainable DeepFake detection platform. Multimedia Tools and Applications, 2026.",
        "Mansoor, N., Iliev, A.I. Explainable AI for DeepFake Detection. Applied Sciences, 2025.",
        "Alghamdi, J., Luo, S., Lin, Y. Survey on ML for fake news detection. Multimedia Tools and Applications, 2023.",
        "Sneha, S.G., et al. BiLSTM-LIME for fake news detection. Discover Artificial Intelligence, 2026.",
        "Men, X., Mariano, V.Y. BERT and SHAP for explainable fake news detection. IJMECS, 2024.",
        "Taneja, K., et al. Fraud-BERT for online recruitment fraud. Discover Computing, 2025.",
        "Naudé, M., et al. ML approach to detecting fraudulent job types. AI & SOCIETY, 2023.",
        "Moyo, B.V., et al. AI-driven conceptual framework for fake news and deepfake content. Frontiers in AI, 2026.",
        "Bezzaoui, I., et al. Explainable AI for online disinformation detection. Electronic Markets, 2025.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"[{i}] {ref}")
        font(r, 11, False)

    p = doc.add_paragraph()
    r = p.add_run(
        "Note: Full reference list and comparative tables are available in the literature review / "
        "survey documents prepared for this dissertation."
    )
    font(r, 11, False)

    outs = [
        Path(r"c:\Users\hp\OneDrive\Desktop\MTECH\3rd SEM") / "MTech_Dissertation_Phase1_Report_Draft.docx",
        Path(r"f:\deepfake_detection_project\docs") / "MTech_Dissertation_Phase1_Report_Draft.docx",
        Path(r"f:\MTECH\Literature Review") / "MTech_Dissertation_Phase1_Report_Draft.docx",
    ]
    for o in outs:
        try:
            o.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(o))
            print(o)
        except Exception as e:
            print("FAIL", o, e)


if __name__ == "__main__":
    main()
