"""Generate Parul University Annexure-I Phase 1 dissertation proposal Word document."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


TITLE = (
    "A Multi-Threat Explainable AI Framework for Digital Deception Detection "
    "and Evidence-Aware Authenticity Verification"
)

OUT_DIRS = [
    Path(r"c:\Users\hp\OneDrive\Desktop\MTECH\3rd SEM"),
    Path(r"f:\deepfake_detection_project\docs"),
    Path(r"f:\MTECH\Literature Review"),
]


def set_run_font(run, size=12, bold=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold


def add_center(doc, text, size=14, bold=True, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def add_heading_left(doc, text, size=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=True)
    return p


def add_body(doc, text, size=12, first_indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_indent:
        p.paragraph_format.first_line_indent = Inches(0.3)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=False)
    return p


def add_bullet(doc, text, size=12):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.clear()
    run = p.add_run(text)
    set_run_font(run, size=size, bold=False)
    return p


def add_label_value(doc, label, value):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run(label)
    set_run_font(r1, bold=True)
    r2 = p.add_run(value)
    set_run_font(r2, bold=False)
    return p


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1)

    add_center(doc, "ANNEXURE – I", size=14, bold=True)
    add_center(doc, "PARUL UNIVERSITY, FACULTY OF ENGINEERING AND TECHNOLOGY", size=12, bold=True)
    add_center(
        doc,
        "Department: M.Tech (Branch: Artificial Intelligence & Data Science)",
        size=12,
        bold=False,
    )
    add_center(doc, "Proposal for Dissertation Title", size=13, bold=True, space_after=12)

    add_heading_left(doc, "Title of Dissertation:")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(TITLE)
    set_run_font(run, size=12, bold=True)

    add_heading_left(doc, "Submitted By:")
    add_label_value(doc, "Name of Student: ", "Joshi Harshida Mukeshbhai")
    add_label_value(doc, "Enrollment Number: ", "2503032340019")
    add_label_value(doc, "Email: ", "harshidhijoshi@gmail.com")

    add_heading_left(doc, "Supervisor / Guide:")
    add_label_value(doc, "Name: ", "Dr. Swati Sharma")
    add_label_value(doc, "Designation: ", "Assistant Professor")
    add_label_value(doc, "Department: ", "Artificial Intelligence & Data Science")

    add_heading_left(doc, "Co-Guide:")
    add_label_value(doc, "Name: ", "Dr. Ashwini Kumar Jha")
    add_label_value(doc, "Designation: ", "Co-Guide")
    add_label_value(doc, "Email: ", "ashwinibjha@gmail.com")
    add_label_value(
        doc,
        "Affiliation: ",
        "Department of Artificial Intelligence & Data Science, "
        "Parul Institute of Engineering & Technology, Faculty of Engineering & Technology, "
        "Parul University, Waghodia, Vadodara, Gujarat, 391760, India",
    )

    add_heading_left(doc, "Abstract")
    add_body(
        doc,
        "Online deception rarely appears as one isolated problem. A synthetic face can sit on a "
        "profile page, a rewritten news claim can travel through messaging apps, and a false job "
        "post can pressure applicants into sharing documents or money. Research communities have "
        "studied DeepFake image detection, fake news classification, and recruitment scam detection "
        "in depth, but many deployed tools still treat these threats as disconnected classifiers "
        "and return only a black-box score.",
        first_indent=True,
    )
    add_body(
        doc,
        "This dissertation proposes a multi-threat explainable AI framework for digital deception "
        "detection and evidence-aware authenticity verification. The image module uses EfficientNetB0 "
        "with Grad-CAM heatmaps so that facial regions influencing the decision can be inspected. "
        "The text modules for fake news and recruitment scam detection use BERT-style classification "
        "with LIME and SHAP explanations to highlight important words and features. The literature "
        "reviewed for this work (survey and dissertation literature chapter based on about fifty "
        "papers) shows strong support for EfficientNet/Grad-CAM on images and BERT/LIME/SHAP on text, "
        "while also showing that recruitment fraud remains thinly covered and that cross-dataset "
        "robustness and explanation fidelity remain open problems.",
        first_indent=True,
    )
    add_body(
        doc,
        "The Phase-1 outcome is a web-based research prototype that returns a threat-specific label, "
        "confidence score, and explanation artifact for image and text inputs. The system is intended "
        "for authenticity checking in social media monitoring, fact-checking support, cyber-security "
        "screening, and educational research use cases, with emphasis on transparency and user trust.",
        first_indent=True,
    )

    add_heading_left(doc, "1. Objective and Scope of the Work")
    add_heading_left(doc, "1.1 Objectives", size=12)
    add_bullet(
        doc,
        "To study multi-threat digital deception covering DeepFake images, fake news, and recruitment scam content.",
    )
    add_bullet(
        doc,
        "To design an evidence-aware authenticity framework that returns label, confidence, and explanation together.",
    )
    add_bullet(
        doc,
        "To detect DeepFake images using EfficientNetB0 and provide Grad-CAM visual explanations.",
    )
    add_bullet(
        doc,
        "To detect fake news and recruitment scam text using BERT-style models with LIME/SHAP explanations.",
    )
    add_bullet(
        doc,
        "To develop a web-based Phase-1 prototype for authenticity verification and reporting.",
    )
    add_bullet(
        doc,
        "To evaluate the prototype using Accuracy, Precision, Recall, F1-score, and qualitative explanation usefulness.",
    )

    add_heading_left(doc, "1.2 Scope", size=12)
    add_bullet(doc, "DeepFake image authenticity detection with Grad-CAM evidence.")
    add_bullet(doc, "Fake news / misinformation detection in English text with LIME and SHAP.")
    add_bullet(
        doc,
        "Recruitment scam / fraudulent job-post screening using the same text pipeline plus domain heuristics.",
    )
    add_bullet(doc, "Integration of Explainable AI into a single multi-threat workflow.")
    add_bullet(doc, "Development of a Django-based research prototype with analysis history and PDF reports.")
    add_bullet(
        doc,
        "Out of scope for Phase 1: large-scale production deployment, video DeepFake streaming at scale, and multilingual production systems.",
    )

    add_heading_left(doc, "1.3 Literature Support (Brief)", size=12)
    add_body(
        doc,
        "The literature base for this proposal is drawn from the conducted survey and dissertation "
        "literature review on multi-threat explainable AI for digital deception detection. Image "
        "studies support EfficientNet-family CNN detectors and Grad-CAM as practical explainability "
        "for faces. Fake news studies strongly support BERT-style contextual classifiers and "
        "token-level explanations using LIME/SHAP. Recruitment fraud has fewer technical papers, "
        "but Fraud-BERT and related job-fraud classifiers justify transferring the text XAI pipeline "
        "to fake job advertisements. The main gaps identified are: separate treatment of threats, "
        "explanation treated as an afterthought, thin recruitment-fraud coverage, and cross-dataset "
        "brittleness. These gaps motivate the present multi-threat evidence-aware design.",
        first_indent=True,
    )

    add_heading_left(doc, "2. Experimental / Computational Work Required with Methodology")
    add_heading_left(doc, "2.1 Methodology", size=12)
    add_bullet(doc, "Collection and organization of DeepFake image datasets (real vs fake) and public fake-news / recruitment text corpora.")
    add_bullet(doc, "Data preprocessing: image resize/normalization; text cleaning and tokenization.")
    add_bullet(doc, "Train / fine-tune EfficientNetB0 for DeepFake image classification.")
    add_bullet(doc, "Integrate Grad-CAM heatmap generation for image decisions.")
    add_bullet(doc, "Deploy BERT-style text classifier for fake news analysis.")
    add_bullet(doc, "Adapt the text pipeline to recruitment scam posts with job-domain heuristics.")
    add_bullet(doc, "Generate LIME and SHAP explanations for text predictions.")
    add_bullet(doc, "Build orchestration layer so each analysis returns label + confidence + explanation artifact.")
    add_bullet(doc, "Develop Django web interface, MySQL analysis history, and PDF report generation.")
    add_bullet(doc, "Evaluate using Accuracy, Precision, Recall, F1-score, and qualitative review of explanations.")

    add_heading_left(doc, "2.2 System Modules (Proposed Framework)", size=12)
    add_bullet(doc, "Module 1 — DeepFake Image Detection: EfficientNetB0 classifier + Grad-CAM.")
    add_bullet(doc, "Module 2 — Fake News Detection: BERT-style text model + LIME/SHAP.")
    add_bullet(doc, "Module 3 — Recruitment Scam Detection: shared text model + job heuristics + LIME/SHAP.")
    add_bullet(doc, "Module 4 — Evidence/Report Layer: store analysis records and export PDF evidence summaries.")

    add_heading_left(doc, "2.3 Technology Stack", size=12)
    add_bullet(doc, "Python, Django, TensorFlow/Keras, HuggingFace Transformers")
    add_bullet(doc, "OpenCV / image utilities, LIME/SHAP explainability libraries")
    add_bullet(doc, "MySQL (XAMPP), ReportLab PDF, VS Code, Google Chrome")

    add_heading_left(doc, "3. Application of Selected Title")
    add_bullet(doc, "Social media content authenticity screening")
    add_bullet(doc, "Fake news / misinformation support for fact-checking")
    add_bullet(doc, "Recruitment platform / job-post fraud screening")
    add_bullet(doc, "Cybersecurity and digital forensics assistance")
    add_bullet(doc, "Digital journalism and content verification workflows")
    add_bullet(doc, "Educational and research demonstration of explainable authenticity systems")

    add_heading_left(doc, "4. Requirement of Fund")
    add_body(
        doc,
        "No major funding is required for this research work. The project will utilize a personal "
        "computer/laptop, internet connectivity, Visual Studio Code, Google Chrome, Python, and "
        "open-source libraries (TensorFlow, Transformers, OpenCV, NumPy, Pandas, Django, etc.), "
        "together with publicly available DeepFake, fake news, and recruitment datasets.",
        first_indent=True,
    )
    add_label_value(doc, "Estimated Cost: ", "Nil / Minimal")

    add_heading_left(doc, "5. Phase-1 Expected Deliverables")
    add_bullet(doc, "Finalized dissertation title and Annexure-I proposal.")
    add_bullet(doc, "Literature survey / literature review aligned to the multi-threat XAI topic.")
    add_bullet(doc, "Working research prototype with DeepFake, fake news, and recruitment analysis paths.")
    add_bullet(doc, "Explainability outputs (Grad-CAM for images; LIME/SHAP for text).")
    add_bullet(doc, "Documentation of architecture, datasets, and Phase-2 evaluation plan.")

    doc.add_paragraph()
    add_heading_left(doc, "Signatures")
    doc.add_paragraph()
    add_body(doc, "Signature and Name of Student: _______________________________", first_indent=False)
    add_body(doc, "Joshi Harshida Mukeshbhai", first_indent=False)
    doc.add_paragraph()
    add_body(doc, "Signature and Name of Internal Faculty Supervisor / Guide: _______________________________", first_indent=False)
    add_body(doc, "Dr. Swati Sharma, Assistant Professor", first_indent=False)
    doc.add_paragraph()
    add_body(doc, "Signature and Name of Co-Guide / External Supervisor: _______________________________", first_indent=False)
    add_body(doc, "Dr. Ashwini Kumar Jha", first_indent=False)

    # Save to multiple locations
    fname = "MTech_Dissertation_Phase1_Annexure_I_Updated.docx"
    saved = []
    for d in OUT_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            path = d / fname
            doc.save(str(path))
            saved.append(str(path))
        except Exception as e:
            saved.append(f"FAILED:{d} -> {e}")

    # Also overwrite/create a clearly named proposal next to original if possible
    alt = Path(r"c:\Users\hp\OneDrive\Desktop\MTECH\3rd SEM") / "MTech Dissertation Phase 1_Updated.docx"
    try:
        doc.save(str(alt))
        saved.append(str(alt))
    except Exception as e:
        saved.append(f"FAILED alt: {e}")

    return saved


if __name__ == "__main__":
    for s in build():
        print(s)
