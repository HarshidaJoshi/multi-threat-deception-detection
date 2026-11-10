"""PDF report generation for analysis history."""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

TEXT = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
BORDER = colors.HexColor("#e2e8f0")
PRIMARY_DARK = colors.HexColor("#3a56d4")
SUCCESS = colors.HexColor("#16a34a")
DANGER = colors.HexColor("#dc2626")
LIGHT_BG = colors.HexColor("#f4f7fb")
WHITE = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=PRIMARY_DARK,
            leading=22,
            spaceAfter=6,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=PRIMARY_DARK,
            spaceBefore=10,
            spaceAfter=6,
            leading=15,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT,
            leading=14,
            spaceAfter=6,
        ),
        "muted": ParagraphStyle(
            "Muted",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=MUTED,
            leading=12,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT,
            leading=14,
            leftIndent=12,
            spaceAfter=4,
        ),
    }


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(escape(str(text)), _styles()[style])


def _summary_table(rows: list[tuple[str, str]], prediction: str = "") -> Table:
    label_w = 5.2 * cm
    value_w = PAGE_W - 2 * MARGIN - label_w
    data = [[_p("Field", "muted"), _p("Value", "muted")]]
    for label, value in rows:
        data.append([_p(label, "body"), _p(value, "body")])

    table = Table(data, colWidths=[label_w, value_w], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#fafbfc")]),
    ]

    pred_lower = prediction.lower()
    if any(k in pred_lower for k in ("fake", "deepfake", "ai generated", "scam")):
        badge_color = colors.HexColor("#fef2f2")
        text_color = DANGER
    else:
        badge_color = colors.HexColor("#f0fdf4")
        text_color = SUCCESS

    for i, (label, _) in enumerate(rows, start=1):
        if label == "Prediction":
            style_cmds.append(("BACKGROUND", (1, i), (1, i), badge_color))
            style_cmds.append(("TEXTCOLOR", (1, i), (1, i), text_color))

    table.setStyle(TableStyle(style_cmds))
    return table


def _indicators_block(lines: list[str]) -> list:
    return [Paragraph(f"&bull; {escape(str(line))}", _styles()["bullet"]) for line in lines[:20]]


def _shap_table(shap_items: list[dict]) -> Table:
    data = [
        [
            _p("Feature", "muted"),
            _p("Value", "muted"),
            _p("Contribution", "muted"),
            _p("Direction", "muted"),
        ]
    ]
    for item in shap_items[:12]:
        data.append(
            [
                _p(item.get("feature", ""), "body"),
                _p(str(item.get("value", "")), "body"),
                _p(f"{item.get('contribution_pct', 0)}%", "body"),
                _p(item.get("direction", ""), "body"),
            ]
        )
    col_w = (PAGE_W - 2 * MARGIN) / 4
    table = Table(data, colWidths=[col_w] * 4, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#fafbfc")]),
            ]
        )
    )
    return table


def generate_pdf_bytes(title: str, sections: list[dict]) -> bytes:
    """Build a formatted PDF report and return raw bytes."""
    buffer = BytesIO()
    prediction = ""
    for section in sections:
        if section.get("heading") == "Analysis Summary" and section.get("rows"):
            for label, value in section["rows"]:
                if label == "Prediction":
                    prediction = value

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=title,
        author="XAI-Verify",
    )

    story: list = [
        _p(title, "title"),
        HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=10),
    ]

    for section in sections:
        heading = section.get("heading", "Section")
        lines = section.get("lines", [])
        rows = section.get("rows")
        shap = section.get("shap")
        kind = section.get("kind", "lines")

        story.append(_p(heading, "section"))

        if kind == "table" and rows:
            story.append(_summary_table(rows, prediction=prediction))
        elif kind == "shap" and shap:
            story.append(_shap_table(shap))
        elif kind == "bullets":
            story.extend(_indicators_block(lines))
        else:
            for line in lines:
                story.append(_p(line, "body"))

        story.append(Spacer(1, 6 * mm))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_pdf_file(title: str, sections: list[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fname = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(output_dir, fname)
    with open(path, "wb") as f:
        f.write(generate_pdf_bytes(title, sections))
    return path


def _result_sections(
    *,
    module: str,
    label: str,
    confidence: float,
    result: dict,
    analyzed_at: str,
    trust_score: float | None = None,
) -> list[dict]:
    exp = result.get("explainability", {})
    sections: list[dict] = []

    summary_rows = [
        ("Module", module),
        ("Prediction", label),
        ("Confidence", f"{confidence:.1f}%"),
        ("Analyzed", analyzed_at),
    ]
    if trust_score is not None:
        summary_rows.append(("Trust score", f"{trust_score:.1f}%"))
    if result.get("fake_score") is not None:
        summary_rows.append(("Fake score", f"{result['fake_score']:.1f}%"))
    if result.get("real_score") is not None:
        summary_rows.append(("Real score", f"{result['real_score']:.1f}%"))
    sections.append({"heading": "Analysis Summary", "kind": "table", "rows": summary_rows})

    model_rows = []
    if result.get("model_label"):
        model_rows.append(("Model", result["model_label"]))
    if result.get("model_used"):
        model_rows.append(("Architecture", result["model_used"]))
    if result.get("inference_backend"):
        model_rows.append(("Backend", result["inference_backend"]))
    if result.get("model_val_accuracy") is not None:
        model_rows.append(("Training val. accuracy", f"{result['model_val_accuracy']:.1f}%"))
    if result.get("model_description"):
        model_rows.append(("Description", result["model_description"]))
    if model_rows:
        sections.append({"heading": "Model Information", "kind": "table", "rows": model_rows})

    if exp.get("summary"):
        sections.append(
            {
                "heading": "Explainability Summary",
                "kind": "lines",
                "lines": [exp["summary"]],
            }
        )

    indicators = result.get("indicators") or exp.get("reasons") or []
    if indicators:
        sections.append(
            {
                "heading": "Detection Indicators",
                "kind": "bullets",
                "lines": indicators,
            }
        )

    shap = exp.get("shap_features") or []
    if shap:
        sections.append({"heading": "Feature Contributions", "kind": "shap", "shap": shap})

    lime = exp.get("lime_tokens") or []
    if lime:
        lime_lines = [
            f"{t.get('token', '')}: {t.get('weight', t.get('importance', 0)):+.3f}"
            for t in lime[:15]
            if t.get("token")
        ]
        if lime_lines:
            sections.append(
                {
                    "heading": "Word Contributions (exact, linear model)",
                    "kind": "bullets",
                    "lines": lime_lines,
                }
            )

    return sections


def build_report_sections(record) -> list[dict]:
    """Build PDF sections from an AnalysisRecord instance."""
    result = record.result_json or {}
    return _result_sections(
        module=record.get_module_display(),
        label=record.label,
        confidence=record.confidence,
        result=result,
        analyzed_at=record.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        trust_score=record.trust_score,
    )


def build_report_sections_from_item(item: dict, result: dict) -> list[dict]:
    """Build PDF sections from a demo/history dict."""
    return _result_sections(
        module=str(item.get("module", "Analysis")),
        label=str(item.get("label", "")),
        confidence=float(item.get("confidence", 0)),
        result=result,
        analyzed_at=str(item.get("created_at", ""))[:16],
        trust_score=item.get("trust_score"),
    )
