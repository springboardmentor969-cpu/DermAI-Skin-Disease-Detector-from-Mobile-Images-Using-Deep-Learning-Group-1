from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy.orm import Session

from app.models.case import CaseRecord
from app.models.image import ImageRecord
from app.models.prediction import PredictionRecord
from app.models.user import User
from app.services.case_service import get_case_by_id

HAM10000_LABELS: dict[str, str] = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

DISCLAIMER_TEXT = (
    "DermAI is an AI-assisted screening research and clinical decision-support prototype. "
    "It does not provide an autonomous medical diagnosis or clinical prescription. "
    "All analytical findings, probability scores, segmentation masks, and Grad-CAM activations "
    "must be clinically verified by a licensed dermatologist or medical practitioner."
)


def build_case_report_data(db: Session, current_user: User, case_id: int) -> dict[str, Any]:
    case = get_case_by_id(db, current_user, case_id)
    if not case:
        raise ValueError("Case not found or access denied")

    patient = db.query(User).filter(User.id == case.user_id).first()
    doctor = db.query(User).filter(User.id == case.reviewed_by_id).first() if case.reviewed_by_id else None

    images = db.query(ImageRecord).filter(ImageRecord.case_id == case_id).order_by(ImageRecord.created_at.desc()).all()
    predictions = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.case_id == case_id)
        .order_by(PredictionRecord.created_at.desc())
        .all()
    )

    image_details: list[dict[str, Any]] = []
    for img in images:
        analysis = img.analysis or {}
        clf = analysis.get("classification") or {}
        risk = analysis.get("risk_assessment") or {}
        expl = analysis.get("explainability") or {}
        seg = analysis.get("lesion_detection") or {}
        qual = analysis.get("quality_assessment") or {}

        pred_class = clf.get("predicted_class")
        image_details.append(
            {
                "image_id": img.id,
                "filename": img.filename,
                "created_at": img.created_at.isoformat() if img.created_at else "",
                "file_size_bytes": img.file_size,
                "dimensions": f"{img.width}x{img.height}" if img.width and img.height else "N/A",
                "quality_score": qual.get("quality_score"),
                "quality_category": qual.get("quality_category", "acceptable"),
                "predicted_class": pred_class,
                "predicted_label": HAM10000_LABELS.get(pred_class, pred_class or "Unknown"),
                "confidence": clf.get("confidence"),
                "calibrated_confidence": clf.get("calibrated_confidence"),
                "uncertainty_entropy": clf.get("uncertainty_entropy"),
                "model_architecture": clf.get("architecture"),
                "risk_level": risk.get("risk_level", "Unknown"),
                "severity_score": risk.get("severity_score"),
                "clinical_urgency": risk.get("urgency", "Routine"),
                "recommended_action": risk.get("recommended_action"),
                "segmentation_status": seg.get("status"),
                "lesion_detected": seg.get("lesion_detected", False),
                "explainability_method": expl.get("method", "Grad-CAM"),
                "peak_attention_coords": expl.get("peak_attention_coords"),
                "attention_coverage_percent": expl.get("attention_coverage_percent"),
                "top_predictions": clf.get("top_predictions", []),
            }
        )

    return {
        "report_id": f"DERMAI-CASE-{case.id}-{int(datetime.utcnow().timestamp())}",
        "generated_at": datetime.utcnow().isoformat(),
        "case": {
            "id": case.id,
            "title": case.title,
            "description": case.description or "No clinical history provided.",
            "status": case.status,
            "review_status": case.review_status,
            "doctor_review_notes": case.doctor_review_notes or "Pending specialist review.",
            "created_at": case.created_at.isoformat() if case.created_at else "",
            "updated_at": case.updated_at.isoformat() if case.updated_at else "",
        },
        "patient": {
            "id": patient.id if patient else case.user_id,
            "name": patient.full_name if patient else "De-identified Patient",
            "email": patient.email if patient else "",
            "role": "patient",
        },
        "reviewer": {
            "id": doctor.id if doctor else None,
            "name": doctor.full_name if doctor else "Pending Dermatologist Review",
            "email": doctor.email if doctor else "",
            "role": "doctor" if doctor else "",
        },
        "total_images": len(image_details),
        "total_predictions": len(predictions),
        "images": image_details,
        "disclaimer": DISCLAIMER_TEXT,
    }


def generate_case_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2)


def generate_case_csv(data: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "Report ID",
            "Case ID",
            "Case Title",
            "Patient Name",
            "Review Status",
            "Reviewer",
            "Image ID",
            "Filename",
            "Upload Date",
            "Quality Category",
            "Predicted Class",
            "Predicted Diagnosis",
            "Raw Confidence",
            "Calibrated Confidence",
            "Uncertainty Entropy",
            "Model Architecture",
            "Risk Level",
            "Severity Score",
            "Urgency",
            "Segmentation Status",
            "Grad-CAM Attention Coverage %",
            "Doctor Notes",
        ]
    )

    case = data["case"]
    patient = data["patient"]
    reviewer = data["reviewer"]
    report_id = data["report_id"]

    if not data["images"]:
        writer.writerow(
            [
                report_id,
                case["id"],
                case["title"],
                patient["name"],
                case["review_status"],
                reviewer["name"],
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                case["doctor_review_notes"],
            ]
        )
    else:
        for img in data["images"]:
            writer.writerow(
                [
                    report_id,
                    case["id"],
                    case["title"],
                    patient["name"],
                    case["review_status"],
                    reviewer["name"],
                    img["image_id"],
                    img["filename"],
                    img["created_at"],
                    img["quality_category"],
                    img["predicted_class"],
                    img["predicted_label"],
                    f"{img['confidence']:.4f}" if img["confidence"] is not None else "N/A",
                    f"{img['calibrated_confidence']:.4f}" if img["calibrated_confidence"] is not None else "N/A",
                    f"{img['uncertainty_entropy']:.3f}" if img["uncertainty_entropy"] is not None else "N/A",
                    img["model_architecture"],
                    img["risk_level"],
                    img["severity_score"],
                    img["clinical_urgency"],
                    img["segmentation_status"],
                    img["attention_coverage_percent"],
                    case["doctor_review_notes"],
                ]
            )

    return output.getvalue()


def generate_case_pdf(data: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0d8f7e"),
        alignment=0,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
    )

    story: list[Any] = []

    # Header
    story.append(Paragraph("<b>DermAI · Comprehensive Medical Analysis Report</b>", title_style))
    story.append(
        Paragraph(
            f"Report ID: {data['report_id']} | Generated: {data['generated_at']} | Milestone 4 Clinical Decision Support",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d8f7e"), spaceAfter=10))

    # Case & Patient Summary Table
    case = data["case"]
    patient = data["patient"]
    reviewer = data["reviewer"]

    case_meta = [
        [
            Paragraph("<b>Case Record:</b>", body_style),
            Paragraph(f"#{case['id']} - {case['title']}", body_style),
            Paragraph("<b>Patient Name:</b>", body_style),
            Paragraph(f"{patient['name']} (ID #{patient['id']})", body_style),
        ],
        [
            Paragraph("<b>Clinical Status:</b>", body_style),
            Paragraph(case["status"].upper(), body_style),
            Paragraph("<b>Triage Review:</b>", body_style),
            Paragraph(case["review_status"].replace("_", " ").upper(), body_style),
        ],
        [
            Paragraph("<b>Created Date:</b>", body_style),
            Paragraph(case["created_at"][:10], body_style),
            Paragraph("<b>Reviewer:</b>", body_style),
            Paragraph(reviewer["name"], body_style),
        ],
    ]
    t_case = Table(case_meta, colWidths=[90, 180, 90, 180])
    t_case.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_case)
    story.append(Spacer(1, 10))

    # Doctor Review Notes Section
    story.append(Paragraph("<b>Dermatologist Consultation & Review Notes</b>", h2_style))
    notes_p = Paragraph(f"<i>{case['doctor_review_notes']}</i>", body_style)
    t_notes = Table([[notes_p]], colWidths=[540])
    t_notes.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#bfdbfe")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_notes)
    story.append(Spacer(1, 10))

    # Image Analyses
    story.append(Paragraph(f"<b>Lesion Analysis & Diagnostic Findings ({len(data['images'])} Images)</b>", h2_style))

    if not data["images"]:
        story.append(Paragraph("No dermoscopic or clinical images recorded for this case.", body_style))
    else:
        for idx, img in enumerate(data["images"], start=1):
            story.append(
                Paragraph(
                    f"<b>Session #{idx}: {img['filename']} (Image ID #{img['image_id']})</b>",
                    ParagraphStyle("ImgH3", parent=body_style, fontSize=10, textColor=colors.HexColor("#087f71")),
                )
            )

            conf_str = f"{(img['confidence'] * 100):.1f}%" if img["confidence"] is not None else "N/A"
            calib_str = f"{(img['calibrated_confidence'] * 100):.1f}%" if img["calibrated_confidence"] is not None else conf_str
            sev_str = f"{img['severity_score']}/100" if img["severity_score"] is not None else "N/A"
            cov_str = f"{img['attention_coverage_percent']}%" if img["attention_coverage_percent"] is not None else "N/A"

            img_table_data = [
                [
                    Paragraph("<b>Primary Disease Class:</b>", body_style),
                    Paragraph(f"<b>{img['predicted_label']}</b> ({img['predicted_class']})", body_style),
                    Paragraph("<b>Risk Categorization:</b>", body_style),
                    Paragraph(f"<b>{img['risk_level']}</b> (Urgency: {img['clinical_urgency']})", body_style),
                ],
                [
                    Paragraph("<b>Raw Confidence:</b>", body_style),
                    Paragraph(conf_str, body_style),
                    Paragraph("<b>Calibrated Confidence:</b>", body_style),
                    Paragraph(calib_str, body_style),
                ],
                [
                    Paragraph("<b>Severity Index:</b>", body_style),
                    Paragraph(sev_str, body_style),
                    Paragraph("<b>AI Architecture:</b>", body_style),
                    Paragraph(str(img["model_architecture"] or "MobileNetV2 / Multi-Model"), body_style),
                ],
                [
                    Paragraph("<b>Lesion Segmentation:</b>", body_style),
                    Paragraph(f"ISBI U-Net ({img['segmentation_status']})", body_style),
                    Paragraph("<b>Grad-CAM Attention:</b>", body_style),
                    Paragraph(f"Coverage: {cov_str}", body_style),
                ],
            ]
            t_img = Table(img_table_data, colWidths=[120, 150, 120, 150])
            t_img.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(t_img)
            story.append(Spacer(1, 6))

            # Top predictions breakdown
            if img["top_predictions"]:
                top_rows = [[Paragraph("<b>Diagnosis Class</b>", body_style), Paragraph("<b>Probability</b>", body_style)]]
                for p in img["top_predictions"][:4]:
                    lbl = p.get("label") or HAM10000_LABELS.get(p.get("class_name"), p.get("class_name"))
                    prob = f"{(p.get('confidence', 0.0) * 100):.1f}%"
                    top_rows.append([Paragraph(str(lbl), body_style), Paragraph(prob, body_style)])
                t_top = Table(top_rows, colWidths=[380, 160])
                t_top.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f4f2")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccdcdc")),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ]
                    )
                )
                story.append(t_top)
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=8))
    story.append(Paragraph(f"<b>Clinical Decision Support Disclaimer:</b> {DISCLAIMER_TEXT}", disclaimer_style))

    doc.build(story)
    return buffer.getvalue()
