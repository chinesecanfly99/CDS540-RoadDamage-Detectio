"""
PDF Generator for CDS540 Project Topic Proposal & Team Charter
Outputs: submission/CDS540_Project_Topic_Proposal.pdf
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_fonts():
    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
    # Try to find a Chinese font on Windows
    windows_fonts = [
        ("MSYaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
    ]
    for name, path in windows_fonts:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                font_name = name
                font_bold = name
                break
            except Exception as e:
                pass
    return font_name, font_bold


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 842 - 36, "CDS540 Computer Vision — Project Topic Proposal & Team Charter")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 842 - 42, 595 - 54, 842 - 42)
        # Footer
        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(595 - 54, 30, footer_text)
        self.drawString(54, 30, "Lingnan University | Academic Year 2026/2027 Term 1")
        self.restoreState()


def build_proposal_pdf():
    repo_root = Path(__file__).resolve().parent.parent
    output_pdf = repo_root / "submission" / "CDS540_Project_Topic_Proposal.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    font_name, font_bold = register_fonts()

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=50,
        rightMargin=50,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8.8,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        spaceAfter=3,
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("CDS540 Computer Vision: Group Project Topic Proposal & Charter", title_style))
    elements.append(Paragraph("Lingnan University | Master of Science in Data Science | Oct 16, 2026", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=8))

    # Section 1: Team & Topic Info Table
    elements.append(Paragraph("1. General Project & Team Information", h1_style))
    info_data = [
        [Paragraph("Project Track", table_cell_bold), Paragraph("Candidate Topic 3: Object Detection and Image Segmentation (Direction 2)", table_cell)],
        [Paragraph("Project Title (EN)", table_cell_bold), Paragraph("Automated Road Damage Detection & Operational Triage for Smart City Infrastructure", table_cell)],
        [Paragraph("Project Title (CN)", table_cell_bold), Paragraph("基于计算机视觉的智慧城市道路病害自动化检测与运维分级系统", table_cell)],
        [Paragraph("Group Leader", table_cell_bold), Paragraph("李杨尚 / Yangshang Li (Student ID: 5586669, Email: yangshangli@LN.hk)", table_cell)],
        [Paragraph("Team Members", table_cell_bold), Paragraph("吴松芫 / Songyuan Wu (5550024), 郭毅锦 / Yijin Guo (5585392),<br/>宋婧仪 / Jingyi Song (5554757), 黄浠沿 / Xiyan Huang (5594367)", table_cell)],
    ]
    t_info = Table(info_data, colWidths=[120, 375])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_info)
    elements.append(Spacer(1, 8))

    # Section 2: Executive Summary
    elements.append(Paragraph("2. Executive Summary", h1_style))
    elements.append(Paragraph(
        "<b>English Abstract:</b> Maintaining municipal road networks is critical for transportation safety and municipal budget allocation. "
        "Traditional manual surveys are labor-intensive and hazardous, while specialized laser inspection vans cost upwards of $300,000 per unit. "
        "This project develops an edge-deployable, end-to-end computer vision pipeline converting municipal fleet dashcams into an intelligent road survey network. "
        "Using the multi-country IEEE CRDDC RDD2022 dataset (19,208 images across Czech Republic, India, and the United States), we detect four canonical damage classes "
        "(<code>D00</code> Longitudinal Crack, <code>D10</code> Transverse Crack, <code>D20</code> Alligator Crack, and <code>D40</code> Pothole). "
        "We systematically compare a lightweight baseline (<b>YOLO11n</b>, 2.6M params) against a scaled detector (<b>YOLO11s</b>, 9.4M params), "
        "model the asymmetric liability of hazardous potholes, and cascade detections into an automated municipal priority triage engine (HIGH / MEDIUM / LOW). "
        "Furthermore, we evaluate zero-shot cross-continent domain transfer on 4,805 unseen US highway images, benchmark 5 adverse environmental stress conditions, "
        "and establish deterministic real-time throughput (>30 FPS) on GPU hardware.",
        body_style
    ))
    elements.append(Spacer(1, 4))

    # Section 3: Background & Key Innovations
    elements.append(Paragraph("3. Problem Motivation & Academic Value", h1_style))
    elements.append(Paragraph("• <b>Cost-Effective Infrastructure Monitoring:</b> Transforms existing municipal transit buses and sanitation sweepers into continuous road scanners, eliminating proprietary inspection van costs.", bullet_style))
    elements.append(Paragraph("• <b>Asymmetric Cost of Municipal Liability:</b> Missing superficial linear cracks (D00) merely defers scheduled sealing, whereas missing severe potholes (D40) directly triggers vehicle blowouts, cyclist crashes, and municipal tort claims. Pothole Recall is optimized as a first-class objective.", bullet_style))
    elements.append(Paragraph("• <b>End-to-End Decision Support:</b> Transcends isolated bounding boxes by computing municipal maintenance priority badges (HIGH/MEDIUM/LOW) paired with live video HUD telemetry overlay.", bullet_style))
    elements.append(Spacer(1, 6))

    # Section 4: Dataset Strategy
    elements.append(Paragraph("4. Dataset & Data Engineering Protocol", h1_style))
    elements.append(Paragraph("• <b>Benchmark Dataset:</b> IEEE BigData Crowdsourced Road Damage Detection Challenge 2022 (RDD2022), comprising 19,208 real dashcam images.", bullet_style))
    elements.append(Paragraph("• <b>Leakage-Free Splitting:</b> Strict image-level splitting with fixed seed ($seed=42$) across 10,535 in-domain images (70% Train, 15% Val, 15% Test). 851 clean road background frames with empty labels are retained to actively penalize phantom false positives.", bullet_style))
    elements.append(Paragraph("• <b>Zero-Shot Target Domain Holdout:</b> 100% of the United States dataset (4,805 images) is completely held out from training to benchmark cross-continental domain generalization (asphalt vs concrete).", bullet_style))
    elements.append(Paragraph("• <b>Coordinate Quality Assurance:</b> Developed automated XML parser verifying all 20,962 bounding boxes with zero coordinate anomalies.", bullet_style))
    elements.append(Spacer(1, 6))

    # Section 5: Experimental Plan
    elements.append(Paragraph("5. Proposed Experimental Benchmarks", h1_style))
    elements.append(Paragraph("1. <b>Baseline vs Scaled Detector:</b> YOLO11n vs YOLO11s comparison across Precision, Recall, mAP@50, mAP50-95, Pothole AP & Recall, and FPS.", bullet_style))
    elements.append(Paragraph("2. <b>Cross-Country Generalization:</b> Direct zero-shot evaluation on 4,805 US images examining geometric invariance of D00 vs material collapse of D20.", bullet_style))
    elements.append(Paragraph("3. <b>Photometric Robustness Stress Test:</b> Evaluating performance degradation across 5 physical conditions: Night/Low-light, Horizontal Motion Blur, Fog/Haze, CMOS Sensor Noise, and Clean baseline.", bullet_style))
    elements.append(Paragraph("4. <b>Precision Latency Profiling:</b> CUDA Event synchronized microsecond profiling on GPU hardware to verify the 30 FPS real-time threshold.", bullet_style))
    elements.append(Spacer(1, 6))

    # Section 6: Deliverables & Timeline Table
    elements.append(Paragraph("6. Deliverables & Course Milestones", h1_style))
    deliv_data = [
        [Paragraph("Milestone Date", table_cell_bold), Paragraph("Course Activity", table_cell_bold), Paragraph("Deliverables Planned", table_cell_bold)],
        [Paragraph("Oct 16", table_cell), Paragraph("Topic Selection (Today)", table_cell), Paragraph("Formal Proposal & Team Charter submitted to TA", table_cell)],
        [Paragraph("Oct 17 - Oct 31", table_cell), Paragraph("Data & Model Benchmarks", table_cell), Paragraph("Leakage-free split, YOLO11n/s training, XML QA inspection", table_cell)],
        [Paragraph("Nov 01 - Nov 10", table_cell), Paragraph("Stress Testing & Heuristics", table_cell), Paragraph("US domain shift (4,805 imgs), 5 perturbations, priority engine", table_cell)],
        [Paragraph("Nov 11 - Nov 19", table_cell), Paragraph("Demo & Paper Drafting", table_cell), Paragraph("120-frame HUD video demo, 6-page report PDF, 18-slide PPTX", table_cell)],
        [Paragraph("Nov 20", table_cell), Paragraph("Final Moodle Submission", table_cell), Paragraph("Formal Report PDF (<=8 pages) & Presentation PPTX on Moodle", table_cell)],
        [Paragraph("Nov 20 & 27", table_cell), Paragraph("Group Presentation", table_cell), Paragraph("17-min oral presentation + 3-min defense Q&A", table_cell)],
    ]
    t_deliv = Table(deliv_data, colWidths=[80, 140, 275])
    t_deliv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_deliv)
    elements.append(Spacer(1, 8))

    # Section 7: Planned Core Task Modules
    elements.append(Paragraph("7. Planned Project Task Modules (5 Team Members)", h1_style))
    team_data = [
        [Paragraph("Task Module", table_cell_bold), Paragraph("Core Responsibilities & Planned Deliverables", table_cell_bold)],
        [Paragraph("Module 1: Model Architecture", table_cell_bold), Paragraph("YOLO11n vs YOLO11s hyperparameter tuning, loss optimization, Pareto benchmark", table_cell)],
        [Paragraph("Module 2: Data Engineering", table_cell_bold), Paragraph("RDD2022 dataset cleaning, Pascal VOC XML validation, leakage-free split, background frames", table_cell)],
        [Paragraph("Module 3: Generalization & Robustness", table_cell_bold), Paragraph("Zero-shot US transfer benchmark (4,805 imgs), 5 environmental perturbation stress tests", table_cell)],
        [Paragraph("Module 4: Operational Priority Triage", table_cell_bold), Paragraph("Municipal risk heuristic rules, real-time dashcam inference HUD telemetry engine & video demo", table_cell)],
        [Paragraph("Module 5: Hardware & Reports", table_cell_bold), Paragraph("CUDA Event precision latency profiling, 8-page academic paper, presentation slides", table_cell)],
    ]
    t_team = Table(team_data, colWidths=[130, 365])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    elements.append(t_team)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("<i>*Note: Individual module ownership will be finalized in group meetings and submitted in the midterm milestone.</i>", subtitle_style))

    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Successfully generated proposal PDF at: {output_pdf}")


if __name__ == "__main__":
    build_proposal_pdf()
