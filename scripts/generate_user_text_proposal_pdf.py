"""
Formal PDF Generator for User's Exact CDS540 Topic Proposal Text
Outputs: submission/CDS540_Group_Project_Topic_Proposal.pdf
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_fonts():
    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
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
            except Exception:
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
        self.setFont("Helvetica", 8.5)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header for page 2+
        if self._pageNumber > 1:
            self.drawString(50, 842 - 36, "CDS540 Computer Vision | Group Project Topic Proposal")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(50, 842 - 42, 595 - 50, 842 - 42)
        # Footer on all pages
        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(595 - 50, 32, footer_text)
        self.drawString(50, 32, "Lingnan University | Master of Science in Data Science | October 16, 2026")
        self.restoreState()


def build_pdf():
    repo_root = Path(__file__).resolve().parent.parent
    output_pdf = repo_root / "submission" / "CDS540_Group_Project_Topic_Proposal.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    font_name, font_bold = register_fonts()

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    doc_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3,
    )
    doc_sub = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=8,
        spaceAfter=6,
    )
    item_title = ParagraphStyle(
        'ItemTitle',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9.6,
        leading=14.8,
        textColor=colors.HexColor("#334155"),
        spaceAfter=7,
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9.3,
        leading=13.6,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName=font_bold,
        fontSize=9.3,
        leading=13.6,
        textColor=colors.HexColor("#0F172A"),
    )
    note_style = ParagraphStyle(
        'NoteStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8.8,
        leading=12.5,
        textColor=colors.HexColor("#64748B"),
        spaceBefore=4,
    )

    elements = []

    # Title & Subtitle Banner
    elements.append(Paragraph("CDS540 Computer Vision | Group Project Topic Proposal", doc_title))
    elements.append(Paragraph("Lingnan University | Master of Science in Data Science | October 16, 2026", doc_sub))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=8))

    # 1. Project Selection (What We Selected)
    elements.append(Paragraph("1. Project Selection (What We Selected)", h1_style))
    selection_table_data = [
        [Paragraph("Project Category", table_cell_bold),
         Paragraph("Candidate Topic 3 — Object Detection and Image Segmentation (Direction 2).", table_cell)],
        [Paragraph("Proposed Title (EN)", table_cell_bold),
         Paragraph("Automated Road Damage Detection and Maintenance Prioritization Using Deep Learning (Operational Triage for Smart City Infrastructure).", table_cell)],
        [Paragraph("Proposed Title (CN)", table_cell_bold),
         Paragraph("基于计算机视觉的智慧城市道路病害自动化检测与运维分级系统", table_cell)],
        [Paragraph("Core Objective", table_cell_bold),
         Paragraph("This project aims to develop a deep learning-based computer vision prototype that automatically detects four canonical categories of road surface defects from dashcam images—longitudinal cracks (D00), transverse cracks (D10), alligator cracks (D20), and potholes (D40)—and assigns illustrative maintenance priority levels (High, Medium, or Low) to support road maintenance decision-making.", table_cell)],
    ]
    t_selection = Table(selection_table_data, colWidths=[120, 379])
    t_selection.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_selection)
    elements.append(Spacer(1, 6))

    # 2. Problem Motivation (Why We Selected This Topic)
    elements.append(Paragraph("2. Problem Motivation (Why We Selected This Topic)", h1_style))
    elements.append(Paragraph(
        "• <b>Cost-Effective & Safe Road Inspection:</b> Road surface defects compromise transportation safety and increase municipal maintenance costs. Traditional manual inspections are time-consuming, labor-intensive, and hazardous, whereas specialized laser inspection vans cost over $300,000 per unit. Automated analysis using existing municipal fleet dashcams (e.g., transit buses and sanitation sweepers) provides a practical and cost-effective complement to manual surveys.",
        body_style
    ))
    elements.append(Paragraph(
        "• <b>Asymmetric Risk of Road Defects:</b> Different road damage types pose varying levels of risk. Missing superficial linear cracks (D00) merely delays routine sealing, whereas missing severe potholes (D40) can directly lead to vehicle blowouts, cyclist accidents, and municipal liability claims. Therefore, optimizing pothole detection recall is a key priority of our study.",
        body_style
    ))
    elements.append(Paragraph(
        "• <b>From Detection to Decision Support:</b> Rather than stopping at isolated bounding boxes, combining defect detection with a simple maintenance priority triage indicator helps translate model outputs into intuitive, actionable information for maintenance planning.",
        body_style
    ))
    
    # Balanced Page Break: Section 1 & 2 on Page 1, Section 3 & 4 on Page 2
    from reportlab.platypus import PageBreak
    elements.append(PageBreak())

    # 3. Methodology & Implementation Plan (How We Will Do It)
    elements.append(Paragraph("3. Methodology & Implementation Plan (How We Will Do It)", h1_style))
    elements.append(Paragraph(
        "• <b>Dataset Preparation:</b> We will use the multi-country Road Damage Dataset 2022 (RDD2022), which contains real road dashcam images from the Czech Republic, India, and the United States. Our data preparation includes dataset cleaning, XML bounding box validation, and a leakage-free train/validation/test split (70% / 15% / 15%), while retaining clean background road images to reduce false positives.",
        body_style
    ))
    elements.append(Paragraph(
        "• <b>Model Training & Comparative Evaluation:</b> We will train and compare two object detection models—a lightweight baseline (YOLO11n) and a scaled detector (YOLO11s). Their performance will be evaluated using precision, recall (with particular attention to pothole recall), mean average precision (mAP), and real-time inference speed (FPS).",
        body_style
    ))
    elements.append(Paragraph(
        "• <b>Generalization & Robustness Testing:</b> Subject to available time and computing resources, we will evaluate cross-country generalization on unseen United States road images and test model robustness under challenging visual conditions, including night/low-light, motion blur, fog/haze, and sensor noise.",
        body_style
    ))
    elements.append(Paragraph(
        "• <b>Rule-Based Priority Triage & Deliverables:</b> We will implement a simple rule-based heuristic module to assign High, Medium, or Low maintenance priorities to detected road damage as illustrative decision-support indicators. The final deliverables will include a working prototype with visual HUD overlays of detected defects and priority levels, a comparative evaluation report, and final presentation slides.",
        body_style
    ))
    elements.append(Spacer(1, 6))

    # 4. Group Members & Task Modules
    elements.append(Paragraph("4. Group Members & Task Modules", h1_style))
    
    # Team members info
    member_table_data = [
        [Paragraph("Group Leader", table_cell_bold),
         Paragraph("Yangshang Li / 李杨尚 (Student ID: 5586669, Email: yangshangli@LN.hk)", table_cell)],
        [Paragraph("Team Members", table_cell_bold),
         Paragraph("Songyuan Wu / 吴松芫 (5550024), Yijin Guo / 郭毅锦 (5585392),<br/>Jingyi Song / 宋婧仪 (5554757), Xiyan Huang / 黄习沿 (5594367)", table_cell)],
    ]
    t_members = Table(member_table_data, colWidths=[120, 379])
    t_members.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    elements.append(t_members)
    elements.append(Spacer(1, 6))

    # Planned Task Modules Table
    elements.append(Paragraph("<b>Planned Task Modules:</b>", item_title))
    elements.append(Spacer(1, 3))
    module_table_data = [
        [Paragraph("Task Module", table_cell_bold), Paragraph("Core Scope & Activities", table_cell_bold)],
        [Paragraph("Module 1 (Model Architecture)", table_cell_bold),
         Paragraph("YOLO11n and YOLO11s training, hyperparameter tuning, and performance benchmarking.", table_cell)],
        [Paragraph("Module 2 (Data Engineering)", table_cell_bold),
         Paragraph("RDD2022 dataset cleaning, annotation validation, and dataset splitting.", table_cell)],
        [Paragraph("Module 3 (Generalization & Robustness)", table_cell_bold),
         Paragraph("Cross-country generalization evaluation and adverse image condition testing.", table_cell)],
        [Paragraph("Module 4 (Operational Priority Triage)", table_cell_bold),
         Paragraph("Rule-based maintenance priority assignment and prototype visualization demo.", table_cell)],
        [Paragraph("Module 5 (Speed Profiling & Reporting)", table_cell_bold),
         Paragraph("Inference speed evaluation, final report writing, and presentation preparation.", table_cell)],
    ]
    t_modules = Table(module_table_data, colWidths=[160, 339])
    t_modules.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_modules)

    # Note
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("<i>Note: Individual module assignments will be finalized during group meetings and reported in the midterm milestone.</i>", note_style))

    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Successfully generated proposal PDF at: {output_pdf}")


if __name__ == "__main__":
    build_pdf()
