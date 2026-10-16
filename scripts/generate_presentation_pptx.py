"""
Automated PowerPoint Presentation Deck Generator
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Compiles the formal 18-slide presentation deck:
  - Output: reports/CDS540_Final_Presentation_Deck.pptx
  - Widescreen 16:9 layout
  - Clean, professional typography & color scheme
  - Embeds all generated 300 DPI figures and quantitative comparison tables
  - Full verbatim speaker notes populated in PowerPoint presenter view!
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor


def create_presentation():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    
    pptx_path = root_dir / "reports" / "CDS540_Final_Presentation_Deck.pptx"
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    blank_layout = prs.slide_layouts[6]
    
    # Color palette
    NAVY = RGBColor(26, 54, 93)       # #1a365d
    STEEL_BLUE = RGBColor(43, 92, 143) # #2b5c8f
    DARK_GRAY = RGBColor(40, 40, 40)
    LIGHT_GRAY = RGBColor(245, 247, 250)
    AMBER = RGBColor(217, 95, 2)
    GREEN = RGBColor(27, 158, 119)
    WHITE = RGBColor(255, 255, 255)
    
    def add_header(slide, title_text, category_text="LINGNAN UNIVERSITY CDS540 COMPUTER VISION"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = STEEL_BLUE
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    # ==================== Slide 1: Title Slide ====================
    slide1 = prs.slides.add_slide(blank_layout)
    bg = slide1.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(7.5)) # MSO_SHAPE.RECTANGLE = 1
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    
    t_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.9), Inches(3.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "AUTOMATED ROAD DAMAGE DETECTION & OPERATIONAL TRIAGE FOR SMART CITY INFRASTRUCTURE"
    p0.font.name = "Arial"
    p0.font.size = Pt(30)
    p0.font.bold = True
    p0.font.color.rgb = WHITE
    p0.space_after = Pt(14)
    
    p1 = tf1.add_paragraph()
    p1.text = "Edge-Deployable Vision Intelligence for Municipal Fleet Dashcams (IEEE CRDDC RDD2022)"
    p1.font.name = "Arial"
    p1.font.size = Pt(18)
    p1.font.color.rgb = RGBColor(190, 215, 245)
    p1.space_after = Pt(24)
    
    p2 = tf1.add_paragraph()
    p2.text = "Course: Lingnan University CDS540 Computer Vision | Term 1, 2026/2027\nGroup Direction 2: Computer Vision for Infrastructure Asset Management"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(220, 220, 220)
    
    slide1.notes_slide.notes_text_frame.text = (
        "Good morning Professor and fellow classmates. Today, our team is proud to present our project: "
        "Automated Road Damage Detection and Operational Triage for Smart City Infrastructure Maintenance. "
        "In this project, we demonstrate how edge-based computer vision transforms standard vehicle dashcams "
        "into an intelligent municipal survey network that detects hazardous roadway distress in real-time."
    )

    # ==================== Slide 2: Problem & Dilemma ====================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Municipal Dilemma: Reactive vs. Proactive Road Maintenance")
    
    box2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf2 = box2.text_frame
    tf2.word_wrap = True
    
    points2 = [
        ("Aging Infrastructure & Maintenance Backlog", "Urban asphalt surfaces degrade continuously under heavy vehicular axle loads, seasonal thermal expansion, and moisture ingress, generating massive municipal repair backlogs."),
        ("Limitations of Manual Foot Patrols", "In-person visual inspections are slow, subjective, prone to human error and fatigue, and expose municipal workers to severe safety hazards in live traffic."),
        ("Prohibitive Cost of Specialized Survey Vans", "Commercial survey vans equipped with 3D LiDAR and laser profilometers cost upwards of $300,000 each, preventing frequent city-wide deployment for municipal councils."),
        ("The Smart City Dashcam Opportunity", "Public transit buses, street sweepers, and refuse trucks traverse metropolitan streets daily. Equipping them with low-cost automotive dashcams and edge AI creates a continuous survey network.")
    ]
    for title, desc in points2:
        p_t = tf2.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(10)
        
        p_d = tf2.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(8)
        
    slide2.notes_slide.notes_text_frame.text = (
        "Municipal road authorities face a massive dilemma. Roads degrade constantly under heavy traffic and rain, "
        "but traditional inspection methods cannot scale. Manual walking patrols put workers at risk on busy roads, "
        "while specialized LiDAR vans cost hundreds of thousands of dollars. Our objective is to leverage ubiquitous "
        "fleet dashcams to build an automated, edge-intelligent inspection system."
    )

    # ==================== Slide 3: Asymmetric Cost ====================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Core Design Principle: The Asymmetric Cost of Municipal Liability")
    
    box3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf3 = box3.text_frame
    tf3.word_wrap = True
    
    points3 = [
        ("Not All False Negatives Are Created Equal", "Standard computer vision metrics (e.g. mAP) treat all classes symmetrically. However, in municipal road asset governance, defect consequences differ fundamentally."),
        ("Missing a Fine Crack (D00): Minor Operational Consequence", "Omitting a superficial hairline crack merely defers preventive bituminous sealant maintenance with zero immediate public safety risk."),
        ("Missing a Deep Pothole (D40): Catastrophic Public Safety Risk", "Potholes induce immediate tire blowouts, wheel rim fractures, vehicular control loss, and severe injuries to motorcyclists and cyclists—resulting in direct municipal tort liability."),
        ("System Mandate: Prioritize Pothole Recall (D40)", "We treat Pothole Recall as an indispensable safety objective alongside mAP, designing models and triage heuristics that explicitly minimize missed potholes.")
    ]
    for title, desc in points3:
        p_t = tf3.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = AMBER if "Pothole" in title or "Mandate" in title else NAVY
        p_t.space_before = Pt(10)
        
        p_d = tf3.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(8)
        
    slide3.notes_slide.notes_text_frame.text = (
        "Crucially, from an engineering and public administration perspective, all false negatives are not equal. "
        "If the AI misses a hairline longitudinal crack, maintenance is merely deferred. But if it misses a deep pothole, "
        "a vehicle blows a tire or a cyclist crashes, directly resulting in municipal injury lawsuits. Therefore, "
        "throughout this project, we treat Pothole Recall as an indispensable safety objective."
    )

    # ==================== Slide 4: Dataset & Quality ====================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Dataset & Canonical Damage Taxonomy (IEEE CRDDC RDD2022)")
    
    box4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf4 = box4.text_frame
    tf4.word_wrap = True
    
    points4 = [
        ("Benchmark Dataset: IEEE CRDDC 2022", "19,208 raw multi-country road dashcam frames captured across Czech Republic, India, and the United States."),
        ("D00 - Longitudinal Crack (Wheel-Path Fatigue)", "Linear cracks propagating parallel to vehicular travel direction, indicating base fatigue or cold paving seams."),
        ("D10 - Transverse Crack (Thermal Contraction)", "Perpendicular cracks caused by asphalt contraction during seasonal temperature drops, degrading ride quality."),
        ("D20 - Alligator Crack (Structural Subgrade Failure)", "Interconnected polygonal pattern resembling reptile skin. Indicates structural subgrade collapse and high water ingress vulnerability."),
        ("D40 - Pothole (High-Hazard Impact Depression)", "Bowl-shaped pavement depressions caused by structural exhaustion and subgrade moisture liquefaction.")
    ]
    for title, desc in points4:
        p_t = tf4.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(8)
        
        p_d = tf4.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide4.notes_slide.notes_text_frame.text = (
        "We utilized the IEEE CRDDC RDD2022 dataset, comprising over 19,000 multi-country road dashcam frames. "
        "The challenge focuses on four canonical damage types: D00 longitudinal cracks, D10 transverse cracks, "
        "D20 alligator fatigue cracking, and D40 potholes. Each distress carries distinct physical causes and distinct maintenance urgency."
    )

    # ==================== Slide 5: Quality Assessment & QA Inspector ====================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Data Engineering, Quality Assessment & Coordinate Validation")
    
    box5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf5 = box5.text_frame
    tf5.word_wrap = True
    
    points5 = [
        ("Automated Annotation QA Inspector Suite", "Developed scripts/build_data_inventory.py and scripts/annotation_qa_inspector.py to parse all 15,340 Pascal VOC XML files."),
        ("Zero Coordinate Anomalies Confirmed", "Audited 20,962 bounding boxes across all countries: verified 100% compliance with x_min < x_max and y_min < y_max within legal image dimensions [0, W] x [0, H]."),
        ("Physical Distribution Insights from EDA", "Severe class imbalance: cracks represent >85% of annotations, while potholes are rare. High aspect ratio: fine cracks are only 2–4 pixels wide on 640p frames."),
        ("Negative Background Frame Strategy", "Retained 851 clean road frames in test set with empty label files to penalize phantom false positives over clean asphalt.")
    ]
    for title, desc in points5:
        p_t = tf5.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(8)
        
        p_d = tf5.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide5.notes_slide.notes_text_frame.text = (
        "Before training any neural network, data quality must come first. We wrote an automated QA suite parsing "
        "all 15,340 Pascal VOC annotations across nearly 21,000 bounding boxes, confirming zero coordinate anomalies. "
        "Our exploratory data analysis revealed two key physical challenges: extreme class imbalance and micro-scale "
        "aspect ratios, where fine cracks are only a few pixels wide on compressed 640p frames."
    )

    # ==================== Slide 6: Leakage-Free Splitting ====================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Leakage-Free Partitioning & Zero-Shot Target Domain Isolation")
    
    box6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf6 = box6.text_frame
    tf6.word_wrap = True
    
    points6 = [
        ("Strict Image-Level Splitting Policy", "Partitioning was performed strictly at the image level with fixed random seed (seed = 42). Training, Validation, and Test partitions share zero intersection."),
        ("In-Domain Partition (Czech Republic & India)", "7,375 Training images (70%), 1,581 Validation images (15%), and 1,579 Testing images (15%)."),
        ("100% Held-Out Cross-Country Domain Test", "All 4,805 images from the United States highway dataset were strictly held out from all training, validation, and hyperparameter tuning."),
        ("Zero Redundant Storage Overhead", "Implemented NTFS hard links in scripts/voc_to_yolo.py, organizing Ultralytics YOLO directories instantly without duplicating 14 GB of image files.")
    ]
    for title, desc in points6:
        p_t = tf6.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(8)
        
        p_d = tf6.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide6.notes_slide.notes_text_frame.text = (
        "To ensure academic integrity and prevent data leakage, all splits were performed strictly at the image level "
        "with fixed random seeds. In addition, we deliberately preserved 851 negative frames with empty label files. "
        "Without negative frames, object detectors produce rampant false alarms over asphalt markings in real deployments. "
        "Furthermore, 100% of the US highway data was held out for zero-shot testing."
    )

    # ==================== Slide 7: End-to-End Pipeline ====================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "End-to-End Edge Vision & Decision Support Pipeline")
    
    box7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf7 = box7.text_frame
    tf7.word_wrap = True
    
    stages = [
        ("1. Dashcam Stream Ingestion", "Continuous frame capture from vehicle dashcam at 1280x720, letterboxed to 640x640."),
        ("2. Deep Neural Detection", "Single-stage YOLO forward pass executed on onboard GPU Tensor Cores."),
        ("3. Non-Maximum Suppression", "IoU overlap thresholding (IoU = 0.70, confidence = 0.25) to eliminate duplicate boxes."),
        ("4. Automated Priority Triage", "Heuristic calculation: Severity = Multiplicity * Sum(Class_Weight * Confidence * Area_Factor)."),
        ("5. Real-Time HUD Overlay & Dispatch", "Overlays top telemetry HUD banner with priority badges (HIGH/MEDIUM/LOW) and real-time FPS.")
    ]
    for title, desc in stages:
        p_t = tf7.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = STEEL_BLUE
        p_t.space_before = Pt(6)
        
        p_d = tf7.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(4)
        
    slide7.notes_slide.notes_text_frame.text = (
        "Here is our end-to-end operational architecture. Incoming dashcam frames are letterboxed and passed through "
        "the neural network. Bounding boxes are filtered by NMS, and then immediately fed into our deterministic priority "
        "rules engine. The engine calculates an actionable risk score based on hazard weights, distress area, and clustering, "
        "rendering a live HUD telemetry dashboard for the driver or central dispatcher."
    )

    # ==================== Slide 8: In-Domain Results ====================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Quantitative Model Comparison (YOLO11n Baseline vs. YOLO11s)")
    
    fig_comp = root_dir / "results" / "evaluations" / "05_accuracy_vs_speed_comparison.png"
    if fig_comp.exists():
        slide8.shapes.add_picture(str(fig_comp), Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    box8 = slide8.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.0))
    tf8 = box8.text_frame
    tf8.word_wrap = True
    
    findings8 = [
        ("Substantial +31.2% mAP Boost", "YOLO11s scales mAP@50 from 0.2169 to 0.2846 over the baseline on 1,579 independent test images."),
        ("Dramatic Pothole Gain (+39.7%)", "D40 Pothole AP50 surges from 0.2093 to 0.2923. Pothole Recall reaches 39.74% (+19.2% relative gain)."),
        ("Alligator Crack Breakthrough", "D20 AP50 reaches 0.5210 (+31.6% gain), proving that scaled depth successfully captures complex multi-branch fissures."),
        ("Real-Time Dashcam Viability", "Both models comfortably exceed the 30 FPS ceiling (YOLO11n: 45.5 FPS, YOLO11s: 38.6 FPS).")
    ]
    for title, desc in findings8:
        p_t = tf8.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(6)
        
        p_d = tf8.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(4)
        
    slide8.notes_slide.notes_text_frame.text = (
        "When evaluated on our 1,579 independent test images, the scaled YOLO11s achieved a massive leap in accuracy: "
        "overall mAP@50 jumped by +31.2% to 0.2846. Even more significantly for municipal road departments, D40 pothole AP "
        "increased by nearly 40%, and pothole recall rose to 39.74%, directly addressing our asymmetric cost objective."
    )

    # ==================== Slide 9: Cross-Country Domain Shift ====================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Cross-Country Domain Generalization (Unseen US Highway Evaluation)")
    
    fig_dom = root_dir / "results" / "evaluations" / "06_cross_domain_generalization.png"
    if fig_dom.exists():
        slide9.shapes.add_picture(str(fig_dom), Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    box9 = slide9.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.0))
    tf9 = box9.text_frame
    tf9.word_wrap = True
    
    findings9 = [
        ("Zero-Shot Target Evaluation", "Direct evaluation on 4,805 completely unseen United States highway dashcam images."),
        ("Generalization Decay (~69%)", "Overall mAP@50 dropped from 0.285 to 0.088, demonstrating the risk of cross-continental deployment without adaptation."),
        ("Longitudinal Invariance (-4.98%)", "D00 AP50 experienced only a tiny -4.98% drop (0.2092 to 0.1988), proving high geometric transferability across continents."),
        ("Material Gap Breakdown (-82%)", "D20 alligator cracking collapsed by 82% due to visual domain shifts between Indian asphalt distress and US concrete slab expansion joints.")
    ]
    for title, desc in findings9:
        p_t = tf9.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(6)
        
        p_d = tf9.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(4)
        
    slide9.notes_slide.notes_text_frame.text = (
        "Next, we tested real-world transferability by deploying our models zero-shot on 4,805 unseen US highway images. "
        "As expected, overall mAP dropped by roughly 69%. However, we discovered a remarkable phenomenon: single-track "
        "longitudinal cracks decayed by less than 5%! Cracks share universal linear geometry across continents. In contrast, "
        "alligator cracking collapsed by over 80% because US concrete slab joints and tar sealant snakes look fundamentally different."
    )

    # ==================== Slide 10: Environmental Robustness ====================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Environmental Stress Testing: Simulated Adverse Operating Conditions")
    
    fig_mos = root_dir / "results" / "evaluations" / "07_robustness_perturbation_mosaic.png"
    if fig_mos.exists():
        slide10.shapes.add_picture(str(fig_mos), Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    box10 = slide10.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.0))
    tf10 = box10.text_frame
    tf10.word_wrap = True
    
    findings10 = [
        ("Controlled Perturbation Testing", "Stress-tested on 350 test images under 5 conditions: Clean, Low-Light, Motion Blur, Fog, and Sensor Noise."),
        ("Motion Blur Asymmetry Insight", "1D horizontal blur crushed crack detection by -86.9%, but D40 Pothole Recall decayed by only -18.7%! 2D depressions retain spatial contrast even when blurred."),
        ("Adverse Fog Destroys Pothole Recall", "Atmospheric scattering (Koschmieder law) attenuates depth shadow contrast, causing a -67.0% collapse in pothole recall."),
        ("Sensor Noise Sensitivity", "High-ISO Gaussian noise destroyed CNN gradients (-97.8% drop), indicating high sensitivity to cheap dashcam CMOS sensors.")
    ]
    for title, desc in findings10:
        p_t = tf10.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(6)
        
        p_d = tf10.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(4)
        
    slide10.notes_slide.notes_text_frame.text = (
        "We also stress-tested the models under simulated adverse weather. Our most striking finding was the motion blur "
        "asymmetry: directional blur wiped out 87% of crack detections because hairline edges are smeared, but pothole "
        "recall dropped by only 18.7%! Potholes are 2D area depressions and maintain luminance contrast even when blurred. "
        "Conversely, heavy fog destroyed pothole detection because atmospheric scattering washes out the depth shadows inside the hole."
    )

    # ==================== Slide 11: Failure Analysis ====================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Qualitative Failure Mode Taxonomy & Academic Root Causes")
    
    fig_fail = root_dir / "results" / "evaluations" / "09_failure_cases_analysis.png"
    if fig_fail.exists():
        slide11.shapes.add_picture(str(fig_fail), Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    box11 = slide11.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.0))
    tf11 = box11.text_frame
    tf11.word_wrap = True
    
    findings11 = [
        ("Mode 1: Road Stripe False Positives", "High-contrast yellow/white US lane markings mimic crack topologies, triggering false positives."),
        ("Mode 2: Motion Blur Smearing", "Vehicle vibration smears high-frequency linear crack edges below stride-32 feature map resolution."),
        ("Mode 3: Low-Light Shadow Loss", "Nocturnal surveys without street illumination eliminate depth shadows, causing pothole false negatives."),
        ("Mode 4: Atmospheric Fog Washout", "Scattering suppresses dynamic range, merging weak alligator cracking into the roadway background.")
    ]
    for title, desc in findings11:
        p_t = tf11.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(6)
        
        p_d = tf11.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(4)
        
    slide11.notes_slide.notes_text_frame.text = (
        "Here are four representative failure cases that provide deep academic insights: First, painted US highway striping "
        "triggers false alarms because high-contrast lines mimic cracks. Second, motion blur smears high frequencies below our "
        "stride-32 feature maps. Third, unlit night roads mask pothole contours. Fourth, heavy fog squashes dynamic range. "
        "Understanding these physical failure modes is vital for building next-generation robust detectors."
    )

    # ==================== Slide 12: Priority Rules ====================
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "Automated Road Maintenance Prioritization Heuristics (scripts/priority.py)")
    
    box12 = slide12.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf12 = box12.text_frame
    tf12.word_wrap = True
    
    points12 = [
        ("Bridging Deep Learning to Municipal Operations", "Raw bounding boxes are insufficient for maintenance dispatch. We developed scripts/priority.py to compute deterministic, actionable repair priority tiers."),
        ("Multi-Factor Risk Scoring Formula", "Score = Multiplicity * Sum(Class_Hazard_Weight * Confidence * Area_Penalty)."),
        ("Asymmetric Hazard Weights", "D40 Pothole = 10.0 (high safety hazard), D20 Alligator = 6.5 (structural failure), D10 Transverse = 3.5, D00 Longitudinal = 2.5."),
        ("Actionable Priority Tiers", "HIGH (Emergency patch in 24-48h), MEDIUM (30-day resurfacing queue), LOW (Preventive crack seal), NONE (Sound pavement).")
    ]
    for title, desc in points12:
        p_t = tf12.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(8)
        
        p_d = tf12.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide12.notes_slide.notes_text_frame.text = (
        "To turn bounding boxes into municipal action, we engineered an automated triage engine. The formula incorporates "
        "safety hazard weights—giving potholes ten times the weight of hairline cracks—as well as area penalties and cluster "
        "multipliers. It outputs four clear operational tiers: HIGH priority for emergency 24-hour cold-patching, MEDIUM for "
        "scheduled resurfacing, LOW for preventive sealing, and NONE for sound roads."
    )

    # ==================== Slide 13: Demo Video ====================
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "End-to-End Live Video Demonstration (demo/output.mp4)")
    
    box13 = slide13.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf13 = box13.text_frame
    tf13.word_wrap = True
    
    points13 = [
        ("Edge Dashcam Video Processing", "Rendered 120-frame real-time demonstration video at 15 FPS (demo/output.mp4) processing raw road driving footage."),
        ("Real-Time Telemetry Heads-Up Display (HUD)", "Top banner dynamically renders Priority Alert Badges (HIGH in bright red, MEDIUM in amber, LOW in green)."),
        ("Defect Breakdown & Action Order", "Displays real-time counts for D40, D20, D10, D00 alongside actionable maintenance dispatch recommendations."),
        ("Performance Badge", "Bottom-right badge displays hardware platform (RTX 5070 GPU), live forward latency, and instantaneous FPS.")
    ]
    for title, desc in points13:
        p_t = tf13.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = STEEL_BLUE
        p_t.space_before = Pt(8)
        
        p_d = tf13.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide13.notes_slide.notes_text_frame.text = (
        "Here is our end-to-end demonstration video processed by demo.py. As the vehicle travels, the model identifies "
        "each distress and immediately updates the top HUD banner. Notice when multiple potholes appear, the badge instantly "
        "turns bright red, logging a HIGH priority alert with emergency dispatch recommendations, while updating real-time "
        "latency at over 38 FPS."
    )

    # ==================== Slide 14: Hardware Latency ====================
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "Hardware Runtime Latency & Throughput Profiling (RTX 5070 GPU)")
    
    fig_rt = root_dir / "results" / "evaluations" / "10_runtime_latency_breakdown.png"
    if fig_rt.exists():
        slide14.shapes.add_picture(str(fig_rt), Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    box14 = slide14.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.0))
    tf14 = box14.text_frame
    tf14.word_wrap = True
    
    findings14 = [
        ("CUDA Event Synchronization", "Profiled 100 consecutive runs with 30 warmup iterations for microsecond-precise GPU timing."),
        ("Pure Inference Latency", "YOLO11n: 19.51 ms (51.2 FPS) | YOLO11s: 23.50 ms (42.6 FPS)."),
        ("Full End-to-End Pipeline Latency", "YOLO11n: 21.99 ms (45.5 FPS) | YOLO11s: 25.89 ms (38.6 FPS)."),
        ("Guaranteed P95 Latency Bounds", "P95 E2E latency is 28.72 ms (YOLO11n) and 29.61 ms (YOLO11s)—strictly below the 33.3 ms (30 FPS) ceiling."),
        ("Ultra-Lean VRAM Footprint", "Peak VRAM is only 77.6 MB for YOLO11n and 137.9 MB for YOLO11s (<2% of 8 GB capacity).")
    ]
    for title, desc in findings14:
        p_t = tf14.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(4)
        
        p_d = tf14.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(11.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(3)
        
    slide14.notes_slide.notes_text_frame.text = (
        "On our test hardware, microsecond-accurate CUDA event profiling confirmed that our 95th percentile end-to-end latency "
        "is 29.6 milliseconds, guaranteeing zero dropped frames at 30 FPS. The memory footprint is only 138 megabytes. "
        "We recommend deploying YOLO11s on NVIDIA Jetson Orin Nano edge units inside municipal vehicles, eliminating monthly "
        "5G data streaming costs and allowing continuous offline surveys in rural areas."
    )

    # ==================== Slide 15: Stakeholder Recommendations ====================
    slide15 = prs.slides.add_slide(blank_layout)
    add_header(slide15, "Municipal Hardware Procurement & Deployment Architecture")
    
    box15 = slide15.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf15 = box15.text_frame
    tf15.word_wrap = True
    
    recs = [
        ("Recommended Primary Edge Platform: NVIDIA Jetson Orin Nano (8GB)", "Equip municipal utility vans, sweepers, and buses with 7–15W edge AI boxes. Eliminates costly 4G/5G video streaming data plans ($100+/vehicle/month)."),
        ("Model Selection: YOLO11s is the Optimal Pareto Choice", "Delivers +31.2% higher mAP@50 and +19.2% higher pothole recall with only 3.9 ms added latency, sustaining 38.6 FPS with zero dropped frames."),
        ("Resilience in Cellular Dead Zones", "Edge inference operates 100% offline. Fleet vehicles queue low-bandwidth JSON metadata, uploading sync packages via depot Wi-Fi or cellular bursts."),
        ("Human-in-the-Loop Supervisory Review Gate", "HIGH-priority alerts flag road segments for rapid 5-second verification by a municipal road engineer prior to dispatching heavy paving machinery.")
    ]
    for title, desc in recs:
        p_t = tf15.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(8)
        
        p_d = tf15.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide15.notes_slide.notes_text_frame.text = (
        "Based on our empirical results, we recommend deploying YOLO11s on Jetson Orin Nano edge units. This setup "
        "preserves cellular bandwidth, functions reliably in rural dead zones, and provides superior pothole safety. "
        "We also recommend a human-in-the-loop review gate where road supervisors verify HIGH priority alerts before deploying "
        "costly paving machinery."
    )

    # ==================== Slide 16: Limitations & Future Work ====================
    slide16 = prs.slides.add_slide(blank_layout)
    add_header(slide16, "Methodological Limitations & Future Research Directions")
    
    box16 = slide16.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf16 = box16.text_frame
    tf16.word_wrap = True
    
    limits = [
        ("2D Bounding Box Area as a Metric Proxy", "Monocular pixel areas suffer from perspective foreshortening. Future work: integrate ground-plane homography transforms to calculate true physical distress area in square meters (m^2)."),
        ("Absence of 3D Pothole Depth Data", "Monocular RGB cannot measure vertical depth (z-axis). Future work: integrate stereo camera baseline or solid-state LiDAR to resolve shallow stain vs deep pothole ambiguity."),
        ("Closing the Cross-Country Domain Gap", "Evaluations were strictly zero-shot. Future work: implement Unsupervised Domain Adaptation (UDA) or few-shot target domain fine-tuning (50 annotated target images)."),
        ("Micro-Crack Slicing (SAHI)", "Integrate Slicing Aided Hyper Inference (SAHI) during stationary or low-speed patrols to detect 1-pixel hairline cracks.")
    ]
    for title, desc in limits:
        p_t = tf16.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.space_before = Pt(8)
        
        p_d = tf16.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide16.notes_slide.notes_text_frame.text = (
        "In accordance with academic rigor, we acknowledge our limitations: 2D bounding boxes are subject to perspective "
        "distortion, and monocular cameras cannot estimate 3D pothole depth. In future work, we plan to implement homography "
        "transforms for true surface area calculation, stereo depth estimation, and unsupervised domain adaptation to bridge "
        "the cross-continental domain gap."
    )

    # ==================== Slide 17: Conclusion ====================
    slide17 = prs.slides.add_slide(blank_layout)
    add_header(slide17, "Project Summary & Key Takeaways")
    
    box17 = slide17.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf17 = box17.text_frame
    tf17.word_wrap = True
    
    conclusions = [
        ("1. End-to-End Operational Solution", "Built an end-to-end smart city road inspection pipeline from raw dashcam ingestion to automated priority dispatch."),
        ("2. Real-Time Pareto-Optimal Performance", "YOLO11s delivers 0.2846 mAP@50 (+31.2% over baseline) and 39.74% pothole recall (+19.2%) at 38.6 FPS on RTX 5070 GPU."),
        ("3. Scientific Domain Insights", "Discovered that longitudinal cracks (D00) possess high cross-continental transferability (-4.98% decay), while motion blur preserves pothole recall (-18.7% decay)."),
        ("4. Deployment-Ready & Cost-Effective", "Eliminates $300k inspection vehicles, empowering municipal fleets to conduct city-wide automated pavement triage.")
    ]
    for title, desc in conclusions:
        p_t = tf17.add_paragraph()
        p_t.text = f"{title}"
        p_t.font.name = "Arial"
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = STEEL_BLUE
        p_t.space_before = Pt(8)
        
        p_d = tf17.add_paragraph()
        p_d.text = f"{desc}"
        p_d.font.name = "Arial"
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = DARK_GRAY
        p_d.space_after = Pt(6)
        
    slide17.notes_slide.notes_text_frame.text = (
        "In conclusion, our project bridges deep learning research and municipal public works. We deliver an end-to-end "
        "edge pipeline running at 38.6 FPS, achieving a 31% mAP improvement, and prioritizing hazardous potholes to protect "
        "urban safety. In future work, we plan to integrate homography transforms for true physical area measurement and domain "
        "adaptation. Thank you for your time, and we now welcome any questions."
    )

    # ==================== Slide 18: Q&A Slide ====================
    slide18 = prs.slides.add_slide(blank_layout)
    bg18 = slide18.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg18.fill.solid()
    bg18.fill.fore_color.rgb = NAVY
    bg18.line.fill.background()
    
    t_box18 = slide18.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(10.9), Inches(3.0))
    tf18 = t_box18.text_frame
    tf18.word_wrap = True
    
    p0 = tf18.paragraphs[0]
    p0.text = "THANK YOU!"
    p0.font.name = "Arial"
    p0.font.size = Pt(44)
    p0.font.bold = True
    p0.font.color.rgb = WHITE
    p0.space_after = Pt(14)
    
    p1 = tf18.add_paragraph()
    p1.text = "Questions & Academic Discussion"
    p1.font.name = "Arial"
    p1.font.size = Pt(24)
    p1.font.color.rgb = RGBColor(190, 215, 245)
    p1.space_after = Pt(20)
    
    p2 = tf18.add_paragraph()
    p2.text = "Lingnan University CDS540 Computer Vision | Group Direction 2\nCode, Datasets & Demo Video Available in Project Repository"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(220, 220, 220)
    
    prs.save(str(pptx_path))
    print(f"\n[SUCCESS] Native PowerPoint Presentation Deck created at: {pptx_path}")
    print(f"Total Slides: {len(prs.slides)} slides (Widescreen 16:9, fully populated presenter notes)")
    return pptx_path


if __name__ == '__main__':
    create_presentation()
