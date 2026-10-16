# Lingnan University — CDS540 Computer Vision
## Group Project Topic Proposal (期末大作业课题开题报告)

**Institution**: Lingnan University | Master of Science in Data Science  
**Date**: October 16, 2026  

---

### 1. Project Selection (What We Selected)

- **Project Category**: Candidate Topic 3 — Object Detection and Image Segmentation (Direction 2).
- **Proposed Project Title (EN)**: Automated Road Damage Detection and Maintenance Prioritization Using Deep Learning (Operational Triage for Smart City Infrastructure).
- **Proposed Project Title (CN)**: 基于计算机视觉的智慧城市道路病害自动化检测与运维分级系统.
- **Core Objective**: This project aims to develop a deep learning-based computer vision prototype that automatically detects four canonical categories of road surface defects from dashcam images—longitudinal cracks (D00), transverse cracks (D10), alligator cracks (D20), and potholes (D40)—and assigns illustrative maintenance priority levels (High, Medium, or Low) to support road maintenance decision-making.

---

### 2. Problem Motivation (Why We Selected This Topic)

- **Cost-Effective & Safe Road Inspection**: Road surface defects compromise transportation safety and increase municipal maintenance costs. Traditional manual inspections are time-consuming, labor-intensive, and hazardous, whereas specialized laser inspection vans cost over $300,000 per unit. Automated analysis using existing municipal fleet dashcams (e.g., transit buses and sanitation sweepers) provides a practical and cost-effective complement to manual surveys.
- **Asymmetric Risk of Road Defects**: Different road damage types pose varying levels of risk. Missing superficial linear cracks (D00) merely delays routine sealing, whereas missing severe potholes (D40) can directly lead to vehicle blowouts, cyclist accidents, and municipal liability claims. Therefore, optimizing pothole detection recall is a key priority of our study.
- **From Detection to Decision Support**: Rather than stopping at isolated bounding boxes, combining defect detection with a simple maintenance priority triage indicator helps translate model outputs into intuitive, actionable information for maintenance planning.

---

### 3. Methodology & Implementation Plan (How We Will Do It)

- **Dataset Preparation**: We will use the multi-country Road Damage Dataset 2022 (RDD2022), which contains real road dashcam images from the Czech Republic, India, and the United States. Our data preparation includes dataset cleaning, XML bounding box validation, and a leakage-free train/validation/test split (70% / 15% / 15%), while retaining clean background road images to reduce false positives.
- **Model Training & Comparative Evaluation**: We will train and compare two object detection models—a lightweight baseline (YOLO11n) and a scaled detector (YOLO11s). Their performance will be evaluated using precision, recall (with particular attention to pothole recall), mean average precision (mAP), and real-time inference speed (FPS).
- **Generalization & Robustness Testing**: Subject to available time and computing resources, we will evaluate cross-country generalization on unseen United States road images and test model robustness under challenging visual conditions, including night/low-light, motion blur, fog/haze, and sensor noise.
- **Rule-Based Priority Triage & Deliverables**: We will implement a simple rule-based heuristic module to assign High, Medium, or Low maintenance priorities to detected road damage as illustrative decision-support indicators. The final deliverables will include a working prototype with visual HUD overlays of detected defects and priority levels, a comparative evaluation report, and final presentation slides.

---

### 4. Group Members & Task Modules

- **Group Leader**: Yangshang Li / 李杨尚 (Student ID: 5586669, Email: yangshangli@LN.hk)
- **Team Members**: 
  - Songyuan Wu / 吴松芫 (Student ID: 5550024)
  - Yijin Guo / 郭毅锦 (Student ID: 5585392)
  - Jingyi Song / 宋婧仪 (Student ID: 5554757)
  - Xiyan Huang / 黄习沿 (Student ID: 5594367)

#### Planned Task Modules:
- **Module 1 (Model Architecture)**: YOLO11n and YOLO11s training, hyperparameter tuning, and performance benchmarking.
- **Module 2 (Data Engineering)**: RDD2022 dataset cleaning, annotation validation, and dataset splitting.
- **Module 3 (Generalization & Robustness)**: Cross-country generalization evaluation and adverse image condition testing.
- **Module 4 (Operational Priority Triage)**: Rule-based maintenance priority assignment and prototype visualization demo.
- **Module 5 (Speed Profiling & Reporting)**: Inference speed evaluation, final report writing, and presentation preparation.

*Note: Individual module assignments will be finalized during group meetings and reported in the midterm milestone.*
