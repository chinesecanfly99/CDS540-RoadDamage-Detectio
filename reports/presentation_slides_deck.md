# CDS540 Final Group Presentation: Slide Deck & Speaker Notes

**Course**: Lingnan University CDS540 Computer Vision  
**Project**: Automated Road Damage Detection & Operational Triage for Smart City Infrastructure  
**Time Limit**: Strict 17 Minutes Presentation + 3 Minutes Q&A (20 Minutes Total)  
**Deliverable**: Task 14.2  

---

## ⏱️ Presentation Timing Architecture (17-Minute Budget)

| Section | Slide Numbers | Topics Covered | Target Duration | Running Time |
|---|:---:|---|:---:|:---:|
| **Part I: Problem & Context** | Slides 1–3 | Introduction, Stakeholder Liability, Asymmetric Cost | 2.5 min | 0:00 – 2:30 |
| **Part II: Data Engineering** | Slides 4–6 | RDD2022 Dataset, Coordinate QA, Leakage-Free Splitting | 2.5 min | 2:30 – 5:00 |
| **Part III: System Architecture** | Slide 7 | End-to-End Edge Pipeline & Decision Support | 1.5 min | 5:00 – 6:30 |
| **Part IV: Model Benchmarks** | Slides 8–10 | YOLO11n vs. YOLO11s, mAP & Pothole Recall, Pareto Frontier | 3.0 min | 6:30 – 9:30 |
| **Part V: Real-World Generalization** | Slides 11–13 | US Cross-Country Shift, Environmental Stress Tests, Failure Modes | 3.5 min | 9:30 – 13:00 |
| **Part VI: Prioritization & Demo** | Slides 14–15 | Automated Triage Heuristics, Video Demo Clip (`output.mp4`) | 2.0 min | 13:00 – 15:00 |
| **Part VII: Edge Deployment & Future** | Slides 16–17 | RTX 5070 Profiling, Jetson Hardware Selection, Limitations | 2.0 min | 15:00 – 17:00 |
| **Part VIII: Defense** | Slide 18 | Thank You & Transition to Q&A | — | 17:00 – 20:00 |

---

## Slide-by-Slide Layout & Verbatim Speaker Script

### Slide 1: Title Slide (Cover)
- **Title**: Automated Road Damage Detection & Operational Triage for Smart City Infrastructure
- **Subtitle**: Edge-Deployable Vision Intelligence for Municipal Fleet Dashcams
- **Metadata**: Lingnan University | CDS540 Computer Vision | Academic Year 2026/2027
- **Speakers**: Group Direction 2 — Team Members
- **Speaker Script (0:00 – 0:45)**:
  > *"Good morning Professor and fellow classmates. Today, our team is proud to present our project: Automated Road Damage Detection and Operational Triage for Smart City Infrastructure Maintenance. In this project, we demonstrate how edge-based computer vision transforms standard vehicle dashcams into an intelligent municipal survey network that detects hazardous roadway distress in real-time."*

---

### Slide 2: The Municipal Dilemma: Reactive vs. Proactive Maintenance
- **Visuals**: Split graphic comparing expensive $300k laser survey vans vs. dangerous manual walking inspections.
- **Key Bullet Points**:
  - Aging municipal road networks face severe deferred maintenance backlogs.
  - Traditional surveys: Manual foot patrols are slow, subjective, and hazardous in live traffic.
  - Specialized 3D LiDAR/laser vans cost >$300,000, preventing continuous city-wide monitoring.
  - Opportunity: Millions of municipal vehicles (buses, sweepers, garbage trucks) drive urban roads daily with low-cost dashcams.
- **Speaker Script (0:45 – 1:30)**:
  > *"Municipal road authorities face a massive dilemma. Roads degrade constantly under heavy traffic and rain, but traditional inspection methods cannot scale. Manual walking patrols put workers at risk on busy roads, while specialized LiDAR vans cost hundreds of thousands of dollars. Our objective is to leverage ubiquitous fleet dashcams—already driving through urban streets daily—to build an automated, edge-intelligent inspection system."*

---

### Slide 3: The Asymmetric Cost of Municipal Liability
- **Visuals**: Risk matrix diagram highlighting Potholes (`D40`) as High Hazard vs. Hairline Cracks (`D00`) as Low Hazard.
- **Key Bullet Points**:
  - **Not all false negatives are equal**:
    - Missing a fine crack (`D00`): Simply defers preventive bituminous sealing.
    - Missing a deep pothole (`D40`): Causes tire blowouts, rim damage, cyclist falls, and direct compensation claims.
  - **Core Engineering Mandate**: Maximize **Pothole Recall (`D40`)** alongside standard mAP metrics.
- **Speaker Script (1:30 – 2:30)**:
  > *"Crucially, from an engineering and public administration perspective, all false negatives are not equal. If the AI misses a hairline longitudinal crack, maintenance is merely deferred. But if it misses a deep pothole, a vehicle blows a tire or a cyclist crashes, directly resulting in municipal injury lawsuits. Therefore, throughout this project, we treat Pothole Recall as an indispensable safety objective, ensuring our models protect both city budgets and public safety."*

---

### Slide 4: Dataset & Canonical Taxonomy (IEEE CRDDC RDD2022)
- **Visuals**: 4 sample image cards representing the 4 IEEE CRDDC classes from `results/eda/`.
- **Key Bullet Points**:
  - Benchmark: IEEE CRDDC 2022 dataset spanning 19,208 raw dashcam images across 3 countries.
  - 4 Standard Damage Categories:
    - `D00`: Longitudinal Crack (wheel-path fatigue)
    - `D10`: Transverse Crack (thermal contraction)
    - `D20`: Alligator Crack (subgrade structural base failure)
    - `D40`: Pothole (severe impact depression)
- **Speaker Script (2:30 – 3:15)**:
  > *"We utilized the IEEE CRDDC RDD2022 dataset, comprising over 19,000 multi-country road dashcam frames. The challenge focuses on four canonical damage types: D00 longitudinal cracks, D10 transverse cracks, D20 alligator fatigue cracking, and D40 potholes. Each distress carries distinct physical causes and distinct maintenance urgency."*

---

### Slide 5: Rigorous Data Engineering & Quality Assessment
- **Visuals**: Flowchart from XML parsing to coordinate QA and anomaly detection.
- **Key Bullet Points**:
  - Developed custom parsing and QA inspector (`scripts/annotation_qa_inspector.py`).
  - Parsed 15,340 Pascal VOC XMLs across 20,962 bounding boxes: **0 coordinate anomalies** detected.
  - Identified physical distribution challenges: extreme class imbalance (cracks predominate, potholes rare) and high aspect ratios (narrow cracks only 2–4 pixels wide).
- **Speaker Script (3:15 – 4:00)**:
  > *"Before training any neural network, data quality must come first. We wrote an automated QA suite parsing all 15,340 Pascal VOC annotations across nearly 21,000 bounding boxes, confirming zero coordinate anomalies. Our exploratory data analysis revealed two key physical challenges: extreme class imbalance and micro-scale aspect ratios, where fine cracks are only a few pixels wide on compressed 640p frames."*

---

### Slide 6: Leakage-Free Partitioning & Negative Background Retention
- **Visuals**: Split pie chart: Train (7,375), Val (1,581), Test (1,579), Domain Test (4,805, 100% US held-out).
- **Key Bullet Points**:
  - Strict image-level splitting with fixed random seed ($seed = 42$). Zero overlap.
  - **Negative Background Frames**: Retained 851 clean road frames in test split with 0-byte label files to penalize phantom false positives.
  - **100% Held-Out Domain**: All 4,805 United States images held out completely for Step 10 cross-country generalization testing.
- **Speaker Script (4:00 – 5:00)**:
  > *"To ensure academic integrity and prevent data leakage, all splits were performed strictly at the image level with fixed random seeds. In addition, we deliberately preserved 851 negative frames—clean, undamaged roads—with empty label files. Without negative frames, object detectors produce rampant false alarms over asphalt markings in real deployments. Furthermore, 100% of the US highway data was held out for zero-shot testing."*

---

### Slide 7: End-to-End System Pipeline
- **Visuals**: System architecture diagram (`Dashcam Ingestion -> YOLO Inference -> NMS -> Priority Rules -> HUD Overlay -> Dispatch`).
- **Key Bullet Points**:
  - Real-time video ingestion with letterbox normalization ($640 \times 640$).
  - Deep neural inference on GPU Tensor Cores.
  - Automated triage heuristics: Class hazard weights $\times$ Area ratio $\times$ Multiplicity bonus.
  - Real-time Heads-Up Display (HUD) telemetry with color-coded maintenance dispatch orders.
- **Speaker Script (5:00 – 6:30)**:
  > *"Here is our end-to-end operational architecture. Incoming dashcam frames are letterboxed and passed through the neural network. Bounding boxes are filtered by NMS, and then immediately fed into our deterministic priority rules engine. The engine calculates an actionable risk score based on hazard weights, distress area, and clustering, rendering a live HUD telemetry dashboard for the driver or central dispatcher."*

---

### Slide 8: Model Architecture & Scaling Regimes
- **Visuals**: Diagram comparing YOLO11n (2.6M params, 6.5 GFLOPs) vs. YOLO11s (9.4M params, 21.5 GFLOPs).
- **Key Bullet Points**:
  - Single-stage architecture: C3k2 cross-stage partial bottlenecks + SPPF spatial pyramid pooling.
  - Task-aligned assigner + dynamic IoU regression loss.
  - Training on RTX 5070 Laptop GPU: Mixed-precision FP16, AdamW, cosine annealing lr.
- **Speaker Script (6:30 – 7:30)**:
  > *"We benchmarked two scales of the latest YOLO11 architecture: an ultra-compact YOLO11n baseline with 2.6M parameters, and a scaled YOLO11s with 9.4M parameters. Both models were trained on an NVIDIA RTX 5070 GPU using mixed-precision FP16, AdamW, and cosine annealing schedules."*

---

### Slide 9: In-Domain Test Benchmark Results
- **Visuals**: Embedded Table 1 from report + Per-Class AP bar chart from `05_accuracy_vs_speed_comparison.png`.
- **Key Bullet Points**:
  - **Overall Detection**: YOLO11s boosts mAP@50 from 0.2169 to 0.2846 (**+31.2% relative gain**).
  - **Pothole AP50**: Surges from 0.2093 to 0.2923 (**+39.7% relative gain**).
  - **Pothole Recall**: Climbs from 33.33% to **39.74% (+19.2% relative improvement)**.
- **Speaker Script (7:30 – 8:30)**:
  > *"When evaluated on our 1,579 independent test images, the scaled YOLO11s achieved a massive leap in accuracy: overall mAP@50 jumped by +31.2% to 0.2846. Even more significantly for municipal road departments, D40 pothole AP increased by nearly 40%, and pothole recall rose to 39.74%, directly addressing our asymmetric cost objective."*

---

### Slide 10: Accuracy vs. Speed Pareto Frontier
- **Visuals**: Left panel of `05_accuracy_vs_speed_comparison.png` showing Pareto curve with 30 FPS line.
- **Key Bullet Points**:
  - YOLO11n: 21.99 ms E2E latency (45.5 FPS) | 77.6 MB VRAM.
  - YOLO11s: 25.89 ms E2E latency (38.6 FPS) | 137.9 MB VRAM.
  - **Takeaway**: A minor 3.9 ms latency trade-off unlocks a +31.2% accuracy gain while keeping throughput safely above the 30 FPS ceiling.
- **Speaker Script (8:30 – 9:30)**:
  > *"Examining our Pareto frontier: while YOLO11n is blisteringly fast at 45.5 FPS, YOLO11s still achieves 38.6 FPS—well above the 30 FPS dashcam threshold. Trading just 3.9 milliseconds of compute provides a 31% accuracy boost, making YOLO11s the clear Pareto-optimal choice."*

---

### Slide 11: Cross-Country Domain Shift (Unseen US Highways)
- **Visuals**: Embedded Figure `06_cross_domain_generalization.png`.
- **Key Bullet Points**:
  - Evaluated zero-shot on 4,805 unseen US highway images.
  - Overall mAP@50 decayed from 0.285 to 0.088 (~69% drop).
  - **Key Discovery**: Longitudinal cracks (`D00`) decayed by only **-4.98%** (high geometric transferability).
  - Alligator cracks (`D20`) collapsed by **-82%** due to material differences: Indian asphalt distress vs. US concrete slab expansion joints.
- **Speaker Script (9:30 – 10:45)**:
  > *"Next, we tested real-world transferability by deploying our models zero-shot on 4,805 unseen US highway images. As expected, overall mAP dropped by roughly 69%. However, we discovered a remarkable phenomenon: single-track longitudinal cracks decayed by less than 5%! Cracks share universal linear geometry across continents. In contrast, alligator cracking collapsed by over 80% because US concrete slab joints and tar sealant snakes look fundamentally different from Indian asphalt degradation."*

---

### Slide 12: Environmental Robustness Stress-Testing
- **Visuals**: Embedded Table 3 + Mosaic Figure `07_robustness_perturbation_mosaic.png`.
- **Key Bullet Points**:
  - Evaluated 5 conditions: Clean, Low-Light, Motion Blur, Fog, Sensor Noise.
  - **Motion Blur Asymmetry**: mAP dropped -86.9% on fine cracks, but **D40 Pothole Recall dropped only -18.7%**!
  - **Fog**: Atmospheric scattering wiped out pothole shadow contrast, causing a -67.0% drop in pothole recall.
- **Speaker Script (10:45 – 11:45)**:
  > *"We also stress-tested the models under simulated adverse weather. Our most striking finding was the motion blur asymmetry: directional blur wiped out 87% of crack detections because hairline edges are smeared, but pothole recall dropped by only 18.7%! Potholes are 2D area depressions and maintain luminance contrast even when blurred. Conversely, heavy fog destroyed pothole detection because atmospheric scattering washes out the depth shadows inside the hole."*

---

### Slide 13: Failure Mode Taxonomy & Academic Root Cause Analysis
- **Visuals**: Embedded 4-quadrant Figure `09_failure_cases_analysis.png`.
- **Key Bullet Points**:
  - Mode 1: Painted yellow/white highway stripes misclassified as longitudinal cracks.
  - Mode 2: Vehicle motion smearing high-frequency crack edges.
  - Mode 3: Nighttime shadow loss masking pothole depth.
  - Mode 4: Fog-induced low contrast suppressing shallow CNN feature maps.
- **Speaker Script (11:45 – 13:00)**:
  > *"Here are four representative failure cases that provide deep academic insights: First, painted US highway striping triggers false alarms because high-contrast lines mimic cracks. Second, motion blur smears high frequencies below our stride-32 feature maps. Third, unlit night roads mask pothole contours. Fourth, heavy fog squashes dynamic range. Understanding these physical failure modes is vital for building next-generation robust detectors."*

---

### Slide 14: Automated Prioritization Heuristics (`scripts/priority.py`)
- **Visuals**: Triage formula box + Table of 4 Priority Badges (HIGH / MEDIUM / LOW / NONE).
- **Key Bullet Points**:
  - Formula: $S = M_{\text{multiplicity}} \times \sum (W_{\text{class}} \times \text{Conf} \times \gamma_{\text{size}})$.
  - Weights: $W(\text{D40})=10.0, W(\text{D20})=6.5, W(\text{D10})=3.5, W(\text{D00})=2.5$.
  - Outputs: **HIGH** (Emergency patch in 24-48h), **MEDIUM** (30-day mill/pave), **LOW** (Crack seal), **NONE** (Sound).
- **Speaker Script (13:00 – 14:00)**:
  > *"To turn bounding boxes into municipal action, we engineered an automated triage engine. The formula incorporates safety hazard weights—giving potholes ten times the weight of hairline cracks—as well as area penalties and cluster multipliers. It outputs four clear operational tiers: HIGH priority for emergency 24-hour cold-patching, MEDIUM for scheduled resurfacing, LOW for preventive sealing, and NONE for sound roads."*

---

### Slide 15: End-to-End Live Video Demonstration (`demo/output.mp4`)
- **Visuals**: Video playback or snapshot strip of `demo/output.mp4` showing live HUD banner and detections.
- **Key Bullet Points**:
  - Ingests road dashcam stream at 15–30 FPS.
  - Real-time overlay: Priority Alert Badge, Defect Count, Breakdown, Action Recommendation, Live FPS.
  - Real-world test: Successfully flags multi-pothole frames as HIGH and crack frames as LOW.
- **Speaker Script (14:00 – 15:00)**:
  > *"Here is our end-to-end demonstration video processed by demo.py. As the vehicle travels, the model identifies each distress and immediately updates the top HUD banner. Notice when multiple potholes appear, the badge instantly turns bright red, logging a HIGH priority alert with emergency dispatch recommendations, while updating real-time latency at over 38 FPS."*

---

### Slide 16: Hardware Runtime Profiling & Deployment Architecture
- **Visuals**: Embedded Figure `10_runtime_latency_breakdown.png` + Jetson Orin Nano hardware recommendation diagram.
- **Key Bullet Points**:
  - Measured on RTX 5070 GPU via CUDA Event sync: P95 latency is **29.6 ms** (< 33.3 ms budget).
  - Ultra-lean memory footprint: **137.9 MB peak VRAM** (< 2% capacity).
  - Recommended deployment: **NVIDIA Jetson Orin Nano (8GB)** onboard municipal fleet vehicles.
  - Eliminates cloud streaming bandwidth and operates 100% offline in rural cellular dead zones.
- **Speaker Script (15:00 – 16:00)**:
  > *"On our test hardware, microsecond-accurate CUDA event profiling confirmed that our 95th percentile end-to-end latency is 29.6 milliseconds, guaranteeing zero dropped frames at 30 FPS. The memory footprint is only 138 megabytes. We recommend deploying YOLO11s on NVIDIA Jetson Orin Nano edge units inside municipal vehicles, eliminating monthly 5G data streaming costs and allowing continuous offline surveys in rural areas."*

---

### Slide 17: Limitations, Future Work & Conclusion
- **Key Bullet Points**:
  - **Limitations**: 2D pixel area is a proxy subject to perspective foreshortening; lack of 3D depth.
  - **Future Work**: Integration of homography transforms for true metric area ($m^2$), stereo/LiDAR depth estimation, and Unsupervised Domain Adaptation (UDA).
  - **Conclusion**: A production-ready, edge-deployable, real-time computer vision system bridging deep learning research with municipal infrastructure management.
- **Speaker Script (16:00 – 17:00)**:
  > *"In conclusion, our project bridges deep learning research and municipal public works. We deliver an end-to-end edge pipeline running at 38.6 FPS, achieving a 31% mAP improvement, and prioritizing hazardous potholes to protect urban safety. In future work, we plan to integrate homography transforms for true physical area measurement and domain adaptation. Thank you for your time, and we now welcome any questions."*

---

### Slide 18: Q&A Slide
- **Title**: Thank You! Questions & Discussion
- **Metadata**: Lingnan University CDS540 Computer Vision | Project Deliverables Available on GitHub & Moodle
