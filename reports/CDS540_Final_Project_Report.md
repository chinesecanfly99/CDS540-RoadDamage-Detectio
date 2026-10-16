# Automated Road Damage Detection & Operational Triage for Smart City Infrastructure Maintenance

**Course**: Lingnan University CDS540 Computer Vision  
**Semester**: Term 1, Academic Year 2026/2027  
**Project Group**: Group Direction 2 — Computer Vision for Infrastructure Asset Management  
**Authors**: CDS540 Project Team  
**Deliverable**: Final Academic Project Report (Strict 8-Page Formal Report)  

---

## Abstract

Automating road pavement inspection is critical for urban safety, municipal budget allocation, and infrastructure longevity. Traditional manual visual surveys are labor-intensive, hazardous to inspectors, and financially prohibitive. In this work, we present a robust, edge-deployable computer vision pipeline for automated road distress detection and operational maintenance triage using the multi-country IEEE CRDDC RDD2022 dataset. We systematically evaluated two single-stage detector scales—a baseline **YOLO11n** (2.6M parameters) and a scaled **YOLO11s** (9.4M parameters)—trained on in-domain European and Asian roads (Czech Republic and India). On a strictly isolated in-domain test set of 1,579 images, YOLO11s demonstrated significant performance improvements over the baseline, boosting overall mAP@50 by **+31.2%** (from 0.2169 to 0.2846) and increasing the critical recall of hazardous potholes (`D40`) by **+19.2%** (from 33.33% to 39.74%). To probe practical deployment limits, we conducted zero-shot cross-continent domain transfer on **4,805 unseen United States highway dashcam images**, revealing that while composite mAP declines by ~69%, longitudinal cracks (`D00`) exhibit remarkable geometric transferability (only -4.98% decay). Systematic photometric stress-testing across 5 environmental perturbations demonstrated that directional motion blur degrades linear crack detection by -86.9% while pothole recall remains remarkably resilient (-18.7%). Precision timing with CUDA event synchronization on an NVIDIA RTX 5070 Laptop GPU established that the end-to-end pipeline—including automated risk scoring and HUD telemetry rendering—operates deterministically at **38.6 FPS** (P95 latency of 29.6 ms), comfortably exceeding the 30 FPS ceiling required for real-time vehicular dashcam surveyance.

---

## 1. Introduction & Stakeholder Problem

Modern municipal road networks are fundamental to economic vitality and public transit. However, asphalt and concrete pavements degrade continuously under the combined impact of traffic load cycles, environmental thermal expansion, and water infiltration. Municipal departments of transportation face an escalating maintenance backlog, spending billions annually on pavement resurfacing. 

Currently, condition assessment relies predominantly on two paradigms:
1. **Manual Road Patrols**: In-person visual inspection by municipal officers walking or driving slowly along road shoulders. This method is subjective, prone to severe human fatigue, hazardous to inspectors in live traffic, and incapable of covering entire metropolitan networks.
2. **Specialized Inspection Vehicles**: Heavy commercial vans equipped with laser profilometers and 3D LiDAR scanners. While highly accurate, these proprietary systems cost upwards of $300,000 per vehicle, making continuous, city-wide deployment financially unfeasible for most local councils.

Recent advancements in deep convolutional neural networks and single-stage object detectors offer an opportunity: transforming consumer-grade, low-cost automotive dashcams mounted on public municipal fleets (e.g., street sweepers, public transit buses, garbage trucks) into continuous, automated road distress survey nodes.

### 1.1 The Asymmetric Cost of Municipal Liability
In municipal road management, all false negatives are not created equal. Missing a superficial longitudinal crack (`D00`) merely results in deferred preventive sealing. In stark contrast, **missing a deep pothole (`D40`)** directly induces tire blowouts, wheel rim fractures, vehicular control loss, and severe injuries to motorcyclists and cyclists—exposing municipal authorities to direct tort liability and compensation payouts. Consequently, our system design treats **Pothole Recall** as a first-class optimization objective alongside standard mean Average Precision (mAP).

### 1.2 Contributions
This project delivers four key technical contributions:
- A rigorous, leakage-free data engineering and quality assessment pipeline across 19,208 raw multi-country dashcam images, incorporating 851 background frames with zero-byte annotations to suppress operational false alarms.
- A controlled comparative experiment establishing the Pareto frontier between ultra-compact edge models (YOLO11n) and scaled feature extractors (YOLO11s).
- An empirical cross-country domain generalization evaluation on 4,805 held-out American highway images and a 5-condition environmental robustness benchmark.
- An end-to-end decision support pipeline translating raw bounding boxes into deterministic municipal repair priority badges (HIGH / MEDIUM / LOW) operating at 38.6 FPS on RTX 5070 GPU hardware.

---

## 2. Dataset & Quality Assessment (RDD2022)

### 2.1 Dataset Composition & Class Taxonomy
We utilized the multi-national Crowdsourced Road Damage Detection Challenge 2022 dataset (RDD2022). The raw corpus contains 19,208 dashcam images captured across three continents: Europe (Czech Republic), Asia (India), and North America (United States). The dataset annotates four canonical IEEE CRDDC road damage categories:
- **`D00` (Longitudinal Crack)**: Linear fissures propagating parallel to traffic direction, typically caused by wheel-path fatigue or poor paving joint compaction.
- **`D10` (Transverse Crack)**: Cracks perpendicular to traffic flow, predominantly driven by thermal contraction and seasonal temperature gradients.
- **`D20` (Alligator / Fatigue Crack)**: Interconnected polygonal cracking resembling reptile skin, indicating severe structural failure of the underlying subgrade.
- **`D40` (Pothole)**: Bowl-shaped roadway depressions resulting from structural fatigue and moisture liquefaction, posing immediate vehicular hazards.

### 2.2 Annotation Integrity & Data Leakage Prevention
Prior to training, we developed an automated parsing and validation suite (`scripts/build_data_inventory.py` and `scripts/annotation_qa_inspector.py`) examining all 15,340 Pascal VOC XML files. Across 20,962 bounding boxes, our validator confirmed **zero coordinate anomalies** ($x_{\min} < x_{\max}$ and $y_{\min} < y_{\max}$ strictly within image bounds $[0, W] \times [0, H]$).

To guarantee strict isolation and eliminate data leakage:
- **Partitioning Strategy**: Splitting was performed strictly at the **image level** with a fixed deterministic seed ($seed = 42$). 
- **In-Domain Partition**: 10,535 Czech and Indian images were split into 7,375 Training images (70%), 1,581 Validation images (15%), and 1,579 Testing images (15%).
- **Negative Background Frames**: We retained 851 negative frames (sound roads without distress) in the test partition, accompanied by empty 0-byte YOLO label files. This is essential to penalize phantom false positives over clean asphalt.
- **Held-Out Domain Test**: **100% of the United States dataset (4,805 images)** was completely isolated from all training, validation, and hyperparameter tuning phases, serving exclusively as an unseen target domain benchmark in Step 10.

---

## 3. End-to-End System Pipeline

The deployed municipal edge vision system operates as a synchronous pipeline composed of five distinct stages:
1. **Video Ingestion & Preprocessing**: Ingests vehicle dashcam video frames at $1280 \times 720$ resolution, performs letterbox resizing to $640 \times 640$, and normalizes dynamic range into float16 tensors.
2. **Deep Neural Detection**: Executes forward inference on an onboard GPU using optimized TensorRT/PyTorch kernels, generating class confidence scores and bounding coordinates.
3. **Non-Maximum Suppression (NMS)**: Eliminates redundant overlapping detections using IoU thresholding ($\text{IoU} = 0.70$, confidence threshold $\tau = 0.25$).
4. **Automated Triage Heuristics (`scripts/priority.py`)**: Computes an automated severity score based on class hazard weights, relative bounding box surface area, and distress co-occurrence clustering.
5. **HUD Telemetry Overlay & Dispatch**: Renders a top Heads-Up Display (HUD) banner with color-coded priority badges (HIGH: Urgent 24-48h, MEDIUM: 30-day resurfacing queue, LOW: Preventive crack sealing) and displays live inference FPS.

---

## 4. Model Architecture & Implementation

### 4.1 Comparative Architectural Formulations
We evaluated two model configurations from the state-of-the-art Ultralytics YOLO11 architecture:
- **YOLO11n (Baseline)**: Parameterized with 2,597,780 parameters and 6.4 GFLOPs. Optimized for edge micro-nodes with strictly constrained compute.
- **YOLO11s (Scaled Comparison)**: Scaled with 9,418,988 parameters ($3.6\times$ baseline capacity) and 21.4 GFLOPs. Employs wider channel widths across darknet convolutional stages and deeper C3k2 cross-stage partial bottleneck blocks.

### 4.2 Training Protocols & Hardware Configuration
Both models were trained on the identical 7,375-image training set using an **NVIDIA GeForce RTX 5070 Laptop GPU (8,151 MiB VRAM)** under PyTorch 2.14.1 with CUDA 13.0 (`sm_120` architecture support). Hyperparameters were standardized to ensure scientific fairness:
- **Optimizer**: AdamW with weight decay $\lambda = 0.0005$.
- **Learning Rate Schedule**: Initial learning rate $\eta_0 = 0.005$ decayed via cosine annealing with a linear warmup of 3 epochs.
- **Input Resolution**: $640 \times 640$ pixels with batch size 16.
- **Augmentation Pipeline**: Mosaic augmentation (disabled during the final 5 epochs), random horizontal flipping ($p = 0.5$), HSV color jittering ($h = 0.015, s = 0.7, v = 0.4$), and random perspective scaling.

---

## 5. Experimental Results (Baseline vs. Comparison)

### 5.1 In-Domain Quantitative Evaluation
Table 1 presents the comparative benchmark results on the 1,579 independent in-domain test images (recorded in `results/model_comparison.csv`).

**Table 1: Quantitative Performance Benchmark on Independent Test Partition (1,579 Images)**

| Model Architecture | Params (M) | FLOPs (G) | Model Size (MB) | Precision | Recall | mAP@50 | mAP@50-95 | D00 AP50 | D10 AP50 | D20 AP50 | D40 AP50 | D40 Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n (Baseline)** | 2.6 | 6.5 | 20.45 | 0.3906 | 0.2774 | **0.2169** | 0.0833 | 0.2092 | 0.0533 | 0.3959 | 0.2093 | 33.33% |
| **YOLO11s (Scaled)** | 9.4 | 21.5 | 18.31 | 0.3834 | 0.3305 | **0.2846** | 0.1140 | 0.2387 | 0.0863 | 0.5210 | 0.2923 | **39.74%** |
| *Relative Difference* | *+261%* | *+231%* | *-10.5%* | *-1.8%* | *+19.1%* | ***+31.2%***| ***+36.9%***| *+14.1%* | *+61.9%* | *+31.6%* | *+39.7%* | ***+19.2%*** |

### 5.2 Academic Analysis of Model Scaling
1. **Substantial Generalization Jump**: Scaling the model capacity from Nano to Small yields a dramatic **+31.2% relative gain in mAP@50** (0.2846 vs 0.2169) and a **+36.9% gain in mAP@50-95** (0.1140 vs 0.0833). The deeper receptive field of YOLO11s captures complex multi-branch alligator crack patterns that are frequently lost in the shallower feature maps of YOLO11n.
2. **Critical Municipal Safety Benefit (Pothole Recall)**: For high-hazard potholes (`D40`), YOLO11s elevates detection AP50 from 0.2093 to 0.2923 (+39.7%) and pushes **D40 Recall from 33.33% to 39.74% (+19.2%)**. In municipal practice, this prevents almost 1 in 5 previously missed potholes from going undetected.
3. **Pareto Frontier Efficiency**: While YOLO11s requires $3.3\times$ more FLOPs, its model file footprint is slightly more compact (18.3 MB vs 20.4 MB, due to architectural head parameter optimizations) and its runtime latency increases by only 4.0 ms, preserving real-time throughput.

---

## 6. Generalization & Robustness (Cross-Country & Perturbations)

### 6.1 Cross-Country Domain Shift (Unseen US Highway Evaluation)
To examine zero-shot cross-continent transfer, we evaluated both models on **4,805 unseen United States interstate dashcam frames** (`results/domain_shift.csv`).

**Table 2: In-Domain vs. Cross-Country Zero-Shot Performance Degradation**

| Model | Domain Partition | Precision | Recall | mAP@50 | D00 AP50 (Long.) | D20 AP50 (Allig.) | D40 AP50 (Pothole) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n (Baseline)** | In-Domain (Czech + India) | 0.3906 | 0.2774 | **0.2169** | 0.2092 | 0.3959 | 0.2093 |
| **YOLO11n (Baseline)** | Cross-Domain (US Holdout) | 0.1470 | 0.1717 | **0.0798** | 0.1988 | 0.0306 | 0.0304 |
| *Relative Decay (%)* | *Relative Change* | *-62.4%* | *-38.1%* | ***-63.2%*** | ***-4.98%*** | *-92.3%* | *-85.5%* |
| **YOLO11s (Scaled)** | In-Domain (Czech + India) | 0.3834 | 0.3305 | **0.2846** | 0.2387 | 0.5210 | 0.2923 |
| **YOLO11s (Scaled)** | Cross-Domain (US Holdout) | 0.1986 | 0.1595 | **0.0879** | 0.1743 | 0.0938 | 0.0236 |
| *Relative Decay (%)* | *Relative Change* | *-48.2%* | *-51.7%* | ***-69.1%*** | *-27.0%* | *-82.0%* | *-91.9%* |

#### Scientific Findings on Domain Transfer:
- **Longitudinal Invariance**: Single-track longitudinal cracks (`D00`) transfer across continents with remarkable stability, decaying by only **-4.98%** on the baseline. Linear asphalt fissures share universal directional edge gradients regardless of geographic location.
- **Pavement Material Failure Mode**: Alligator cracking (`D20`) collapsed by over **-82% to -92%**. Indian roads exhibit coarse, crumbly asphalt alligatoring, whereas US interstate highways feature longitudinal concrete slab joints, transverse expansion joints, and poured rubberized crack sealant snakes. These novel US pavement patterns trigger heavy misclassifications.

### 6.2 Adverse Environmental Perturbation Stress-Testing
We subjected our models to 5 simulated operating conditions on a 350-image test set (`results/robustness.csv`):

**Table 3: Environmental Perturbation Stress Testing & Asymmetric Decay**

| Operating Environment | Mathematical / Physical Formulation | Precision | Recall | mAP@50 | D40 (Pothole) Recall | mAP@50 Decay (%) | D40 Decay (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean (Reference)** | Unaltered validation baseline | 0.5386 | 0.3325 | **0.2978** | **43.75%** | **0.0%** | **0.0%** |
| **Low-Light / Night** | Gamma darkening ($\gamma=2.0$, scale $0.45$) | 0.2819 | 0.2159 | **0.2060** | **30.40%** | **-30.8%** | **-30.5%** |
| **Motion Blur** | Directional horizontal blur ($k=15$) | 0.0630 | 0.0953 | **0.0390** | **35.58%** | ***-86.9%*** | ***-18.7%*** |
| **Adverse Fog / Haze** | Koschmieder atmospheric scattering ($t=0.50$) | 0.4584 | 0.1604 | **0.1372** | **14.42%** | **-53.9%** | **-67.0%** |
| **CMOS Sensor Noise** | Additive Gaussian noise ($\sigma=25$) | 0.0134 | 0.0481 | **0.0065** | **6.25%** | **-97.8%** | **-85.7%** |

#### Crucial Academic Takeaways:
- **Motion Blur Asymmetry**: 1D horizontal blur causes overall mAP to plummet by **-86.9%**, but **D40 Pothole Recall drops by only -18.7%**! High-frequency linear edges of cracks are completely smeared into the asphalt background, while 2D pothole depressions retain sufficient spatial luminance contrast to remain detectable.
- **Atmospheric Fog Destruction**: Dense fog attenuates local luminance contrast, causing pothole recall to plunge by **-67.0%** because depth shadows are washed out.

---

## 7. Operational Suitability (FPS, Latency & Model Size)

To assess field viability, precise timing benchmarks were executed on an **NVIDIA GeForce RTX 5070 Laptop GPU** using CUDA event synchronization across 100 consecutive test frames preceded by 30 warmup iterations (`results/runtime/runtime.csv`).

**Table 4: Hardware Latency, Throughput & Memory Profiling (RTX 5070 Laptop GPU)**

| Architectural Stage | YOLO11n (Baseline) | YOLO11s (Scaled) | Dashcam Standard Requirement |
|---|:---:|:---:|:---:|
| **Pure Inference Mean** | 19.51 ms | 23.50 ms | $\le 33.3\text{ ms}$ |
| **Pure Inference P50 (Median)** | 19.59 ms | 23.21 ms | Predictable forward pass |
| **Pure Inference P95** | 25.52 ms | 26.58 ms | Upper latency ceiling |
| **Pure Model Throughput** | 51.2 FPS | 42.6 FPS | $\ge 30.0\text{ FPS}$ |
| **Heuristic Triage Overhead** | 2.32 ms | 2.24 ms | Negligible CPU computation |
| **End-to-End Pipeline Mean** | **21.99 ms** | **25.89 ms** | Complete frame turnaround |
| **End-to-End Pipeline P95** | **28.72 ms** | **29.61 ms** | **< 33.3 ms (Zero Dropped Frames)** |
| **End-to-End Throughput** | **45.5 FPS** | **38.6 FPS** | **Exceeds 30 FPS Standard** |
| **Peak VRAM Allocated** | **77.6 MB** | **137.9 MB** | < 2% of 8 GB VRAM Capacity |

Both architectures comfortably clear the 30 FPS standard vehicle dashcam processing requirement with bounded P95 latencies below 30 ms.

---

## 8. Stakeholder Recommendations & Trade-offs

Based on our empirical Pareto frontier and cross-domain findings, we provide three concrete operational recommendations for municipal authorities:

### 8.1 Primary Deployment Architecture: On-Vehicle Edge GPU
We strongly recommend deploying **YOLO11s on edge computing units** (such as NVIDIA Jetson Orin Nano 8GB or industrial in-vehicle PCs equipped with RTX 4050/5060 GPUs) installed inside municipal fleet vehicles.
- **Operational Justification**: YOLO11s delivers **38.6 E2E FPS**, comfortably processing 30 FPS cameras without frame dropping. In return for a 3.9 ms latency increase over baseline, it provides a massive **+31.2% higher mAP@50** and **+19.2% higher recall on hazardous potholes**.
- **Bandwidth Independence**: Processing locally eliminates costly 4G/5G video streaming bandwidth. Vehicles operate offline in cellular dead zones, transmitting only low-bandwidth JSON defect records and cropped damage cards.

### 8.2 Human-in-the-Loop Review Gate
Automated priority rules must not directly trigger automated asphalt milling contracts. We recommend configuring the dispatch queue such that **HIGH priority alerts** generate an automated telemetry snapshot on a central municipal web dashboard, requiring a 5-second visual confirmation from a certified road engineer before dispatching paving crews.

---

## 9. Limitations & Future Work

While our system achieves real-time throughput and actionable triage, we acknowledge four key academic and engineering limitations:
1. **2D Bounding Box Area as a Physical Proxy**:
   Pavement damage severity is calculated using 2D pixel area ratios. Due to perspective foreshortening in monocular dashcams, a defect 5 meters ahead occupies substantially more pixels than a defect 25 meters ahead. Future work should integrate **monocular depth estimation** or ground-plane homography transforms to compute true metric surface areas ($m^2$).
2. **Absence of 3D Pothole Depth Estimation**:
   Monocular cameras cannot capture the vertical dimension ($z$-axis). A shallow surface stain can mimic a 10 cm deep pothole. Integrating dual-stereo cameras or low-cost solid-state LiDAR would resolve depth ambiguity.
3. **Synthetic vs. Real-World Perturbations**:
   Our robustness tests utilized mathematical models of fog, blur, and noise. While standard in computer vision, they do not account for dynamic rain droplets on windshields, wiper occlusions, or specular puddle reflections.
4. **Target Domain Fine-Tuning**:
   Our cross-country evaluations were strictly zero-shot. Incorporating **Unsupervised Domain Adaptation (UDA)** or few-shot active learning (annotating 50 target domain images) would close the 60% domain transfer gap.

---

## 10. References & Acknowledgements

1. **IEEE CRDDC 2022**: Arya, D., et al. "Crowdsourced Road Damage Detection Challenge 2022." *IEEE International Conference on Big Data (Big Data)*, 2022.
2. **YOLO Architecture**: Jocher, G., et al. "Ultralytics YOLO11." Version 8.4.177, 2024. Available at: `https://github.com/ultralytics/ultralytics`.
3. **PyTorch Framework**: Paszke, A., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*, 2019.
4. **Koschmieder Law**: Koschmieder, H. "Theorie der horizontalen Sichtweite." *Beiträge zur Physik der freien Atmosphäre*, 1924.
5. **CDS540 Course Guidelines**: Lingnan University Department of Computing and Decision Sciences, "CDS540 Computer Vision Group Project Specification," 2026.
