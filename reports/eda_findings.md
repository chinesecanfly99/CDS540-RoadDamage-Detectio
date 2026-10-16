# Exploratory Data Analysis (EDA) Report: RDD2022 Benchmark

**Project**: Automated Road Damage Detection for Smart City Maintenance  
**Course**: CDS540 Computer Vision  
**Analyzed Countries**: Czech Republic, India, United States  
**Dataset Scope**: 15,340 Training Images | 20,962 Bounding Boxes  

---

## 1. Executive Summary & Key Dataset Statistics

| Metric | Czech Republic | India | United States | Total / Combined |
|:---|:---:|:---:|:---:|:---:|
| **Total Images** | 2,829 | 7,706 | 4,805 | **15,340** |
| **Damaged Images (Positive)** | 1,072 (37.9%) | 3,785 (49.1%) | 4,805 (100.0%) | **9,662 (63.0%)** |
| **Undamaged Images (Negative)**| 1,757 (62.1%) | 3,921 (50.9%) | 0 (0.0%) | **5,678 (37.0%)** |
| **Total Bounding Boxes** | 1,745 | 8,203 | 11,014 | **20,962** |
| **Mean Boxes per Positive Image**| 1.63 | 2.17 | 2.29 | **2.17** |
| **Input Resolution** | 600 × 600 | 720 × 720 | 640 × 640 | Highly Consistent |

---

## 2. Damage Category Distribution & Class Imbalance Analysis

### 2.1 Official 4-Class Breakdown
Across the 20,962 annotations, the distribution across the official IEEE CRDDC 2022 taxonomy is as follows:
* **D00 (Longitudinal Crack)**: **9,294 boxes (44.3%)** — Heavily dominant category, especially in the United States and Czech Republic.
* **D10 (Transverse Crack)**: **3,762 boxes (17.9%)** — Prevalent in the United States (3,295), but rare in India (68).
* **D40 (Pothole)**: **3,519 boxes (16.8%)** — Highly concentrated in India (3,187 out of 3,519), while severely scarce in the United States (135) and Czech (197).
* **D20 (Alligator Crack)**: **3,016 boxes (14.4%)** — Spread across all three countries.
* **Non-Standard Subtypes**: 1,371 boxes (6.5%) — Primarily historical sub-annotations from India (D44, D01, D43, D11, D50). These will be excluded or mapped to standard classes to maintain benchmark purity.

### 2.2 The Asymmetric Cost of Potholes (D40)
From an operational smart-city perspective, **missing a severe pothole (`D40`) introduces substantial safety hazards (tire blowouts, suspension failures, traffic accidents)** compared to false-detecting a hairline crack (`D00`).  
**Actionable takeaway**: Model evaluation must report **D40 Recall independently**, and training loss should avoid letting the abundant D00 class dominate gradient updates.

---

## 3. Cross-Domain Discrepancies (Domain Shift Evidence)

1. **Pavement Composition & Distress Profiles**:
   * **United States**: Dominated by well-maintained highways/urban asphalt with longitudinal and transverse expansion cracks (D00 + D10 account for >90% of instances).
   * **India**: Features unpaved/heavily degraded road networks with severe surface depressions and craters (D40 + D20 account for >63% of instances).
   * **Czech**: Moderate road aging, balanced between cracks and localized patching.
2. **Camera Sensor & Perspective Variations**:
   * Camera mounting height and windshield glare vary distinctly between the dashcams in the US and smartphone mounts in India/Czech.
   * **Academic Implication**: This provides the perfect empirical foundation for **Step 10 (Cross-Country Generalization Experiments)**: training on US/Czech and zero-shot evaluating on India will exhibit clear domain degradation.

---

## 4. Geometric Insights & Modeling Recommendations

1. **Aspect Ratio Diversity (Small & Slender Objects)**:
   * Median bounding box scale is only ~14% of the image size.
   * Longitudinal cracks (`D00`) and transverse cracks (`D10`) display extreme aspect ratios ($w/h > 4.0$ or $h/w > 4.0$).
   * *Recommendation*: Anchors/priors in YOLO and input resolution (`imgsz=640`) must preserve high-frequency features. Slicing Aided Hyper Inference (SAHI) or P2 shallow detection heads represent promising enhancement directions.
2. **Value of Negative Samples (5,678 Undamaged Images)**:
   * 37% of the total dataset consists of verified undamaged road scenes.
   * *Recommendation*: Retaining negative background images during YOLO training is mandatory to suppress false positive detections on road markings, shadows, and manholes.

---

## 5. Visual Artifacts Produced

The following publication-grade charts have been generated in `results/eda/`:
1. `01_class_distribution_overall.png`: Comprehensive category distribution with percentage shares.
2. `02_class_distribution_by_country.png`: Direct cross-country class breakdown revealing domain differences.
3. `03_positive_vs_negative_images.png`: Ratio of damaged vs undamaged background frames per nation.
4. `04_bbox_geometric_properties.png`: Aspect ratio histograms and relative scale distributions.
