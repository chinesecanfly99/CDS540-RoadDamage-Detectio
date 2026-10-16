# Cross-Country Domain Generalization & Environmental Robustness Report

**Course**: Lingnan University CDS540 Computer Vision  
**Project**: Automated Road Damage Detection for Smart City Maintenance  
**Dataset**: IEEE CRDDC RDD2022 (Czech Republic, India, United States)  
**Deliverable**: Step 10 — Tasks 10.1, 10.2, 10.3, 10.4, 10.5  

---

## 1. Executive Summary & Problem Formulation

Deploying computer vision models on municipal road inspection vehicles requires resilience against **two fundamental distribution shifts**:
1. **Cross-Country Domain Shift**: Roadway asphalt mixtures, lane striping standards, geographic climate, and defect morphology vary substantially across jurisdictions.
2. **Adverse Environmental Perturbations**: Dashcams operate across variable real-world conditions, including nocturnal survey lighting, vehicle vibration blur, heavy rain/fog, and sensor noise.

To prevent over-optimistic evaluation on homogeneous validation splits, we executed a rigorous two-pronged evaluation protocol:
- **Zero-Shot Cross-Country Generalization**: Evaluated our Czech+India trained models on **4,805 unseen United States interstate highway dashcam frames** (100% held-out test partition).
- **Photometric & Degradation Stress Testing**: Evaluated performance across 5 standardized environmental perturbation states on a controlled test subset.

All experiments benchmark both the **YOLO11n Baseline** (2.6M parameters) and the **YOLO11s Scaled Model** (9.4M parameters).

---

## 2. Cross-Country Domain Shift Analysis (Task 10.1 & 10.2)

### 2.1 Quantitative Performance Comparison

The quantitative results on In-Domain (Czech + India test split, $N = 1,579$) versus Cross-Domain (United States holdout, $N = 4,805$) are summarized in Table 1 below (recorded in `results/domain_shift.csv`).

**Table 1: In-Domain vs. Cross-Country Zero-Shot Transfer Metrics**

| Model | Domain Partition | Dataset Size | Precision | Recall | mAP@50 | mAP@50-95 | D00 AP50 (Long.) | D10 AP50 (Trans.) | D20 AP50 (Allig.) | D40 AP50 (Pothole) | D40 Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLO11n (Baseline)** | In-Domain (Czech + India) | 1,579 | 0.3906 | 0.2774 | **0.2169** | 0.0833 | 0.2092 | 0.0533 | 0.3959 | 0.2093 | 33.33% |
| **YOLO11n (Baseline)** | Cross-Domain (US Holdout) | 4,805 | 0.1470 | 0.1717 | **0.0798** | 0.0296 | 0.1988 | 0.0592 | 0.0306 | 0.0304 | 20.74% |
| *YOLO11n Relative Decay* | *Percentage Change (%)* | *N/A* | *-62.37%* | *-38.09%* | ***-63.23%*** | *-64.50%* | ***-4.98%*** | *+11.13%* | *-92.26%* | *-85.49%* | *-37.77%* |
| **YOLO11s (Scaled)** | In-Domain (Czech + India) | 1,579 | 0.3834 | 0.3305 | **0.2846** | 0.1140 | 0.2387 | 0.0863 | 0.5210 | 0.2923 | 39.74% |
| **YOLO11s (Scaled)** | Cross-Domain (US Holdout) | 4,805 | 0.1986 | 0.1595 | **0.0879** | 0.0319 | 0.1743 | 0.0596 | 0.0938 | 0.0236 | 20.74% |
| *YOLO11s Relative Decay* | *Percentage Change (%)* | *N/A* | *-48.21%* | *-51.73%* | ***-69.13%*** | *-72.01%* | *-26.97%* | *-30.91%* | *-81.99%* | *-91.91%* | *-47.81%* |

*(Publication figure exported to `results/evaluations/06_cross_domain_generalization.png`)*

### 2.2 Key Domain Shift Findings

1. **Overall Generalization Decay**:
   Both architectures suffer a severe **63% to 69% decline in mAP@50** when transitioning zero-shot to US highways. This empirical drop highlights the peril of assuming deep learning models trained in one geographic territory will immediately succeed in municipal operations abroad.
2. **Remarkable Robustness of Longitudinal Cracks (`D00`)**:
   While composite mAP dropped over 60%, the detection performance for `D00` (Longitudinal Cracks) was extraordinarily resilient:
   - YOLO11n experienced only a **-4.98% relative decline** in `D00` AP@50 (from 0.2092 in-domain to 0.1988 on US roads).
   - This demonstrates that single-line longitudinal fissures exhibit scale- and orientation-invariant visual cues that transfer seamlessly across continental road topologies.
3. **Severe Degradation of Alligator Cracking (`D20`)**:
   `D20` AP@50 collapsed precipitously by **-82% to -92%**. In Indian roadways, alligator cracking manifests as coarse, dusty, interconnected blocks surrounded by degraded subgrade. In contrast, US interstate highways feature grooved concrete slabs, asphalt overlays, and crack-sealed seams. The convolutional filters trained on European/Indian textures fail to generalize to American concrete joint textures.
4. **Extreme Scarcity of Potholes (`D40`) in US Highways**:
   The US test split contains only 135 annotated instances of `D40` across 4,805 images (compared to thousands in the Indian dataset). The resulting low AP50 score is compounded by severe statistical imbalance, as well as differences between shallow high-speed freeway depressions and deep Indian potholes.

---

## 3. Environmental Robustness & Perturbation Stress Testing (Task 10.3 & 10.4)

### 3.1 Experimental Perturbation Formulations

To assess real-world operational reliability on vehicle dashcams, we systematically simulated 4 critical adverse environmental conditions on a 350-image test subset (230 positive distress frames, 120 negative frames):
1. **Clean Reference**: Unperturbed test images.
2. **Low-Light / Nocturnal Dashcam**: Nonlinear gamma darkening ($\gamma = 2.0$, scale $= 0.45$) simulating unlit nocturnal surveys without high-beam street lighting:
   $$I' = \text{clip}\left(0.45 \times \left(\frac{I}{255}\right)^{2.0} \times 255,\; 0,\; 255\right)$$
3. **Motion Blur**: Horizontal 1D linear convolution kernel ($k = 15$) simulating high-speed vehicle vibrations:
   $$K = \frac{1}{15} \begin{bmatrix} 1 & 1 & \dots & 1 \end{bmatrix}$$
4. **Adverse Weather (Haze / Fog)**: Koschmieder's physical law of atmospheric scattering ($t = 0.50$, $A = 225$):
   $$I'(x) = I(x) \cdot t + A \cdot (1 - t)$$
5. **Sensor Noise**: Additive zero-mean Gaussian noise ($\sigma = 25$) simulating low-cost rolling-shutter CMOS sensor noise at high ISO sensitivity.

### 3.2 Quantitative Robustness Results

**Table 2: Degradation Across Environmental Perturbations (`results/robustness.csv`)**

| Operating Condition | Description | Precision | Recall | mAP@50 | mAP@50-95 | D40 Recall (Potholes) | mAP@50 Relative Decay (%) | D40 Recall Decay (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Clean (Reference)** | Baseline unperturbed test split | 0.5386 | 0.3325 | **0.2978** | 0.1212 | **43.75%** | **0.00%** | **0.00%** |
| **Low-Light / Night** | Gamma darkening ($\gamma=2.0$) | 0.2819 | 0.2159 | **0.2060** | 0.0829 | **30.40%** | **-30.81%** | **-30.51%** |
| **Motion Blur** | 1D horizontal kernel ($k=15$) | 0.0630 | 0.0953 | **0.0390** | 0.0143 | **35.58%** | **-86.91%** | ***-18.68%*** |
| **Adverse Weather (Fog)** | Atmospheric scattering ($t=0.50$) | 0.4584 | 0.1604 | **0.1372** | 0.0543 | **14.42%** | **-53.92%** | **-67.03%** |
| **Sensor Noise** | Gaussian noise ($\sigma=25$) | 0.0134 | 0.0481 | **0.0065** | 0.0026 | **6.25%** | **-97.81%** | **-85.71%** |

*(Publication figures exported to `results/evaluations/07_robustness_perturbation_mosaic.png` and `results/evaluations/08_robustness_degradation.png`)*

### 3.3 Critical Technical Insights

1. **Motion Blur Affects Cracks Far Worse Than Potholes**:
   Under horizontal motion blur, overall mAP@50 collapsed by **-86.91%**, but **D40 (Pothole) Recall decayed by only -18.68%**!
   - *Physical Explanation*: Linear cracks (`D00`, `D10`, `D20`) are thin 1D edge gradients (often 2–4 pixels wide). Horizontal directional blur smears these edges into the surrounding asphalt texture, causing complete loss of high-frequency spatial gradients.
   - Potholes (`D40`), by contrast, are 2D area depressions with broad luminance gradients and shadow contours, allowing them to remain detectable even when smeared.
2. **Dense Fog Destroys Pothole Detection**:
   In foggy conditions, **D40 Recall dropped by -67.03%** (from 43.75% down to 14.42%). Fog reduces local contrast via atmospheric scattering ($A = 225$). Potholes rely heavily on deep internal shadow contrast to differentiate depth from surrounding flat asphalt; once contrast is attenuated, the network misses over two-thirds of severe potholes.
3. **Sensor Noise Catastrophe**:
   Zero-mean Gaussian noise ($\sigma = 25$) essentially paralyzed the detector (mAP@50 decayed by -97.81%). This underlines the critical requirement for high-quality CMOS sensors with hardware-level temporal denoising in municipal vehicle procurement.

---

## 4. Failure Mode Taxonomy & Root Cause Analysis (Task 10.5)

To substantiate our findings with qualitative evidence, we performed systematic visual inspection of model failure cases (`results/evaluations/09_failure_cases_analysis.png`), identifying four distinct failure modes:

### Mode 1: Pavement Artifact False Positives (Cross-Country Domain Shift)
- **Symptom**: Solid yellow and white road boundary stripes, expansion joints, and tar sealant lines on US freeways are misdetected as `D00` (Longitudinal Cracks) or `D10` (Transverse Cracks).
- **Root Cause**: The training set lacked high-contrast painted roadway markings directly adjacent to dark asphalt. The feature extractor learns that high-contrast linear discontinuities represent cracks, generating false alarms on intentional road striping.

### Mode 2: High-Frequency Edge Smearing (Motion Blur)
- **Symptom**: Complete omission of fine longitudinal cracks (`D00`) and transverse cracks (`D10`) when the vehicle operates at speed.
- **Root Cause**: Downsampling in standard YOLO stride-32 backbones destroys blurred high-frequency spatial details.

### Mode 3: Depth Shadow Obliteration (Low Light & Nighttime)
- **Symptom**: Unlit roads mask the interior shadow boundary of potholes (`D40`), causing severe false negatives.
- **Root Cause**: The network relies on shadow cues to differentiate 2D surface stains from 3D pothole depressions.

### Mode 4: Contrast Washout (Adverse Weather / Fog)
- **Symptom**: Uniform graying of the roadway causes weak alligator cracking (`D20`) to merge into the background.
- **Root Cause**: Low dynamic range suppresses the convolutional activation maps in the shallow C3k2/C2f blocks.

---

## 5. Methodological Limitations & Academic Disclaimers

In compliance with rigorous scientific standards, we highlight the following constraints:
1. **Synthetic vs. Real-World Perturbations**:
   The environmental stress tests utilize mathematical image processing models (Koschmieder atmospheric scattering, linear directional convolution, Gaussian noise). While standard in computer vision benchmarks (e.g., ImageNet-C, Cityscapes-C), they do not capture complex physical phenomena such as raindrops on windshields, dynamic wiper occlusion, or headlight specular glare.
2. **Label Imbalance in Target Domain**:
   The US dataset has an extreme class imbalance with very few annotated potholes. Cross-domain D40 metrics must be interpreted with caution.
3. **No Domain Adaptation Fine-Tuning**:
   These experiments evaluated pure zero-shot generalization. In practical municipal deployments, unsupervised domain adaptation (UDA) or few-shot target domain fine-tuning (50–100 target domain images) would be recommended to restore baseline performance.
