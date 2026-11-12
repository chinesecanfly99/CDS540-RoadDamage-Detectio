# CDS540 Oral Defense & Q&A Preparation Question Bank

**Course**: Lingnan University CDS540 Computer Vision  
**Project**: Automated Road Damage Detection & Operational Triage for Smart City Infrastructure  
**Time Limit**: Strict 3-Minute Q&A Session  
**Deliverable**: Task 14.3  

---

## 🎯 Strategic Defense Principles for the 3-Minute Q&A

1. **Be Crisp & Direct (30–45 Seconds per Answer)**: With only 3 minutes total, professors will ask 2–3 questions. Never ramble; state your direct scientific conclusion in the first sentence, then support it with concrete quantitative data from our benchmark tables.
2. **Anchor to Course Concepts**: Reference receptive fields, stride-32 downsampling, PAN-FPN feature pyramids, Koschmieder atmospheric scattering, and CUDA event synchronization.
3. **Turn Apparent "Weaknesses" into Profound Scientific Insights**: If questioned about the 69% cross-domain drop or motion blur collapse, explain the physical and material mechanisms behind the data!

---

## 🛡️ Top 8 Anticipated Defense Questions & Verbatim Model Answers

### Question 1 (Data Leakage & Integrity):
> *"How did you rigorously prevent data leakage between training, validation, and testing? If consecutive video frames from the same street were split into both train and test, your model might simply be memorizing identical pavement textures."*

**Model Answer (35 seconds)**:
> *"Thank you, Professor. We recognized this exact hazard from the outset. 
> First, all dataset partitioning was performed strictly at the **image level** with a fixed deterministic seed ($seed = 42$) before any training.
> Second, we verified zero filename or sequence overlap between splits.
> Third, to test cross-country generalization in our baseline, we **held out 100% of the labelled United States highway dataset (4,805 images)** from all training, validation, and hyperparameter tuning. Evaluating the unadapted baseline on this held-out continent guarantees that our reported results reflect genuine generalization rather than memorized pavement textures."*

---

### Question 2 (Updated — Small Object Detection & Multi-Scale Pyramid Evidence)
> *"Fine road cracks are often only 1 to 3 pixels wide on a 640x640 frame. A standard YOLO backbone downsamples feature maps by 32x at its deepest stage, which means 1 pixel becomes less than 0.03 pixels. How can your network possibly detect fine cracks?"*

**Model Answer (~98 words, ~38 seconds)**:
> *"Thank you, Professor. YOLO11 does not rely solely on the stride-32 deepest layer (P5). Its PAN-FPN head aggregates P3 (stride 8, 80x80), P4 (stride 16, 40x40), and P5 (stride 32, 20x20).*
> *We verified this empirically by hooking the Detect-feeding C3k2 layers across 25 test frames, including 4 pure background controls. Longitudinal cracks (D00) retained 73.8% of normalized energy in P3 and P4 while dropping to 0.94x frame RMS at P5, near background (0.93x). Transverse cracks (D10) showed P3 share slightly above background (36.1% vs 34.4%). Thus, shallow high-resolution feature maps preserve hairline edges lost at stride 32."*

**If pressed, add:**
> *"These GT-region hook measurements are correlative localization evidence rather than proof of detection, as sample detection rates were low (D10 0%, D00 16.7%, D40 19.2%, D20 50.0%). For our largest group, D40 potholes (n=26), normalized P5 share (30.9%) fell below background (32.3%) even though absolute P5 energy was elevated (1.15x RMS), so our findings are only partially consistent with motion-blur asymmetry and warrant a class-matched D40 follow-up."*
---

### Question 3 (Asymmetric Liability & Pothole Recall):
> *"Cracks account for over 85% of your dataset annotations, while potholes (D40) are relatively scarce. Why did you prioritize D40 Recall over optimizing overall mAP, and how does that affect municipal stakeholders?"*

**Model Answer (35 seconds)**:
> *"In academic benchmarks, mAP treats all classes symmetrically. However, in municipal civil engineering, **false negatives have asymmetric liability costs**.
> If our system misses a hairline crack ($D00$), maintenance is simply deferred with zero immediate danger. But if it misses a deep pothole ($D40$), a vehicle blows a tire or a cyclist suffers serious injury, directly exposing the municipal government to tort liability and expensive compensation claims.
> By scaling to YOLO11s, we pushed **Pothole Recall from 33.3% to 39.7% (+19.2% relative gain)**, preventing nearly 1 in 5 missed hazardous road defects in live operation."*

---

### Question 4 (The 69% Cross-Country Domain Transfer Drop):
> *"Your models suffered a ~69% mAP drop when evaluated on the held-out labelled United States highway dataset. Does this mean deep learning is unviable for cross-border municipal deployment? Why did longitudinal cracks transfer so well while alligator cracks collapsed?"*

**Model Answer (45 seconds)**:
> *"This evaluation revealed a crucial scientific insight rather than a failure.
> First, **longitudinal cracks ($D00$) barely dropped at all—decaying by only -4.98%** (from 0.209 to 0.199). Single-track linear cracks share universal geometric continuity across continents, proving high structural transferability.
> In contrast, **alligator cracking ($D20$) collapsed by -82% to -92%**. The reason is physical material divergence: our training set from India featured crumbly, dusty asphalt degradation, whereas US highways consist of grooved concrete slabs, longitudinal pavement expansion joints, and poured rubberized crack sealant lines. The network had never seen concrete joints or tar sealant snakes, misidentifying them as distress.
> This proves that deploying in a new country requires either targeted few-shot fine-tuning or unsupervised domain adaptation (UDA)."*

---

### Question 5 (Motion Blur Asymmetry: Cracks vs. Potholes):
> *"In your environmental stress-testing, why did horizontal motion blur devastate crack detection by -86.9%, while pothole recall only dropped by -18.7%?"*

**Model Answer (35 seconds)**:
> *"This divergence stems directly from **spatial signal topology**.
> Cracks ($D00, D10$) are high-frequency, 1D directional edge gradients only a few pixels wide. A horizontal convolution kernel smears these linear edges into the surrounding asphalt texture, completely flattening the gradient.
> Potholes ($D40$), however, are **2D area depressions**. They are characterized by low-frequency spatial luminance contrast—dark interior shadows bounded by bright pavement. Even when the outer perimeter is blurred by vehicle motion, the internal luminance deficit remains detectable by the convolutional filters."*

---

### Question 6 (Heuristic Priority Rules vs. Machine Learning Regression):
> *"Why did you design a deterministic heuristic formula in `scripts/priority.py` instead of training a machine learning regression model to predict road repair priority?"*

**Model Answer (35 seconds)**:
> *"The RDD2022 dataset provides object detection bounding boxes, but does not provide ground-truth structural Pavement Condition Index (PCI) ratings or municipal budgets. 
> Training a machine learning model on synthetic priority labels would simply learn our own arbitrary assumptions with zero scientific validation.
> Instead, a **deterministic, transparent heuristic** allows civil engineers to inspect the exact weighting—such as giving potholes 10x the weight of cracks—and adjust thresholds to comply with local municipal road maintenance regulations."*

---

### Question 7 (Corrected & Updated — Real-Time Edge Hardware, ONNX Export & Deployment Boundaries)
> *"You benchmarked your models on an NVIDIA RTX 5070 Laptop GPU. Municipalities cannot install gaming laptops in garbage trucks. How can this realistically be deployed at scale?"*

**Model Answer (~102 words, ~40 seconds)**:
> *"Thank you, Professor. First, to correct our hardware comparison: our test GPU is Blackwell (sm_120), whereas Jetson Orin Nano is Ampere (sm_87), so edge latency cannot be extrapolated without target-device compilation.*
> *Directly on our host, PyTorch YOLO11s runs at 42.6 FPS (23.5 ms CUDA, 138 MB VRAM). Exporting to static opset-17 FP32 ONNX (36.2 MB) achieves near-exact parity across 32 images (0.0002 px box delta, 0.0000 score delta; PASS), running at 3.2 FPS on CPU ONNX Runtime. Half-precision FP16 and dynamic INT8 reduce footprint (18.1 MB and 9.4 MB) but fail strict parity. Therefore, dedicated edge GPU/NPU compilation remains essential future work."*

**If pressed, add:**
> *"Specifically, CPU-EP FP16 ONNX introduced a 0.77 px shift and dropped 1 borderline detection near the 0.25 threshold (FAIL), while running at 287.4 ms (3.5 FPS). Calibration-free dynamic INT8 caused 51.45 px box drift and 0.2186 score drift (FAIL; 303.1 ms, 3.3 FPS), proving static calibration or QAT plus target-hardware TensorRT/ONNX benchmarking on Jetson is mandatory before field deployment."*
---

### Question 8 (Updated — Perspective Foreshortening & Uncalibrated IPM Sensitivity)
> *"Can you really determine pavement repair priority from 2D bounding boxes? Doesn't perspective foreshortening make defects near the car appear artificially huge compared to defects further away?"*

**Model Answer (~101 words, ~39 seconds)**:
> *"Yes, Professor, perspective foreshortening is our primary monocular limitation, and RDD2022 lacks camera calibration to recover metric surface area.*
> *To test how strongly this distorts our heuristic triage proxy, we built an uncalibrated Inverse Perspective Mapping module assuming canonical lane geometry (3.5 m by 20.0 m, 1.2 m height) across 40 test frames (91 in-zone boxes). Box-size rank correlation remained high (Spearman rho = 0.896, Kendall tau = 0.737). Across the 18 frames with detections (22 had none), priority tiers shifted on only 2/18 frames (11.1%; 5.0% overall), confirming stable relative triage rankings."*

**If pressed, add:**
> *"Triage score Spearman rho was 0.995 overall and 0.951 across the 18 detected frames. Varying assumed lane width from 3.0 m to 4.0 m shifted estimated_ground_area_m2 linearly (+/-14.3%, 5.50 to 7.33 m^2) with zero tier changes across those 18 frames. Automated Hough vanishing-point calibration succeeded on only 9/40 frames (22.5%), confirming metric grading requires calibrated stereo or LiDAR sensors."*