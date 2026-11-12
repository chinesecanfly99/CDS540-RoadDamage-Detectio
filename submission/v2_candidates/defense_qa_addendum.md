# Oral Defense Q&A Addendum (v2 Candidate — Not Merged)

> **Scope**: Candidate updates to `submission/defense_qa_guide.md` incorporating quantitative evidence from the C-Tier Technical Upgrade (C1–C4).
> **Format**: Each primary `Model Answer` is compressed to ~95–105 words (~35–42 seconds at 140–160 wpm) with zero LaTeX macros for clean spoken delivery, followed by an `If pressed, add:` backup block containing exact granular numbers.
> **Constraints**: Strict adherence to R4 (`held-out labelled US subset`), R5 (`estimated_ground_area_m2` with uncertainty, triage proxy), and R10 (host measurements only).

---

### Question 2 (Updated — Small Object Detection & Multi-Scale Pyramid Evidence)
> *"Fine road cracks are often only 1 to 3 pixels wide on a 640x640 frame. A standard YOLO backbone downsamples feature maps by 32x at its deepest stage, which means 1 pixel becomes less than 0.03 pixels. How can your network possibly detect fine cracks?"*

**Model Answer (~98 words, ~38 seconds)**:
> *"Thank you, Professor. YOLO11 does not rely solely on the stride-32 deepest layer (P5). Its PAN-FPN head aggregates P3 (stride 8, 80x80), P4 (stride 16, 40x40), and P5 (stride 32, 20x20).*
> *We verified this empirically by hooking the Detect-feeding C3k2 layers across 25 test frames, including 4 pure background controls. Longitudinal cracks (D00) retained 73.8% of normalized energy in P3 and P4 while dropping to 0.94x frame RMS at P5, near background (0.93x). Transverse cracks (D10) showed P3 share slightly above background (36.1% vs 34.4%). Thus, shallow high-resolution feature maps preserve hairline edges lost at stride 32."*

**If pressed, add:**
> *"These GT-region hook measurements are correlative localization evidence rather than proof of detection, as sample detection rates were low (D10 0%, D00 16.7%, D40 19.2%, D20 50.0%). For our largest group, D40 potholes (n=26), normalized P5 share (30.9%) fell below background (32.3%) even though absolute P5 energy was elevated (1.15x RMS), so our findings are only partially consistent with motion-blur asymmetry and warrant a class-matched D40 follow-up."*

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
