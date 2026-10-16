# Automated Road Damage Detection & Municipal Triage Demo

**Course**: Lingnan University CDS540 Computer Vision  
**System**: Edge Dashcam Vision Pipeline for Smart Municipal Road Maintenance  
**Models**: YOLO11n (Ultra-low latency baseline) & YOLO11s (High-accuracy inspection)

---

## 1. System Overview

This demonstration showcases an end-to-end, edge-deployable computer vision system that ingests raw road dashcam video feeds, executes real-time multi-hazard defect localization, and computes automated municipal maintenance triage priorities.

### The Pipeline:
1. **Video Ingestion**: Continuous processing of dashcam frames at up to 60+ FPS on edge GPU hardware.
2. **Deep Learning Detection**: Detects 4 IEEE CRDDC road damage categories:
   - `D00`: Longitudinal Crack (Wheel-path fatigue)
   - `D10`: Transverse Crack (Thermal contraction)
   - `D20`: Alligator Crack (Severe structural base distress)
   - `D40`: Pothole (High-hazard impact depression)
3. **Automated Triage Heuristics (`scripts/priority.py`)**:
   - Assesses class hazard weights (Potholes given $10\times$ weight due to asymmetrical liability costs).
   - Computes defect bounding box area ratios and co-occurrence density penalties.
4. **Heads-Up Display (HUD) Telemetry**:
   - Overlays maintenance priority badges: **HIGH (Urgent 24-48h)**, **MEDIUM (30-day queue)**, **LOW (Preventive monitoring)**, or **SOUND**.
   - Displays real-time inference latency and FPS for operational observability.

---

## 2. Running the Demo

To run the end-to-end video inference demonstration:

```powershell
.\.venv\Scripts\python.exe demo/demo.py
```

### Arguments & Customization:
- **`model_path`**: Path to trained PyTorch weights (default: `models/compare_yolo11s_best.pt`).
- **`input_source`**: Video file (`.mp4`, `.avi`) or directory containing road dashcam frames (`data/yolo/images/test`).
- **`output_video_path`**: Target rendered video path (default: `demo/output.mp4`).
- **`fps`**: Video output framerate (default: 15 FPS).
- **`max_frames`**: Number of frames to process in the demonstration clip (default: 120 frames).

---

## 3. Output Artifacts

- **`demo/output.mp4`**: Annotated demonstration video clip displaying bounding boxes, class labels, confidence scores, and real-time municipal triage telemetry.
- **`results/priority_tests/`**: Individual high-resolution inspection cards showing before/after overlay for critical pavement distress scenarios.
