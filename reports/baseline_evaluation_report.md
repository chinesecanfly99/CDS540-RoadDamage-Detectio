# Baseline Evaluation & Operational Risk Report (YOLO11n)

**Course**: CDS540 Computer Vision  
**Project**: Automated Road Damage Detection for Smart City Maintenance  
**Evaluated Model**: YOLO11n (`models/baseline_best.pt`, 2.6M parameters)  
**Trained Epochs**: 22 full epochs on RTX 5070 Laptop GPU  
**Evaluation Benchmark**: Independent In-Domain Test Set (1,579 images, 1,309 ground truth damage objects)  

---

## 1. Test Set Quantitative Performance

| Class / Scope | Category Name | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Overall** | **Macro Average** | **0.3906** | **0.2774** | **0.2169** | **0.0833** |
| `D00` | Longitudinal Crack | 0.3459 | 0.2334 | 0.2092 | 0.0806 |
| `D10` | Transverse Crack | 0.6190 | 0.0147 | 0.0533 | 0.0187 |
| `D20` | Alligator Crack | 0.3056 | 0.5283 | 0.3959 | 0.1578 |
| `D40` | **Pothole (Severe Hazard)** | **0.2921** | **0.3333** | **0.2093** | **0.0762** |

---

## 2. Municipal Operational Analysis: The Asymmetric Cost of Potholes (D40)

In automated municipal road maintenance systems, **not all false predictions carry equal real-world consequences**:

1. **False Negative on Hairline Crack (`D00` / `D10`)**:
   - *Operational Consequence*: Low immediate liability. Road surface wear progresses slowly; missing a 2mm crack incurs low safety risk.
2. **False Negative on Pothole (`D40`)**:
   - *Operational Consequence*: **Catastrophic safety hazard**. Missing an active pothole directly causes vehicle tire blowouts, rim deformation, suspension breakage, or motorcycle collisions.
3. **Engineering Verdict (Task 8.4)**:
   - The baseline YOLO11n achieves a **D40 Recall of 33.33%** and **D40 AP@50 of 20.93%**.
   - For real-world smart-city deployment, a lower detection threshold (e.g., `conf=0.15` specifically for D40) or loss-weight re-balancing is strongly recommended to bias the model against fatal false negatives.

---

## 3. Speed & Deployment Suitability

* **Inference Latency on RTX 5070**: ~5.1 ms per frame (Preprocess: ~0.2 ms, NMS Postprocess: ~1.7 ms).
* **Throughput**: ~140–190 FPS, far exceeding real-time dashcam requirements (30 FPS).
* **Storage Footprint**: 5.5 MB weight checkpoint, fully capable of on-device edge deployment (NVIDIA Jetson / Raspberry Pi with NPU).
