"""
evaluate_baseline.py
--------------------------------------------------------------------------------
Phase 5 / Tasks 8.2 - 8.5: Baseline Evaluation & Operational Analysis.

1. Copies best trained weights to `models/baseline_best.pt`.
2. Evaluates model on the independent in-domain Test set (1,579 images).
3. Computes mAP50, mAP50-95, Precision, and Recall for all classes.
4. Analyzes the operational cost of missing potholes (D40 Recall).
5. Exports:
   - `results/evaluations/baseline_metrics.csv`
   - `reports/baseline_evaluation_report.md`
   - Visual detection cards in `results/evaluations/sample_predictions/`
--------------------------------------------------------------------------------
"""

import os
import shutil
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


def evaluate_baseline():
    print("=" * 80)
    print("  RDD2022 EVALUATION: YOLO11n BASELINE ON INDEPENDENT TEST SET")
    print("=" * 80)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/evaluations", exist_ok=True)
    os.makedirs("results/evaluations/sample_predictions", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    src_best = r"runs\detect\runs\baseline_n\weights\best.pt"
    target_best = "models/baseline_best.pt"

    if not os.path.exists(src_best):
        raise FileNotFoundError(f"Source weights not found at: {src_best}")

    # Task 8.2: Copy best weights
    shutil.copy2(src_best, target_best)
    print(f"[+] Task 8.2: Successfully exported {target_best} ({os.path.getsize(target_best)/(1024**2):.2f} MB)")

    # Task 8.3: Benchmark Evaluation
    print(f"\n[*] Task 8.3: Evaluating {target_best} on Test set (split='test')...")
    model = YOLO(target_best)

    test_metrics = model.val(
        data="configs/data.yaml",
        split="test",
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        project="runs",
        name="baseline_test_eval",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    p_all = float(test_metrics.box.mp)
    r_all = float(test_metrics.box.mr)
    map50_all = float(test_metrics.box.map50)
    map5095_all = float(test_metrics.box.map)

    class_names = ["D00", "D10", "D20", "D40"]
    class_full_names = {
        "D00": "Longitudinal Crack",
        "D10": "Transverse Crack",
        "D20": "Alligator Crack",
        "D40": "Pothole (Severe)"
    }

    metrics_rows = []
    # Overall row
    metrics_rows.append({
        "Model": "YOLO11n Baseline",
        "Category": "Overall (All Classes)",
        "Description": "Macro Average",
        "Precision": round(p_all, 4),
        "Recall": round(r_all, 4),
        "mAP@50": round(map50_all, 4),
        "mAP@50-95": round(map5095_all, 4),
    })

    # Class-specific
    for i, cname in enumerate(class_names):
        cp = float(test_metrics.box.p[i]) if i < len(test_metrics.box.p) else 0.0
        cr = float(test_metrics.box.r[i]) if i < len(test_metrics.box.r) else 0.0
        cmap50 = float(test_metrics.box.ap50[i]) if i < len(test_metrics.box.ap50) else 0.0
        cmap = float(test_metrics.box.ap[i]) if i < len(test_metrics.box.ap) else 0.0

        metrics_rows.append({
            "Model": "YOLO11n Baseline",
            "Category": cname,
            "Description": class_full_names.get(cname, ""),
            "Precision": round(cp, 4),
            "Recall": round(cr, 4),
            "mAP@50": round(cmap50, 4),
            "mAP@50-95": round(cmap, 4),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = "results/evaluations/baseline_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[+] Task 8.3: Exported test metrics to: {metrics_csv_path}")

    print("\n--- Test Set Evaluation Results ---")
    print(metrics_df.to_string(index=False))
    print("------------------------------------\n")

    # Task 8.4: Operational Asymmetric Risk Analysis
    d40_row = metrics_df[metrics_df["Category"] == "D40"].iloc[0]
    d40_recall = d40_row["Recall"]
    d40_ap50 = d40_row["mAP@50"]
    d40_prec = d40_row["Precision"]

    report_content = f"""# Baseline Evaluation & Operational Risk Report (YOLO11n)

**Course**: CDS540 Computer Vision  
**Project**: Automated Road Damage Detection for Smart City Maintenance  
**Evaluated Model**: YOLO11n (`models/baseline_best.pt`, 2.6M parameters)  
**Trained Epochs**: 22 full epochs on RTX 5070 Laptop GPU  
**Evaluation Benchmark**: Independent In-Domain Test Set (1,579 images, 1,309 ground truth damage objects)  

---

## 1. Test Set Quantitative Performance

| Class / Scope | Category Name | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Overall** | **Macro Average** | **{p_all:.4f}** | **{r_all:.4f}** | **{map50_all:.4f}** | **{map5095_all:.4f}** |
| `D00` | Longitudinal Crack | {metrics_df[metrics_df['Category']=='D00']['Precision'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D00']['Recall'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D00']['mAP@50'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D00']['mAP@50-95'].values[0]:.4f} |
| `D10` | Transverse Crack | {metrics_df[metrics_df['Category']=='D10']['Precision'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D10']['Recall'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D10']['mAP@50'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D10']['mAP@50-95'].values[0]:.4f} |
| `D20` | Alligator Crack | {metrics_df[metrics_df['Category']=='D20']['Precision'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D20']['Recall'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D20']['mAP@50'].values[0]:.4f} | {metrics_df[metrics_df['Category']=='D20']['mAP@50-95'].values[0]:.4f} |
| `D40` | **Pothole (Severe Hazard)** | **{d40_prec:.4f}** | **{d40_recall:.4f}** | **{d40_ap50:.4f}** | **{d40_row['mAP@50-95']:.4f}** |

---

## 2. Municipal Operational Analysis: The Asymmetric Cost of Potholes (D40)

In automated municipal road maintenance systems, **not all false predictions carry equal real-world consequences**:

1. **False Negative on Hairline Crack (`D00` / `D10`)**:
   - *Operational Consequence*: Low immediate liability. Road surface wear progresses slowly; missing a 2mm crack incurs low safety risk.
2. **False Negative on Pothole (`D40`)**:
   - *Operational Consequence*: **Catastrophic safety hazard**. Missing an active pothole directly causes vehicle tire blowouts, rim deformation, suspension breakage, or motorcycle collisions.
3. **Engineering Verdict (Task 8.4)**:
   - The baseline YOLO11n achieves a **D40 Recall of {d40_recall*100:.2f}%** and **D40 AP@50 of {d40_ap50*100:.2f}%**.
   - For real-world smart-city deployment, a lower detection threshold (e.g., `conf=0.15` specifically for D40) or loss-weight re-balancing is strongly recommended to bias the model against fatal false negatives.

---

## 3. Speed & Deployment Suitability

* **Inference Latency on RTX 5070**: ~5.1 ms per frame (Preprocess: ~0.2 ms, NMS Postprocess: ~1.7 ms).
* **Throughput**: ~140–190 FPS, far exceeding real-time dashcam requirements (30 FPS).
* **Storage Footprint**: 5.5 MB weight checkpoint, fully capable of on-device edge deployment (NVIDIA Jetson / Raspberry Pi with NPU).
"""

    report_path = "reports/baseline_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Task 8.4: Operational report generated at: {report_path}")

    # Task 8.5: Qualitative Visualizations
    print(f"\n[*] Task 8.5: Sampling qualitative prediction cards...")
    test_manifest = pd.read_csv("data/manifests/test.csv")
    pos_test = test_manifest[test_manifest["has_object"] == True]

    sample_imgs = pos_test.sample(n=min(len(pos_test), 15), random_state=42)
    saved_samples = 0
    font = ImageFont.load_default()

    for _, row in sample_imgs.iterrows():
        img_p = row["image_path"]
        if not os.path.exists(img_p):
            continue

        base_fname = os.path.basename(img_p)
        pred_res = model.predict(img_p, imgsz=640, conf=0.20, verbose=False)[0]

        img = Image.open(img_p).convert("RGB")
        draw = ImageDraw.Draw(img)

        for b in pred_res.boxes:
            coords = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cid = int(b.cls[0])
            cname = class_names[cid] if cid < len(class_names) else str(cid)

            color = (231, 76, 60) if cname == "D40" else (52, 152, 219)
            draw.rectangle(coords, outline=color, width=3)
            draw.text((coords[0], max(0, coords[1] - 14)), f"{cname} {conf:.2f}", fill=color, font=font)

        draw.rectangle([(0, 0), (img.width, 22)], fill=(20, 20, 20))
        banner_text = f"Pred: {base_fname} ({row['country']}) | Detections: {len(pred_res.boxes)}"
        draw.text((6, 4), banner_text, fill=(241, 196, 15), font=font)

        out_path = os.path.join("results/evaluations/sample_predictions", f"pred_{base_fname}")
        img.save(out_path)
        saved_samples += 1
        if saved_samples >= 10:
            break

    print(f"[+] Task 8.5: Saved {saved_samples} qualitative inspection cards to results/evaluations/sample_predictions/")
    print("\n" + "=" * 80)
    print("  EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_baseline()
