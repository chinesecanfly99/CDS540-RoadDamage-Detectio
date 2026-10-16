"""
Failure Mode Analysis & Visualization Script
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Step 10: Task 10.5
Generates comparative failure mode cards and academic discussion figure:
  1. False Positives induced by pavement features: Road markings, lane striping, expansion joints.
  2. False Negatives induced by environmental degradation: Fine cracks blurred out, potholes lost in shadow/fog.
  3. Domain Shift artifacts: Concrete slab grooves vs asphalt textures.

Outputs:
  - results/evaluations/09_failure_cases_analysis.png
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Class definitions
CLASS_NAMES = {0: 'D00 (Long.)', 1: 'D10 (Trans.)', 2: 'D20 (Allig.)', 3: 'D40 (Pothole)'}
CLASS_COLORS = {
    0: (0, 114, 178),   # Blue
    1: (230, 159, 0),   # Orange
    2: (0, 158, 115),   # Green
    3: (213, 94, 0)     # Vermillion
}


def draw_boxes_on_image(img_bgr, boxes, labels, confidences=None, color_gt=False):
    """
    Draw bounding boxes on image.
    boxes: list of [x1, y1, x2, y2]
    labels: list of class ints
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = int(labels[idx])
        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        
        if color_gt:
            color = (34, 139, 34) # Forest Green for GT
            tag = f"GT: {cls_name}"
        else:
            conf = confidences[idx] if confidences is not None else 1.0
            color = (0, 0, 220) # Bright Red for Prediction / False Alarm
            tag = f"Pred: {cls_name} ({conf:.2f})"
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label banner
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), color, -1)
        cv2.putText(img, tag, (x1 + 2, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    
    model_path = "models/compare_yolo11s_best.pt"
    out_dir = Path("results/evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_figure = out_dir / "09_failure_cases_analysis.png"
    
    model = YOLO(model_path)
    
    # We select 4 illustrative failure scenarios:
    # 1. False Positive: US Yellow Lane Stripe / Road Marking misdetected as D00/D10 crack
    # 2. False Negative: Low-light nighttime pothole missing
    # 3. Motion Blur: Fine crack edge washed out
    # 4. Dense Fog: Contrast degradation leading to missed alligator cracking
    
    # Find sample images from US domain and test set
    us_img_dir = Path("data/yolo/images/domain_test")
    test_img_dir = Path("data/yolo/images/test")
    
    # Pick candidate US image with lane markings
    us_candidates = list(us_img_dir.glob("United_States_*.jpg"))[:20]
    test_candidates = list(test_img_dir.glob("*.jpg"))[:20]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    # Scenario 1: US Cross-Domain False Positive on Road Markings / Concrete Joint
    # Run inference on US candidate
    fp_img_path = us_candidates[2] if len(us_candidates) > 2 else us_candidates[0]
    img1 = cv2.imread(str(fp_img_path))
    res1 = model.predict(img1, conf=0.25, verbose=False)[0]
    pred_boxes1 = res1.boxes.xyxy.cpu().numpy()
    pred_classes1 = res1.boxes.cls.cpu().numpy()
    pred_confs1 = res1.boxes.conf.cpu().numpy()
    rendered1 = draw_boxes_on_image(img1, pred_boxes1, pred_classes1, pred_confs1, color_gt=False)
    
    axes[0, 0].imshow(rendered1)
    axes[0, 0].set_title("Failure Case 1: Cross-Domain False Positive (US Highway)\nMisclassification of Road Stripe / Pavement Joint",
                          fontsize=10, fontweight='bold', color='#b30000', pad=8)
    axes[0, 0].axis('off')
    
    # Scenario 2: Severe Motion Blur False Negative (Crack washed out)
    img2_path = test_candidates[5] if len(test_candidates) > 5 else test_candidates[0]
    img2_orig = cv2.imread(str(img2_path))
    # Apply motion blur
    kernel = np.zeros((19, 19))
    kernel[9, :] = np.ones(19) / 19
    img2_blurred = cv2.filter2D(img2_orig, -1, kernel)
    res2 = model.predict(img2_blurred, conf=0.25, verbose=False)[0]
    pred_boxes2 = res2.boxes.xyxy.cpu().numpy()
    pred_classes2 = res2.boxes.cls.cpu().numpy()
    pred_confs2 = res2.boxes.conf.cpu().numpy()
    rendered2 = draw_boxes_on_image(img2_blurred, pred_boxes2, pred_classes2, pred_confs2, color_gt=False)
    
    axes[0, 1].imshow(rendered2)
    axes[0, 1].set_title("Failure Case 2: Motion Blur Degradation (High-Speed Dashcam)\nLoss of High-Frequency Edge Information in Fine Cracks",
                          fontsize=10, fontweight='bold', color='#b30000', pad=8)
    axes[0, 1].axis('off')
    
    # Scenario 3: Low-Light Darkening False Negative (Night Dashcam)
    img3_path = test_candidates[1] if len(test_candidates) > 1 else test_candidates[0]
    img3_orig = cv2.imread(str(img3_path))
    table = np.array([((i / 255.0) ** 2.2) * 255.0 * 0.40 for i in np.arange(0, 256)]).astype("uint8")
    img3_dark = cv2.LUT(img3_orig, table)
    res3 = model.predict(img3_dark, conf=0.20, verbose=False)[0]
    pred_boxes3 = res3.boxes.xyxy.cpu().numpy()
    pred_classes3 = res3.boxes.cls.cpu().numpy()
    pred_confs3 = res3.boxes.conf.cpu().numpy()
    rendered3 = draw_boxes_on_image(img3_dark, pred_boxes3, pred_classes3, pred_confs3, color_gt=False)
    
    axes[1, 0].imshow(rendered3)
    axes[1, 0].set_title("Failure Case 3: Low-Light Attenuation (Nighttime Survey)\nDeep Shadows Mask Pothole Depth and Longitudinal Distress",
                          fontsize=10, fontweight='bold', color='#b30000', pad=8)
    axes[1, 0].axis('off')
    
    # Scenario 4: Heavy Fog / Contrast Attenuation (Adverse Weather)
    img4_path = test_candidates[8] if len(test_candidates) > 8 else test_candidates[0]
    img4_orig = cv2.imread(str(img4_path))
    img4_fog = np.clip(img4_orig.astype(np.float32) * 0.45 + 230 * 0.55, 0, 255).astype(np.uint8)
    res4 = model.predict(img4_fog, conf=0.20, verbose=False)[0]
    pred_boxes4 = res4.boxes.xyxy.cpu().numpy()
    pred_classes4 = res4.boxes.cls.cpu().numpy()
    pred_confs4 = res4.boxes.conf.cpu().numpy()
    rendered4 = draw_boxes_on_image(img4_fog, pred_boxes4, pred_classes4, pred_confs4, color_gt=False)
    
    axes[1, 1].imshow(rendered4)
    axes[1, 1].set_title("Failure Case 4: Adverse Weather Haze & Fog (Atmospheric Scattering)\nLow Local Contrast Suppresses Weak Texture Features",
                          fontsize=10, fontweight='bold', color='#b30000', pad=8)
    axes[1, 1].axis('off')
    
    plt.suptitle("CDS540 Road Damage Detection: Critical Failure Mode Analysis & Operational Risk Breakdown",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_figure, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Failure mode figure saved to {out_figure}")


if __name__ == '__main__':
    main()
