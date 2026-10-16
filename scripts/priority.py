"""
Municipal Road Maintenance Prioritization Heuristic Module
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Step 11: Task 11.1 & Task 11.2
Translates bounding-box object detections into actionable municipal repair priorities:
  - Class Asymmetry: High hazard weighting for D40 (potholes) and structural fatigue D20 (alligator).
  - Spatial Extent: Relative bounding box area ratio (penalty for large distress patches).
  - Distress Multiplicity: Compounding penalty for multiple co-occurring defects on a roadway frame.
  - Priority Categorization: HIGH (Urgent Dispatch), MEDIUM (Scheduled Resurfacing), LOW (Preventive Sealing).

DISCLAIMER: This heuristic prioritization formula is designed as an automated triage proxy
for municipal road survey workflows and does not replace certified geotechnical pavement inspections.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Distress Class Weights reflecting municipal safety and structural degradation risk
CLASS_WEIGHTS = {
    'D40': 10.0,  # Potholes: Acute danger of tire rupture, vehicle loss of control, cyclist accidents
    'D20': 6.5,   # Alligator Cracking: Subgrade structural failure, high moisture vulnerability
    'D10': 3.5,   # Transverse Cracking: Thermal contraction joints, ride quality degradation
    'D00': 2.5    # Longitudinal Cracking: Wheel-path fatigue, early stage wear
}

CLASS_MAP = {
    0: 'D00',
    1: 'D10',
    2: 'D20',
    3: 'D40'
}

CLASS_FULL_NAMES = {
    'D00': 'Longitudinal Crack',
    'D10': 'Transverse Crack',
    'D20': 'Alligator Crack',
    'D40': 'Pothole'
}

PRIORITY_LEVELS = {
    'HIGH': {
        'color_bgr': (0, 0, 220),       # Bright Red
        'color_rgb': (220, 0, 0),
        'action': 'Urgent repair crew dispatch (24-48h). Cold-patch / safety warning.',
        'min_score': 12.0
    },
    'MEDIUM': {
        'color_bgr': (0, 165, 255),     # Orange / Amber
        'color_rgb': (255, 165, 0),
        'action': 'Scheduled maintenance queue (30 days). Mill-and-overlay program.',
        'min_score': 6.0
    },
    'LOW': {
        'color_bgr': (0, 200, 0),       # Green
        'color_rgb': (0, 200, 0),
        'action': 'Preventive maintenance. Crack sealing to inhibit water ingress.',
        'min_score': 0.1
    },
    'NONE': {
        'color_bgr': (180, 180, 180),   # Gray
        'color_rgb': (180, 180, 180),
        'action': 'Pavement sound. Maintain standard biennial inspection schedule.',
        'min_score': 0.0
    }
}


class RoadDamagePrioritizer:
    def __init__(self, conf_threshold=0.25):
        self.conf_threshold = conf_threshold

    def evaluate_detections(self, boxes, classes, confidences, img_width, img_height):
        """
        Calculates priority score and actionable recommendations from raw detections.
        
        Parameters:
            boxes: array-like of [x1, y1, x2, y2]
            classes: array-like of class IDs (0-3)
            confidences: array-like of float confidence scores
            img_width: int
            img_height: int
            
        Returns:
            dict containing triage score, priority level, defect breakdown, and action recommendation.
        """
        img_area = float(img_width * img_height)
        total_score = 0.0
        defect_summary = {'D00': 0, 'D10': 0, 'D20': 0, 'D40': 0}
        detected_items = []
        has_severe_pothole = False
        
        for idx in range(len(boxes)):
            conf = float(confidences[idx])
            if conf < self.conf_threshold:
                continue
                
            cls_id = int(classes[idx])
            cls_code = CLASS_MAP.get(cls_id, 'D00')
            defect_summary[cls_code] += 1
            
            x1, y1, x2, y2 = boxes[idx]
            box_area = max(0.0, float((x2 - x1) * (y2 - y1)))
            area_ratio = box_area / img_area
            
            # Severity calculation:
            # Base weight * Confidence * Size Scaling Multiplier
            base_wt = CLASS_WEIGHTS.get(cls_code, 2.0)
            
            # Non-linear size penalty
            if area_ratio >= 0.10:
                size_factor = 2.0
            elif area_ratio >= 0.04:
                size_factor = 1.5
            elif area_ratio >= 0.01:
                size_factor = 1.2
            else:
                size_factor = 1.0
                
            item_score = base_wt * conf * size_factor
            total_score += item_score
            
            # Special emergency trigger: High-confidence pothole with substantial area
            if cls_code == 'D40' and conf >= 0.35 and area_ratio >= 0.005:
                has_severe_pothole = True
                
            detected_items.append({
                'class_code': cls_code,
                'class_name': CLASS_FULL_NAMES[cls_code],
                'confidence': conf,
                'area_ratio': area_ratio,
                'item_score': item_score,
                'box': [float(x1), float(y1), float(x2), float(y2)]
            })
            
        # Distress Multiplicity compounding factor (density penalty)
        num_distress = len(detected_items)
        if num_distress >= 4:
            multiplicity_bonus = 1.4
        elif num_distress >= 2:
            multiplicity_bonus = 1.2
        else:
            multiplicity_bonus = 1.0
            
        final_score = round(total_score * multiplicity_bonus, 2)
        
        # Categorization logic
        if has_severe_pothole or final_score >= PRIORITY_LEVELS['HIGH']['min_score']:
            level = 'HIGH'
        elif final_score >= PRIORITY_LEVELS['MEDIUM']['min_score']:
            level = 'MEDIUM'
        elif final_score >= PRIORITY_LEVELS['LOW']['min_score']:
            level = 'LOW'
        else:
            level = 'NONE'
            
        return {
            'total_score': final_score,
            'priority_level': level,
            'action_recommendation': PRIORITY_LEVELS[level]['action'],
            'total_distress_count': num_distress,
            'defect_summary': defect_summary,
            'items': detected_items,
            'has_critical_pothole': has_severe_pothole
        }

    def render_overlay(self, img_bgr, triage_result):
        """
        Renders HUD dashboard overlay on road image displaying detections and priority level.
        """
        img = img_bgr.copy()
        h, w = img.shape[:2]
        level = triage_result['priority_level']
        level_color = PRIORITY_LEVELS[level]['color_bgr']
        
        # 1. Draw detection boxes
        for item in triage_result['items']:
            x1, y1, x2, y2 = [int(v) for v in item['box']]
            cls_code = item['class_code']
            conf = item['confidence']
            
            # Distinct box color per class
            if cls_code == 'D40':
                b_color = (0, 0, 240) # Bright red
            elif cls_code == 'D20':
                b_color = (0, 140, 255) # Orange
            elif cls_code == 'D10':
                b_color = (255, 180, 0) # Cyan/Yellow
            else:
                b_color = (255, 100, 0) # Blue
                
            cv2.rectangle(img, (x1, y1), (x2, y2), b_color, 2)
            label_text = f"{cls_code} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), b_color, -1)
            cv2.putText(img, label_text, (x1 + 2, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Top Telemetry HUD Dashboard Banner
        banner_h = 75
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        
        # Priority Badge (Left)
        badge_w = 160
        cv2.rectangle(img, (10, 10), (badge_w, banner_h - 10), level_color, -1)
        cv2.putText(img, f"PRIORITY", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, level, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Telemetry Details (Center / Right)
        score_text = f"Triage Score: {triage_result['total_score']:.1f}  |  Distress Count: {triage_result['total_distress_count']}"
        cv2.putText(img, score_text, (badge_w + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        summary_str = f"D40(Pothole): {triage_result['defect_summary']['D40']} | D20(Allig): {triage_result['defect_summary']['D20']} | D10(Trans): {triage_result['defect_summary']['D10']} | D00(Long): {triage_result['defect_summary']['D00']}"
        cv2.putText(img, summary_str, (badge_w + 20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        action_short = triage_result['action_recommendation'][:65] + "..." if len(triage_result['action_recommendation']) > 65 else triage_result['action_recommendation']
        cv2.putText(img, f"Action: {action_short}", (badge_w + 20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        return img


def test_sample_images(model_path, test_csv, num_samples=6, out_dir="results/priority_tests"):
    """
    Task 11.2: Tests priority rules on sample road images, producing annotated visual cards and summary CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prioritizer = RoadDamagePrioritizer(conf_threshold=0.25)
    model = YOLO(model_path)
    
    df_test = pd.read_csv(test_csv)
    # Select samples covering diverse severity: Pothole (D40), Alligator (D20), Crack, Negative
    target_stems = [
        'India_000061',  # High severity Pothole (D40)
        'India_000209',  # Multi-distress Pothole (D40) + Alligator (D20)
        'India_000302',  # Severe Alligator Fatigue Cracking (D20)
        'Czech_003338',  # Longitudinal Crack (D00)
        'Czech_003025',  # Sound Road / Negative Frame
        'India_000168'   # Multiple Potholes
    ]
    
    selected_rows = []
    for stem in target_stems:
        matched = df_test[df_test['image_path'].str.contains(stem)]
        if not matched.empty:
            selected_rows.append(matched.iloc[0])
            
    if selected_rows:
        selected_df = pd.DataFrame(selected_rows)
    else:
        selected_df = df_test.head(num_samples)
    
    records = []
    
    for idx, row in selected_df.iterrows():
        img_name = Path(row['image_path']).name
        img_path = Path("data/yolo/images/test") / img_name
        if not img_path.exists():
            img_path = Path(row['image_path'])
            
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
            
        h, w = bgr.shape[:2]
        res = model.predict(bgr, conf=0.25, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        
        triage = prioritizer.evaluate_detections(boxes, classes, confs, w, h)
        annotated_img = prioritizer.render_overlay(bgr, triage)
        
        save_card_path = out_dir / f"priority_{idx}_{triage['priority_level']}_{img_name}"
        cv2.imwrite(str(save_card_path), annotated_img)
        
        records.append({
            'Image_ID': img_name,
            'Ground_Truth_Classes': row['raw_classes'] if row['has_object'] else 'None (Sound Pavement)',
            'Detected_Distress_Count': triage['total_distress_count'],
            'D40_Potholes': triage['defect_summary']['D40'],
            'D20_Alligator': triage['defect_summary']['D20'],
            'D10_Transverse': triage['defect_summary']['D10'],
            'D00_Longitudinal': triage['defect_summary']['D00'],
            'Triage_Score': triage['total_score'],
            'Priority_Level': triage['priority_level'],
            'Action_Recommendation': triage['action_recommendation'],
            'Visual_Output_Card': str(save_card_path.as_posix())
        })
        
    res_df = pd.DataFrame(records)
    csv_out = out_dir / "priority_evaluation_summary.csv"
    res_df.to_csv(csv_out, index=False)
    print(f"\n[SUCCESS] Priority test results saved to {csv_out}")
    print(res_df[['Image_ID', 'Priority_Level', 'Triage_Score', 'Detected_Distress_Count', 'D40_Potholes']].to_string())
    return res_df


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    test_sample_images(
        model_path="models/compare_yolo11s_best.pt",
        test_csv="data/manifests/test.csv",
        num_samples=6
    )
