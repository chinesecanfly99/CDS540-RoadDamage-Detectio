"""
Cross-Country Domain Generalization Evaluation Script
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Step 10: Task 10.1 & Task 10.2
Evaluates in-domain trained models (Czech + India) on zero-shot unseen target domain:
United States highway dashcam images (4,805 images, held-out).
Quantifies domain shift decay and per-class performance degradation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Set plot style for publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def evaluate_model_on_domain(model_path, data_yaml, project_name, exp_name):
    """
    Run YOLO evaluation on target domain test set.
    """
    print(f"\n=======================================================")
    print(f"Evaluating {model_path} on {data_yaml} ({exp_name})...")
    print(f"=======================================================")
    
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split='val',
        batch=16,
        imgsz=640,
        device=0,
        workers=0,
        project=project_name,
        name=exp_name,
        exist_ok=True,
        verbose=True
    )
    
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    
    class_map50 = metrics.box.ap50.tolist() if hasattr(metrics.box, 'ap50') else []
    class_r = metrics.box.r.tolist() if hasattr(metrics.box, 'r') else []
    
    # Class mapping: 0: D00, 1: D10, 2: D20, 3: D40
    d00_ap = class_map50[0] if len(class_map50) > 0 else 0.0
    d10_ap = class_map50[1] if len(class_map50) > 1 else 0.0
    d20_ap = class_map50[2] if len(class_map50) > 2 else 0.0
    d40_ap = class_map50[3] if len(class_map50) > 3 else 0.0
    d40_recall = class_r[3] if len(class_r) > 3 else 0.0
    
    return {
        'Precision': mp,
        'Recall': mr,
        'mAP@50': map50,
        'mAP@50-95': map50_95,
        'D00_AP50': d00_ap,
        'D10_AP50': d10_ap,
        'D20_AP50': d20_ap,
        'D40_AP50': d40_ap,
        'D40_Recall': d40_recall
    }


def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    
    domain_yaml = "configs/data_domain_test.yaml"
    out_csv = "results/domain_shift.csv"
    eval_fig_dir = Path("results/evaluations")
    eval_fig_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_test = [
        {
            'name': 'YOLO11n (Baseline)',
            'weights': 'models/baseline_best.pt',
            'exp_name': 'us_domain_yolo11n',
            # In-domain metrics from test set (results/model_comparison.csv)
            'in_domain': {
                'Precision': 0.3906,
                'Recall': 0.2774,
                'mAP@50': 0.2169,
                'mAP@50-95': 0.0833,
                'D00_AP50': 0.2092,
                'D10_AP50': 0.0533,
                'D20_AP50': 0.3959,
                'D40_AP50': 0.2093,
                'D40_Recall': 0.3333
            }
        },
        {
            'name': 'YOLO11s (Scaled)',
            'weights': 'models/compare_yolo11s_best.pt',
            'exp_name': 'us_domain_yolo11s',
            # In-domain metrics from test set (results/model_comparison.csv)
            'in_domain': {
                'Precision': 0.3834,
                'Recall': 0.3305,
                'mAP@50': 0.2846,
                'mAP@50-95': 0.1140,
                'D00_AP50': 0.2387,
                'D10_AP50': 0.0863,
                'D20_AP50': 0.5210,
                'D40_AP50': 0.2923,
                'D40_Recall': 0.3974
            }
        }
    ]
    
    results_records = []
    
    for item in models_to_test:
        model_name = item['name']
        weights_path = item['weights']
        exp_name = item['exp_name']
        in_metrics = item['in_domain']
        
        cross_metrics = evaluate_model_on_domain(
            model_path=weights_path,
            data_yaml=domain_yaml,
            project_name='runs/detect/domain_eval',
            exp_name=exp_name
        )
        
        # In-domain record
        results_records.append({
            'Model': model_name,
            'Domain': 'In-Domain (Czech + India Test)',
            'Dataset Size': 1579,
            'Precision': round(in_metrics['Precision'], 4),
            'Recall': round(in_metrics['Recall'], 4),
            'mAP@50': round(in_metrics['mAP@50'], 4),
            'mAP@50-95': round(in_metrics['mAP@50-95'], 4),
            'D00_AP50': round(in_metrics['D00_AP50'], 4),
            'D10_AP50': round(in_metrics['D10_AP50'], 4),
            'D20_AP50': round(in_metrics['D20_AP50'], 4),
            'D40_AP50': round(in_metrics['D40_AP50'], 4),
            'D40_Recall': round(in_metrics['D40_Recall'], 4)
        })
        
        # Cross-domain record
        results_records.append({
            'Model': model_name,
            'Domain': 'Cross-Domain (United States Holdout)',
            'Dataset Size': 4805,
            'Precision': round(cross_metrics['Precision'], 4),
            'Recall': round(cross_metrics['Recall'], 4),
            'mAP@50': round(cross_metrics['mAP@50'], 4),
            'mAP@50-95': round(cross_metrics['mAP@50-95'], 4),
            'D00_AP50': round(cross_metrics['D00_AP50'], 4),
            'D10_AP50': round(cross_metrics['D10_AP50'], 4),
            'D20_AP50': round(cross_metrics['D20_AP50'], 4),
            'D40_AP50': round(cross_metrics['D40_AP50'], 4),
            'D40_Recall': round(cross_metrics['D40_Recall'], 4)
        })
        
        # Decay record (%)
        def calc_decay(in_val, cross_val):
            if in_val == 0:
                return 0.0
            return round(((cross_val - in_val) / in_val) * 100.0, 2)
            
        results_records.append({
            'Model': model_name,
            'Domain': 'Relative Change (%)',
            'Dataset Size': 0,
            'Precision': f"{calc_decay(in_metrics['Precision'], cross_metrics['Precision']):+0.2f}%",
            'Recall': f"{calc_decay(in_metrics['Recall'], cross_metrics['Recall']):+0.2f}%",
            'mAP@50': f"{calc_decay(in_metrics['mAP@50'], cross_metrics['mAP@50']):+0.2f}%",
            'mAP@50-95': f"{calc_decay(in_metrics['mAP@50-95'], cross_metrics['mAP@50-95']):+0.2f}%",
            'D00_AP50': f"{calc_decay(in_metrics['D00_AP50'], cross_metrics['D00_AP50']):+0.2f}%",
            'D10_AP50': f"{calc_decay(in_metrics['D10_AP50'], cross_metrics['D10_AP50']):+0.2f}%",
            'D20_AP50': f"{calc_decay(in_metrics['D20_AP50'], cross_metrics['D20_AP50']):+0.2f}%",
            'D40_AP50': f"{calc_decay(in_metrics['D40_AP50'], cross_metrics['D40_AP50']):+0.2f}%",
            'D40_Recall': f"{calc_decay(in_metrics['D40_Recall'], cross_metrics['D40_Recall']):+0.2f}%"
        })

    df = pd.DataFrame(results_records)
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Domain shift evaluation results saved to {out_csv}")
    print(df.to_string())
    
    # Generate Publication-Quality Figure: In-Domain vs Cross-Domain Generalization Decay
    plot_domain_shift_chart(df, eval_fig_dir / "06_cross_domain_generalization.png")


def plot_domain_shift_chart(df, save_path):
    """
    Plots a dual-panel figure:
    Panel A: Overall mAP@50 & D40 Recall Decay across Models
    Panel B: Per-Damage Class AP@50 Comparison (In-Domain vs Cross-Domain)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    models = ['YOLO11n (Baseline)', 'YOLO11s (Scaled)']
    x = np.arange(len(models))
    width = 0.35
    
    # Filter numeric rows
    df_in = df[df['Domain'].str.contains('In-Domain')].set_index('Model')
    df_cross = df[df['Domain'].str.contains('Cross-Domain')].set_index('Model')
    
    # ---------------- Panel A: mAP@50 and D40 Recall ----------------
    in_map50 = [df_in.loc[m, 'mAP@50'] for m in models]
    cross_map50 = [df_cross.loc[m, 'mAP@50'] for m in models]
    in_d40_r = [df_in.loc[m, 'D40_Recall'] for m in models]
    cross_d40_r = [df_cross.loc[m, 'D40_Recall'] for m in models]
    
    rects1 = ax1.bar(x - width/2, in_map50, width, label='In-Domain mAP@50 (Czech+India)', color='#2b5c8f', alpha=0.9, edgecolor='black', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, cross_map50, width, label='Cross-Domain mAP@50 (US Holdout)', color='#d95f02', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # Marker lines for D40 Recall
    ax1.plot(x - width/2, in_d40_r, 's--', color='#1b9e77', markersize=8, linewidth=2, label='In-Domain D40 Recall')
    ax1.plot(x + width/2, cross_d40_r, 'o--', color='#e7298a', markersize=8, linewidth=2, label='Cross-Domain D40 Recall')
    
    ax1.set_title('A. Overall Detection & Pothole Recall Under Domain Shift', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 0.65)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ---------------- Panel B: Per-Class AP50 Comparison for YOLO11s ----------------
    classes = ['D00 (Long.)', 'D10 (Trans.)', 'D20 (Allig.)', 'D40 (Pothole)']
    class_cols = ['D00_AP50', 'D10_AP50', 'D20_AP50', 'D40_AP50']
    
    x2 = np.arange(len(classes))
    y11s_in = [df_in.loc['YOLO11s (Scaled)', c] for c in class_cols]
    y11s_cross = [df_cross.loc['YOLO11s (Scaled)', c] for c in class_cols]
    
    b1 = ax2.bar(x2 - width/2, y11s_in, width, label='In-Domain AP@50 (YOLO11s)', color='#386cb0', alpha=0.9, edgecolor='black', linewidth=0.8)
    b2 = ax2.bar(x2 + width/2, y11s_cross, width, label='Cross-Domain AP@50 (YOLO11s)', color='#f0027f', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax2.set_title('B. Per-Damage Class Generalization Breakdown (YOLO11s)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax2.set_ylabel('AP@50 Score', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 0.65)
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    for rect in b1:
        h = rect.get_height()
        ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in b2:
        h = rect.get_height()
        ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Chart saved to {save_path}")


if __name__ == '__main__':
    main()
