"""
Environmental Robustness and Adverse Condition Perturbation Benchmark
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Step 10: Task 10.3 & Task 10.4
Simulates adverse operational conditions for municipal vehicle dashcams:
  1. Clean (Baseline reference)
  2. Low-Light / Night Dashcam (Gamma darkening + brightness reduction)
  3. Motion Blur (Horizontal dashcam camera vibration at highway speed)
  4. Adverse Weather (Haze / fog via Koschmieder atmospheric scattering)
  5. Sensor Noise (Additive zero-mean Gaussian noise from cheap rolling-shutter CMOS)

Evaluates performance decay across perturbations without altering bounding box geometry.
Saves results to results/robustness.csv and generates publication-grade visualizations.
"""

import os
import sys
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# --- Perturbation Mathematical Formulations ---

def perturb_clean(img):
    """Clean reference image."""
    return img.copy()


def perturb_low_light(img, gamma=2.0, brightness=0.45):
    """
    Simulates unlit nighttime dashcam conditions.
    I' = clip(brightness * (I / 255.0)^gamma * 255.0, 0, 255)
    """
    inv_gamma = gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 * brightness for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def perturb_motion_blur(img, kernel_size=15):
    """
    Simulates horizontal motion blur caused by vehicle velocity and roadway roughness.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)


def perturb_haze_fog(img, t=0.50, atmospheric_light=225):
    """
    Simulates adverse weather fog and haze using Koschmieder's Law:
    I'(x) = I(x) * t + A * (1 - t)
    """
    fogged = img.astype(np.float32) * t + atmospheric_light * (1.0 - t)
    return np.clip(fogged, 0, 255).astype(np.uint8)


def perturb_sensor_noise(img, sigma=25):
    """
    Simulates low-cost dashcam sensor noise under high ISO gain.
    I'(x) = clip(I(x) + N(0, sigma^2), 0, 255)
    """
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


PERTURBATIONS = {
    'Clean (Reference)': {
        'func': perturb_clean,
        'description': 'Unaltered validation test images',
        'color': '#2b5c8f'
    },
    'Low-Light / Night': {
        'func': perturb_low_light,
        'description': 'Gamma darkening (g=2.0, scale=0.45) simulating unlit nighttime',
        'color': '#4a148c'
    },
    'Motion Blur': {
        'func': perturb_motion_blur,
        'description': 'Horizontal 1D kernel (k=15) simulating vehicle vibration',
        'color': '#e65100'
    },
    'Adverse Weather (Haze/Fog)': {
        'func': perturb_haze_fog,
        'description': 'Atmospheric scattering (t=0.50, A=225) simulating heavy mist/fog',
        'color': '#00838f'
    },
    'Sensor Noise': {
        'func': perturb_sensor_noise,
        'description': 'Zero-mean Gaussian noise (sigma=25) simulating high-ISO dashcam CMOS',
        'color': '#c2185b'
    }
}


def create_perturbation_mosaic(sample_img_paths, out_path):
    """
    Generates a 3x5 multi-panel figure displaying 3 sample damaged roads under all 5 conditions.
    """
    num_samples = len(sample_img_paths)
    num_conditions = len(PERTURBATIONS)
    
    fig, axes = plt.subplots(num_samples, num_conditions, figsize=(18, 3.8 * num_samples), dpi=300)
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    for row_idx, img_path in enumerate(sample_img_paths):
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        for col_idx, (cond_name, cond_info) in enumerate(PERTURBATIONS.items()):
            ax = axes[row_idx, col_idx]
            perturbed = cond_info['func'](rgb)
            ax.imshow(perturbed)
            
            if row_idx == 0:
                ax.set_title(cond_name, fontsize=11, fontweight='bold', pad=8, color=cond_info['color'])
            if col_idx == 0:
                ax.set_ylabel(f"Sample #{row_idx + 1}\n{Path(img_path).name[:14]}", fontsize=9, fontweight='bold')
                
            ax.axis('off')
            
    plt.suptitle("CDS540 Municipal Dashcam Robustness Benchmark: Simulated Adverse Road Conditions",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Perturbation mosaic saved to {out_path}")


def evaluate_perturbations(model_path, test_csv, sample_size=400, seed=42):
    """
    Subsets test set, creates perturbed datasets, and evaluates model.
    """
    np.random.seed(seed)
    df_test = pd.read_csv(test_csv)
    
    # Stratified selection: ensure representation of positive damage frames and negative backgrounds
    df_pos = df_test[df_test['has_object'] == True]
    df_neg = df_test[df_test['has_object'] == False]
    
    pos_sample_size = min(len(df_pos), int(sample_size * 0.75))
    neg_sample_size = min(len(df_neg), sample_size - pos_sample_size)
    
    sample_df = pd.concat([
        df_pos.sample(n=pos_sample_size, random_state=seed),
        df_neg.sample(n=neg_sample_size, random_state=seed)
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print(f"[INFO] Robustness subset constructed: {len(sample_df)} images ({pos_sample_size} positive, {neg_sample_size} negative)")
    
    # Working directory for robustness test
    rob_dir = Path("data/robustness_test")
    rob_images_dir = rob_dir / "images"
    rob_labels_dir = rob_dir / "labels"
    
    if rob_dir.exists():
        shutil.rmtree(rob_dir)
    rob_images_dir.mkdir(parents=True, exist_ok=True)
    rob_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Symlink/copy labels once (geometry does not change)
    for _, row in sample_df.iterrows():
        img_name = Path(row['image_path']).name
        stem = Path(img_name).stem
        src_label = Path(f"data/yolo/labels/test/{stem}.txt")
        dst_label = rob_labels_dir / f"{stem}.txt"
        
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.touch()
            
    # Write temporary YAML config
    rob_yaml = rob_dir / "robustness_data.yaml"
    with open(rob_yaml, 'w', encoding='utf-8') as f:
        f.write(f"""path: {rob_dir.resolve().as_posix()}
train: images
val: images
test: images

names:
  0: D00 # Longitudinal Crack
  1: D10 # Transverse Crack
  2: D20 # Alligator Crack
  3: D40 # Pothole
""")
        
    model = YOLO(model_path)
    records = []
    clean_metrics = {}
    
    for cond_name, cond_info in PERTURBATIONS.items():
        print(f"\n---> Evaluating condition: {cond_name}...")
        
        # Clear images folder and populate with perturbed images
        for f in rob_images_dir.glob("*.jpg"):
            f.unlink()
            
        p_func = cond_info['func']
        for _, row in sample_df.iterrows():
            orig_img_path = Path("data/yolo/images/test") / Path(row['image_path']).name
            if not orig_img_path.exists():
                orig_img_path = Path(row['image_path'])
            img = cv2.imread(str(orig_img_path))
            if img is None:
                continue
            perturbed = p_func(img)
            cv2.imwrite(str(rob_images_dir / orig_img_path.name), perturbed)
            
        # Run YOLO evaluation
        exp_name = f"rob_{cond_name.split()[0].lower()}"
        metrics = model.val(
            data=str(rob_yaml),
            split='val',
            batch=16,
            imgsz=640,
            device=0,
            workers=0,
            project='runs/detect/robustness',
            name=exp_name,
            exist_ok=True,
            verbose=False
        )
        
        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        
        class_map50 = metrics.box.ap50.tolist() if hasattr(metrics.box, 'ap50') else []
        class_r = metrics.box.r.tolist() if hasattr(metrics.box, 'r') else []
        
        d00_ap = class_map50[0] if len(class_map50) > 0 else 0.0
        d10_ap = class_map50[1] if len(class_map50) > 1 else 0.0
        d20_ap = class_map50[2] if len(class_map50) > 2 else 0.0
        d40_ap = class_map50[3] if len(class_map50) > 3 else 0.0
        d40_recall = class_r[3] if len(class_r) > 3 else 0.0
        
        if cond_name == 'Clean (Reference)':
            clean_metrics = {
                'mAP@50': map50,
                'Recall': mr,
                'D40_Recall': d40_recall
            }
            map_decay = 0.0
            d40_decay = 0.0
        else:
            ref_map = clean_metrics.get('mAP@50', 1.0)
            ref_d40 = clean_metrics.get('D40_Recall', 1.0)
            map_decay = ((map50 - ref_map) / ref_map) * 100.0 if ref_map > 0 else 0.0
            d40_decay = ((d40_recall - ref_d40) / ref_d40) * 100.0 if ref_d40 > 0 else 0.0
            
        records.append({
            'Condition': cond_name,
            'Description': cond_info['description'],
            'Precision': round(mp, 4),
            'Recall': round(mr, 4),
            'mAP@50': round(map50, 4),
            'mAP@50-95': round(map50_95, 4),
            'D00_AP50': round(d00_ap, 4),
            'D10_AP50': round(d10_ap, 4),
            'D20_AP50': round(d20_ap, 4),
            'D40_AP50': round(d40_ap, 4),
            'D40_Recall': round(d40_recall, 4),
            'mAP@50_Decay (%)': f"{map_decay:+0.2f}%",
            'D40_Recall_Decay (%)': f"{d40_decay:+0.2f}%"
        })
        
    # Clean up temporary test files
    shutil.rmtree(rob_dir)
    return pd.DataFrame(records), sample_df


def plot_robustness_chart(df, save_path):
    """
    Plots multi-bar degradation chart across adverse perturbations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    conditions = df['Condition'].tolist()
    # Shorten condition labels for clean plotting
    short_labels = [c.split(' / ')[0].split(' (')[0] for c in conditions]
    
    x = np.arange(len(short_labels))
    width = 0.35
    
    map50 = df['mAP@50'].tolist()
    d40_rec = df['D40_Recall'].tolist()
    
    # ---------------- Panel A: Metric Values ----------------
    r1 = ax1.bar(x - width/2, map50, width, label='Overall mAP@50', color='#2b5c8f', alpha=0.9, edgecolor='black', linewidth=0.8)
    r2 = ax1.bar(x + width/2, d40_rec, width, label='Critical D40 (Pothole) Recall', color='#d95f02', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax1.set_title('A. Absolute Detection Performance Under Perturbations', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=10, fontweight='bold', rotation=15)
    ax1.set_ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 0.65)
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    
    for r in r1:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for r in r2:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ---------------- Panel B: Relative Decay (%) vs Clean Baseline ----------------
    map_decay_vals = [float(val.replace('%', '')) for val in df['mAP@50_Decay (%)']]
    d40_decay_vals = [float(val.replace('%', '')) for val in df['D40_Recall_Decay (%)']]
    
    ax2.bar(x - width/2, map_decay_vals, width, label='mAP@50 Relative Decay (%)', color='#7570b3', alpha=0.9, edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, d40_decay_vals, width, label='D40 Recall Relative Decay (%)', color='#e7298a', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_title('B. Relative Percentage Degradation vs Clean Reference', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=10, fontweight='bold', rotation=15)
    ax2.set_ylabel('Relative Change (%)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True, fontsize=10)
    
    for idx, (v1, v2) in enumerate(zip(map_decay_vals, d40_decay_vals)):
        if idx > 0:
            ax2.annotate(f'{v1:+.1f}%', xy=(x[idx] - width/2, v1),
                         xytext=(0, -12 if v1 < 0 else 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
            ax2.annotate(f'{v2:+.1f}%', xy=(x[idx] + width/2, v2),
                         xytext=(0, -12 if v2 < 0 else 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Robustness degradation chart saved to {save_path}")


def main():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    
    test_csv = "data/manifests/test.csv"
    best_model = "models/compare_yolo11s_best.pt"
    out_csv = "results/robustness.csv"
    eval_fig_dir = Path("results/evaluations")
    eval_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pick 3 representative sample images for the visual mosaic
    df_test = pd.read_csv(test_csv)
    sample_rows = df_test[df_test['has_object'] == True].head(3)
    sample_paths = [Path("data/yolo/images/test") / Path(p).name for p in sample_rows['image_path']]
    
    create_perturbation_mosaic(
        sample_img_paths=sample_paths,
        out_path=eval_fig_dir / "07_robustness_perturbation_mosaic.png"
    )
    
    # 2. Run robustness evaluation across conditions
    df_results, _ = evaluate_perturbations(
        model_path=best_model,
        test_csv=test_csv,
        sample_size=350,
        seed=42
    )
    
    df_results.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Robustness benchmark saved to {out_csv}")
    print(df_results.to_string())
    
    # 3. Plot publication-grade decay chart
    plot_robustness_chart(df_results, eval_fig_dir / "08_robustness_degradation.png")


if __name__ == '__main__':
    main()
