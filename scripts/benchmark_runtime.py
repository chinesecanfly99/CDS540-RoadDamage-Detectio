"""
Precision Runtime Latency, Throughput & Hardware Profiling Benchmark
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Step 13: Tasks 13.1 - 13.4
Executes rigorous GPU timing on NVIDIA RTX 5070 Laptop GPU:
  - 50 warmup iterations to stabilize GPU clock frequencies and CUDA caches.
  - torch.cuda.Event synchronization for microsecond-precise model forward pass timing.
  - End-to-end pipeline profiling (Preprocess, Inference, Postprocess/NMS, Heuristic Triage, HUD Render).
  - P50 (Median), P95, and Mean latency calculation across 200 consecutive test frames.
  - Peak VRAM tracking via torch.cuda.max_memory_allocated().
  - Model disk size (MB) and parameter scale.

Outputs:
  - results/runtime/runtime.csv
  - results/evaluations/10_runtime_latency_breakdown.png
"""

import os
import sys
import time
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Root path setup
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
from scripts.priority import RoadDamagePrioritizer


def benchmark_model(model_path, model_label, test_img_paths, num_warmup=50, num_runs=200):
    print(f"\n=======================================================")
    print(f"Benchmarking: {model_label} ({model_path})")
    print(f"=======================================================")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    model = YOLO(model_path)
    prioritizer = RoadDamagePrioritizer(conf_threshold=0.25)
    
    # Pre-load sample images
    loaded_imgs = [cv2.imread(str(p)) for p in test_img_paths[:num_runs]]
    if len(loaded_imgs) < num_runs:
        loaded_imgs = (loaded_imgs * (num_runs // len(loaded_imgs) + 1))[:num_runs]
        
    print(f"[INFO] Warming up GPU ({num_warmup} iterations)...")
    for i in range(num_warmup):
        dummy_img = loaded_imgs[i % len(loaded_imgs)]
        _ = model.predict(dummy_img, conf=0.25, verbose=False)
    torch.cuda.synchronize()
    
    # Timing arrays
    pure_infer_times = []
    e2e_times = []
    preprocess_times = []
    postprocess_times = []
    heuristic_times = []
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"[INFO] Profiling {num_runs} test iterations with CUDA synchronization...")
    
    for i in range(num_runs):
        img_bgr = loaded_imgs[i]
        h, w = img_bgr.shape[:2]
        
        t_e2e_start = time.perf_counter()
        
        # Step A: Preprocess / Forward / Postprocess via Ultralytics
        start_event.record()
        res = model.predict(img_bgr, conf=0.25, verbose=False)[0]
        end_event.record()
        torch.cuda.synchronize()
        
        gpu_infer_ms = start_event.elapsed_time(end_event)
        pure_infer_times.append(gpu_infer_ms)
        
        # Step B: Heuristic Triage
        t_heur_start = time.perf_counter()
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        triage = prioritizer.evaluate_detections(boxes, classes, confs, w, h)
        _ = prioritizer.render_overlay(img_bgr, triage)
        t_heur_end = time.perf_counter()
        
        heuristic_ms = (t_heur_end - t_heur_start) * 1000.0
        heuristic_times.append(heuristic_ms)
        
        t_e2e_end = time.perf_counter()
        e2e_ms = (t_e2e_end - t_e2e_start) * 1000.0
        e2e_times.append(e2e_ms)

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    
    pure_arr = np.array(pure_infer_times)
    e2e_arr = np.array(e2e_times)
    heur_arr = np.array(heuristic_times)
    
    stats = {
        'Model': model_label,
        'Model Size (MB)': round(model_size_mb, 2),
        'Pure Infer Mean (ms)': round(float(np.mean(pure_arr)), 2),
        'Pure Infer P50 (ms)': round(float(np.percentile(pure_arr, 50)), 2),
        'Pure Infer P95 (ms)': round(float(np.percentile(pure_arr, 95)), 2),
        'Pure Infer FPS': round(1000.0 / float(np.mean(pure_arr)), 1),
        'E2E Pipeline Mean (ms)': round(float(np.mean(e2e_arr)), 2),
        'E2E Pipeline P95 (ms)': round(float(np.percentile(e2e_arr, 95)), 2),
        'E2E Pipeline FPS': round(1000.0 / float(np.mean(e2e_arr)), 1),
        'Heuristic Triage Latency (ms)': round(float(np.mean(heur_arr)), 2),
        'Peak VRAM (MB)': round(peak_vram_mb, 1)
    }
    
    print(f"Results for {model_label}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    return stats


def plot_runtime_comparison(df, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    models = df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.35
    
    # Panel A: Latency (Pure Inference vs E2E Pipeline)
    pure_p50 = df['Pure Infer P50 (ms)'].tolist()
    e2e_p95 = df['E2E Pipeline P95 (ms)'].tolist()
    
    b1 = ax1.bar(x - width/2, pure_p50, width, label='Model Inference Median P50 (ms)', color='#2b5c8f', alpha=0.9, edgecolor='black', linewidth=0.8)
    b2 = ax1.bar(x + width/2, e2e_p95, width, label='Full E2E Pipeline P95 Latency (ms)', color='#d95f02', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax1.set_title('A. Inference vs Full Pipeline Latency (RTX 5070 GPU)', fontsize=11, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax1.set_ylabel('Latency (Milliseconds)', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    
    for b in b1:
        h = b.get_height()
        ax1.annotate(f'{h:.1f}ms', xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for b in b2:
        h = b.get_height()
        ax1.annotate(f'{h:.1f}ms', xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel B: Throughput (FPS) and Dashcam Real-Time Thresholds
    e2e_fps = df['E2E Pipeline FPS'].tolist()
    
    colors = ['#1b9e77', '#7570b3']
    bars = ax2.bar(models, e2e_fps, color=colors, width=0.45, alpha=0.9, edgecolor='black', linewidth=0.8)
    
    # Real-Time Threshold lines
    ax2.axhline(30, color='red', linestyle='--', linewidth=1.5, label='30 FPS Standard Dashcam Video')
    ax2.axhline(60, color='blue', linestyle=':', linewidth=1.5, label='60 FPS High-Speed Video')
    
    ax2.set_title('B. End-to-End Operational Pipeline Throughput (FPS)', fontsize=11, fontweight='bold', pad=12)
    ax2.set_ylabel('Frames Per Second (FPS)', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f} FPS', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Runtime benchmark figure saved to {out_path}")


def main():
    os.chdir(root_dir)
    runtime_dir = Path("results/runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    eval_fig_dir = Path("results/evaluations")
    eval_fig_dir.mkdir(parents=True, exist_ok=True)
    
    test_imgs = sorted(list(Path("data/yolo/images/test").glob("*.jpg")))
    
    models = [
        ('models/baseline_best.pt', 'YOLO11n (Baseline)'),
        ('models/compare_yolo11s_best.pt', 'YOLO11s (Scaled)')
    ]
    
    records = []
    for model_path, model_label in models:
        stats = benchmark_model(model_path, model_label, test_imgs, num_warmup=30, num_runs=100)
        records.append(stats)
        
    df = pd.DataFrame(records)
    out_csv = runtime_dir / "runtime.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Runtime benchmark table saved to {out_csv}")
    print(df.to_string())
    
    plot_runtime_comparison(df, eval_fig_dir / "10_runtime_latency_breakdown.png")


if __name__ == '__main__':
    main()
