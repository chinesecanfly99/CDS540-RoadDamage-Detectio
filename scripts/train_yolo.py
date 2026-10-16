"""
train_yolo.py
--------------------------------------------------------------------------------
Phase 5 / Steps 7-9: Ultralytics YOLO Training & Benchmark Runner.

Supports:
- Task 7.2: Smoke test (yolo11n.pt, epochs=3, batch=4)
- Task 8.1: Formal baseline training (yolo11n.pt, epochs=50, early stopping)
- Task 9.1: Comparative model training (e.g., yolo11s.pt)

Usage:
  # Smoke test:
  python scripts/train_yolo.py --model yolo11n.pt --epochs 3 --batch 4 --name baseline_smoke

  # Baseline training:
  python scripts/train_yolo.py --model yolo11n.pt --epochs 50 --batch 8 --patience 12 --name baseline_n
--------------------------------------------------------------------------------
"""

import os
import argparse
import torch
from ultralytics import YOLO


def train(
    model_name="yolo11n.pt",
    data_yaml="configs/data.yaml",
    epochs=3,
    batch=4,
    imgsz=640,
    device=0,
    workers=0,
    seed=42,
    patience=12,
    project="runs",
    name="baseline_smoke",
    exist_ok=True
):
    print("=" * 70)
    print("  RDD2022 Road Damage Detection: YOLO Training Pipeline")
    print("=" * 70)
    # Handle device formatting (ensure integer for device index)
    cuda_device = int(device) if str(device).isdigit() else device

    print(f"[*] PyTorch Version : {torch.__version__}")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Active GPU      : {torch.cuda.get_device_name(cuda_device)}")
        print(f"[*] GPU Memory      : {torch.cuda.get_device_properties(cuda_device).total_memory / (1024**3):.2f} GB")
    else:
        print("[!] Warning: CUDA is not available! Training will run on CPU.")

    print(f"[*] Model Skeleton  : {model_name}")
    print(f"[*] Dataset Config  : {data_yaml}")
    print(f"[*] Epochs          : {epochs}")
    print(f"[*] Batch Size      : {batch}")
    print(f"[*] Image Size      : {imgsz}")
    print(f"[*] Run Directory   : {project}/{name}")
    print("=" * 70)

    # Instantiate model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=cuda_device,
        workers=workers,
        seed=seed,
        patience=patience,
        project=project,
        name=name,
        exist_ok=exist_ok,
        save=True,
        save_period=-1,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 70)
    print(f"[+] Training run '{name}' finished successfully!")
    print(f"[+] Output artifacts directory: {project}/{name}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on RDD2022")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model checkpoint or yaml")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default=0, help="CUDA device index or 'cpu'")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 recommended for Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs", help="Project save directory")
    parser.add_argument("--name", type=str, default="baseline_smoke", help="Experiment name")

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=args.project,
        name=args.name
    )
