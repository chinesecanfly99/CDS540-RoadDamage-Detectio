"""
ONNX Export (FP32 / FP16 / Dynamic INT8), Numerical Parity Verification, and
Inference Latency Benchmark (v2).
Course: CDS540 Computer Vision (C-Tier Upgrade C4)

Addresses Oral Defense Q7 by measuring real export consistency and runtime
latency boundaries rather than extrapolating RTX 5070 numbers to unseen Jetson
hardware.

Outputs:
  - models/v2/exports/yolo11s_fp32.onnx (plus fp16 / int8_dynamic if supported)
  - results/v2/runtime/onnx_parity.json
  - results/v2/runtime/export_benchmark.csv
  - results/v2/runtime/14_export_accuracy_speed.png (300 DPI, English-only)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def letterbox_image(img_bgr: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Deterministic stride-32 letterbox matching Ultralytics inference input."""
    h, w = img_bgr.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    dw = (new_shape - nw) / 2.0
    dh = (new_shape - nh) / 2.0
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, (left, top)


def preprocess_bgr_to_nchw(
    img_bgr: np.ndarray, imgsz: int = 640, dtype: np.dtype = np.float32
) -> tuple[np.ndarray, float, tuple[int, int]]:
    lb, scale, pad = letterbox_image(img_bgr, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    arr = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32) / 255.0
    return arr.astype(dtype, copy=False), scale, pad


def decode_yolo_output_to_orig(
    raw_output: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    conf_thres: float = 0.25,
    iou_thres: float = 0.70,
    max_det: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YOLO11 (1, 4+nc, 8400) output using torchvision batched_nms."""
    pred = torch.from_numpy(raw_output[0].astype(np.float32)).T  # (8400, 4+nc)
    boxes_cxcywh = pred[:, :4]
    cls_probs = pred[:, 4:]
    scores, cls_ids = torch.max(cls_probs, dim=1)
    mask = scores >= conf_thres
    if not torch.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    boxes_xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dim=1)
    keep = batched_nms(boxes_xyxy, scores, cls_ids, iou_thres)[:max_det]

    boxes_out = boxes_xyxy[keep].cpu().numpy()
    pad_x, pad_y = float(pad[0]), float(pad[1])
    orig_h, orig_w = float(orig_shape[0]), float(orig_shape[1])
    boxes_out[:, [0, 2]] = np.clip((boxes_out[:, [0, 2]] - pad_x) / scale, 0.0, orig_w)
    boxes_out[:, [1, 3]] = np.clip((boxes_out[:, [1, 3]] - pad_y) / scale, 0.0, orig_h)
    return (
        boxes_out.astype(np.float32),
        scores[keep].cpu().numpy().astype(np.float32),
        cls_ids[keep].cpu().numpy().astype(np.int64),
    )


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float((a[2] - a[0]) * (a[3] - a[1])))
    area_b = max(0.0, float((b[2] - b[0]) * (b[3] - b[1])))
    return inter / max(1e-6, area_a + area_b - inter)


def select_parity_images(n_images: int = 32, seed: int = 42) -> list[Path]:
    df = pd.read_csv(ROOT / "data/manifests/test.csv")
    obj_df = df[df["has_object"] == True].copy()  # noqa: E712
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(obj_df.index.tolist(), size=min(n_images, len(obj_df)), replace=False).tolist()
    return [ROOT / str(obj_df.loc[i, "image_path"]) for i in chosen_idx]


def verify_onnx_parity(
    pt_model: YOLO,
    onnx_path: Path,
    img_paths: list[Path],
    ep: str,
    input_dtype: np.dtype = np.float32,
    conf_thres: float = 0.25,
    box_tol_px: float = 1.0,
    score_tol: float = 0.01,
) -> dict[str, Any]:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=[ep])
    inp_name = sess.get_inputs()[0].name

    max_box_delta = 0.0
    max_score_delta = 0.0
    mean_box_deltas: list[float] = []
    mean_score_deltas: list[float] = []
    total_pt_dets = 0
    total_ort_dets = 0
    matched_pairs = 0
    unmatched_pt = 0
    unmatched_ort = 0

    for p in img_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        h, w = img_bgr.shape[:2]

        # Run PyTorch using identical letterbox tensor so preprocessing differences do not confound numerical parity
        inp_fp32, scale, pad = preprocess_bgr_to_nchw(img_bgr, 640, np.float32)
        with torch.no_grad():
            pt_raw = pt_model.model(torch.from_numpy(inp_fp32).to("cpu"))
        pt_out = pt_raw[0].cpu().numpy() if isinstance(pt_raw, (list, tuple)) else pt_raw.cpu().numpy()
        pt_boxes, pt_scores, pt_cls = decode_yolo_output_to_orig(pt_out, (h, w), scale, pad, conf_thres=conf_thres)

        ort_inp = inp_fp32.astype(input_dtype, copy=False)
        ort_out = sess.run(None, {inp_name: ort_inp})[0]
        ort_boxes, ort_scores, ort_cls = decode_yolo_output_to_orig(ort_out, (h, w), scale, pad, conf_thres=conf_thres)

        total_pt_dets += len(pt_boxes)
        total_ort_dets += len(ort_boxes)

        used_ort: set[int] = set()
        for i in range(len(pt_boxes)):
            best_j = -1
            best_iou = 0.0
            for j in range(len(ort_boxes)):
                if j in used_ort or int(pt_cls[i]) != int(ort_cls[j]):
                    continue
                iou = compute_iou(pt_boxes[i], ort_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.5:
                used_ort.add(best_j)
                matched_pairs += 1
                b_delta = float(np.max(np.abs(pt_boxes[i] - ort_boxes[best_j])))
                s_delta = float(abs(pt_scores[i] - ort_scores[best_j]))
                mean_box_deltas.append(b_delta)
                mean_score_deltas.append(s_delta)
                max_box_delta = max(max_box_delta, b_delta)
                max_score_delta = max(max_score_delta, s_delta)
            else:
                unmatched_pt += 1
        unmatched_ort += len(ort_boxes) - len(used_ort)

    det_count_diff = abs(total_pt_dets - total_ort_dets)
    passed = (
        max_box_delta <= box_tol_px
        and max_score_delta <= score_tol
        and unmatched_pt == 0
        and unmatched_ort == 0
    )

    failure_reasons = []
    if max_box_delta > box_tol_px:
        failure_reasons.append(f"max_box_delta_px={max_box_delta:.4f} > {box_tol_px}")
    if max_score_delta > score_tol:
        failure_reasons.append(f"max_score_delta={max_score_delta:.6f} > {score_tol}")
    if unmatched_pt > 0 or unmatched_ort > 0:
        failure_reasons.append(f"unmatched detections (pt={unmatched_pt}, ort={unmatched_ort})")

    return {
        "onnx_file": onnx_path.name,
        "execution_provider": ep,
        "n_images": len(img_paths),
        "total_pytorch_detections": total_pt_dets,
        "total_ort_detections": total_ort_dets,
        "detection_count_diff": det_count_diff,
        "matched_pairs": matched_pairs,
        "unmatched_pt": unmatched_pt,
        "unmatched_ort": unmatched_ort,
        "max_box_delta_px": round(max_box_delta, 4),
        "mean_box_delta_px": round(float(np.mean(mean_box_deltas)) if mean_box_deltas else 0.0, 4),
        "max_score_delta": round(max_score_delta, 6),
        "mean_score_delta": round(float(np.mean(mean_score_deltas)) if mean_score_deltas else 0.0, 6),
        "tolerances": {"max_box_delta_px": box_tol_px, "max_score_delta": score_tol},
        "status": "PASS" if passed else "FAIL",
        "failure_reasons": failure_reasons,
    }


def benchmark_ort_session(
    onnx_path: Path,
    img_paths: list[Path],
    ep: str,
    input_dtype: np.dtype = np.float32,
    num_warmup: int = 30,
    num_runs: int = 200,
) -> dict[str, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=[ep])
    inp_name = sess.get_inputs()[0].name

    preloaded = [preprocess_bgr_to_nchw(cv2.imread(str(p)), 640, input_dtype)[0] for p in img_paths[: min(len(img_paths), 40)]]
    for i in range(num_warmup):
        _ = sess.run(None, {inp_name: preloaded[i % len(preloaded)]})

    latencies_ms: list[float] = []
    for i in range(num_runs):
        inp = preloaded[i % len(preloaded)]
        t0 = time.perf_counter()
        _ = sess.run(None, {inp_name: inp})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "fps": round(1000.0 / mean_ms, 1),
    }


def run_export_and_benchmark(
    weights_path: Path = ROOT / "models/compare_yolo11s_best.pt",
    seed: int = 42,
    num_warmup: int = 30,
    num_runs: int = 200,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    exports_dir = ROOT / "models/v2/exports"
    runtime_dir = ROOT / "results/v2/runtime"
    exports_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    available_eps = ort.get_available_providers()
    gpu_ep_available = "CUDAExecutionProvider" in available_eps
    active_ep = "CUDAExecutionProvider" if gpu_ep_available else "CPUExecutionProvider"

    # Copy source weights temporarily inside exports_dir so ultralytics.export never touches models/
    tmp_pt = exports_dir / "_tmp_yolo11s_src.pt"
    shutil.copy2(weights_path, tmp_pt)

    export_params = {"imgsz": 640, "batch": 1, "opset": 17, "simplify": True, "dynamic": False}
    fp32_onnx = exports_dir / "yolo11s_fp32.onnx"
    fp16_onnx = exports_dir / "yolo11s_fp16.onnx"
    int8_onnx = exports_dir / "yolo11s_int8_dynamic.onnx"

    try:
        yolo_src = YOLO(str(tmp_pt))
        exported_fp32 = Path(
            yolo_src.export(
                format="onnx",
                imgsz=export_params["imgsz"],
                batch=export_params["batch"],
                opset=export_params["opset"],
                simplify=export_params["simplify"],
                dynamic=export_params["dynamic"],
                half=False,
            )
        )
        if exported_fp32 != fp32_onnx:
            shutil.move(str(exported_fp32), str(fp32_onnx))

        # Export FP16 ONNX (either via ultralytics half=True on CUDA or onnxconverter/fp16 fallback)
        fp16_export_error = ""
        try:
            if torch.cuda.is_available():
                yolo_fp16 = YOLO(str(tmp_pt))
                exported_fp16 = Path(
                    yolo_fp16.export(
                        format="onnx",
                        imgsz=export_params["imgsz"],
                        batch=export_params["batch"],
                        opset=export_params["opset"],
                        simplify=export_params["simplify"],
                        dynamic=export_params["dynamic"],
                        half=True,
                        device=0,
                    )
                )
                if exported_fp16 != fp16_onnx:
                    shutil.move(str(exported_fp16), str(fp16_onnx))
            else:
                fp16_export_error = "CUDA not available for Ultralytics half=True export"
        except Exception as exc:
            fp16_export_error = f"FP16 export failed: {type(exc).__name__}: {exc}"
    finally:
        if tmp_pt.exists():
            tmp_pt.unlink()

    # Dynamic INT8 quantization via onnxruntime.quantization (using ASCII temp dir to avoid Windows UTF-8 path bugs in shape_inference)
    int8_export_error = ""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_in = Path(tmp_dir) / "fp32.onnx"
            tmp_out = Path(tmp_dir) / "int8_dynamic.onnx"
            shutil.copy2(fp32_onnx, tmp_in)
            quantize_dynamic(
                model_input=str(tmp_in),
                model_output=str(tmp_out),
                weight_type=QuantType.QUInt8,
            )
            shutil.copy2(tmp_out, int8_onnx)
    except Exception as exc:
        int8_export_error = f"quantize_dynamic failed: {type(exc).__name__}: {exc}"

    pt_cpu_model = YOLO(str(weights_path))
    pt_cpu_model.model.to("cpu").eval()
    img_paths = select_parity_images(n_images=32, seed=seed)

    # Read existing PyTorch RTX 5070 GPU baseline from results/runtime/runtime.csv for side-by-side trend reference
    rt_csv = pd.read_csv(ROOT / "results/runtime/runtime.csv")
    yolo11s_row = rt_csv[rt_csv["Model"].astype(str).str.contains("YOLO11s", na=False)].iloc[0]

    benchmark_rows: list[dict[str, Any]] = [
        {
            "weights_version": weights_path.name,
            "eval_split": "baseline_runtime_csv (100 frames)",
            "runtime": "PyTorch 2.14 (from runtime.csv)",
            "precision": "FP32",
            "ep": "CUDA (RTX 5070 Laptop GPU)",
            "model_size_mb": round(float(yolo11s_row["Model Size (MB)"]), 2),
            "mean_ms": round(float(yolo11s_row["Pure Infer Mean (ms)"]), 2),
            "p50_ms": round(float(yolo11s_row["Pure Infer P50 (ms)"]), 2),
            "p95_ms": round(float(yolo11s_row["Pure Infer P95 (ms)"]), 2),
            "fps": round(float(yolo11s_row["Pure Infer FPS"]), 1),
            "max_box_delta_px": 0.0,
            "max_score_delta": 0.0,
            "not_benchmarked_reason": "",
            "notes": "Baseline reference from results/runtime/runtime.csv (GPU CUDA, 100 frames vs ONNX 200 frames); frame count and runtime/EP differ, trend reference only.",
        }
    ]

    parity_results: dict[str, Any] = {
        "source_weights": str(weights_path.relative_to(ROOT)),
        "export_params": export_params,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "available_execution_providers": available_eps,
        "active_ep_for_onnx": active_ep,
        "variants": {},
    }

    # Evaluate FP32 ONNX
    fp32_size_mb = round(fp32_onnx.stat().st_size / (1024 * 1024), 2)
    fp32_parity = verify_onnx_parity(pt_cpu_model, fp32_onnx, img_paths, ep="CPUExecutionProvider", input_dtype=np.float32)
    fp32_bench = benchmark_ort_session(fp32_onnx, img_paths, ep="CPUExecutionProvider", input_dtype=np.float32, num_warmup=num_warmup, num_runs=num_runs)
    parity_results["variants"]["fp32"] = fp32_parity
    benchmark_rows.append(
        {
            "weights_version": fp32_onnx.name,
            "eval_split": f"test_parity_32 ({num_runs} frames bench)",
            "runtime": f"ONNX Runtime {ort.__version__}",
            "precision": "FP32",
            "ep": "CPUExecutionProvider",
            "model_size_mb": fp32_size_mb,
            "mean_ms": fp32_bench["mean_ms"],
            "p50_ms": fp32_bench["p50_ms"],
            "p95_ms": fp32_bench["p95_ms"],
            "fps": fp32_bench["fps"],
            "max_box_delta_px": fp32_parity["max_box_delta_px"],
            "max_score_delta": fp32_parity["max_score_delta"],
            "not_benchmarked_reason": "",
            "notes": f"Parity={fp32_parity['status']}; different runtime & EP (CPU) vs PyTorch GPU baseline, trend reference only.",
        }
    )

    # Evaluate FP16 ONNX (export succeeded? testable on active EP?)
    if fp16_onnx.is_file():
        fp16_size_mb = round(fp16_onnx.stat().st_size / (1024 * 1024), 2)
        if gpu_ep_available:
            try:
                fp16_parity = verify_onnx_parity(pt_cpu_model, fp16_onnx, img_paths, ep="CUDAExecutionProvider", input_dtype=np.float16)
                fp16_bench = benchmark_ort_session(fp16_onnx, img_paths, ep="CUDAExecutionProvider", input_dtype=np.float16, num_warmup=num_warmup, num_runs=num_runs)
                parity_results["variants"]["fp16"] = fp16_parity
                benchmark_rows.append(
                    {
                        "weights_version": fp16_onnx.name,
                        "eval_split": f"test_parity_32 ({num_runs} frames bench)",
                        "runtime": f"ONNX Runtime {ort.__version__}",
                        "precision": "FP16",
                        "ep": "CUDAExecutionProvider",
                        "model_size_mb": fp16_size_mb,
                        "mean_ms": fp16_bench["mean_ms"],
                        "p50_ms": fp16_bench["p50_ms"],
                        "p95_ms": fp16_bench["p95_ms"],
                        "fps": fp16_bench["fps"],
                        "max_box_delta_px": fp16_parity["max_box_delta_px"],
                        "max_score_delta": fp16_parity["max_score_delta"],
                        "not_benchmarked_reason": "",
                        "notes": f"Parity={fp16_parity['status']}.",
                    }
                )
            except Exception as exc:
                reason = f"CUDAExecutionProvider runtime failed on FP16 ONNX: {type(exc).__name__}: {exc}"
                parity_results["variants"]["fp16"] = {"status": "NOT_BENCHMARKED", "reason": reason}
                benchmark_rows.append(
                    {
                        "weights_version": fp16_onnx.name,
                        "eval_split": "export_only",
                        "runtime": f"ONNX Runtime {ort.__version__}",
                        "precision": "FP16",
                        "ep": "CPUExecutionProvider (GPU EP unavailable)",
                        "model_size_mb": fp16_size_mb,
                        "mean_ms": np.nan,
                        "p50_ms": np.nan,
                        "p95_ms": np.nan,
                        "fps": np.nan,
                        "max_box_delta_px": np.nan,
                        "max_score_delta": np.nan,
                        "not_benchmarked_reason": reason,
                        "notes": "Exported FP16 ONNX only; not benchmarked due to EP error.",
                    }
                )
        else:
            # Test briefly whether CPUExecutionProvider supports FP16 or raises NotImplemented
            try:
                fp16_parity = verify_onnx_parity(pt_cpu_model, fp16_onnx, img_paths[:5], ep="CPUExecutionProvider", input_dtype=np.float16)
                # Even if 5 images pass, run full 32 and benchmark if supported
                fp16_parity = verify_onnx_parity(pt_cpu_model, fp16_onnx, img_paths, ep="CPUExecutionProvider", input_dtype=np.float16)
                fp16_bench = benchmark_ort_session(fp16_onnx, img_paths, ep="CPUExecutionProvider", input_dtype=np.float16, num_warmup=num_warmup, num_runs=num_runs)
                parity_results["variants"]["fp16"] = fp16_parity
                benchmark_rows.append(
                    {
                        "weights_version": fp16_onnx.name,
                        "eval_split": f"test_parity_32 ({num_runs} frames bench)",
                        "runtime": f"ONNX Runtime {ort.__version__}",
                        "precision": "FP16",
                        "ep": "CPUExecutionProvider",
                        "model_size_mb": fp16_size_mb,
                        "mean_ms": fp16_bench["mean_ms"],
                        "p50_ms": fp16_bench["p50_ms"],
                        "p95_ms": fp16_bench["p95_ms"],
                        "fps": fp16_bench["fps"],
                        "max_box_delta_px": fp16_parity["max_box_delta_px"],
                        "max_score_delta": fp16_parity["max_score_delta"],
                        "not_benchmarked_reason": "",
                        "notes": f"Parity={fp16_parity['status']} on CPU EP (not hardware FP16 acceleration).",
                    }
                )
            except Exception as exc:
                reason = f"onnxruntime-gpu not installed (CPUExecutionProvider only); FP16 ops unsupported on CPU EP: {type(exc).__name__}: {str(exc).splitlines()[0]}"
                parity_results["variants"]["fp16"] = {"status": "NOT_BENCHMARKED", "reason": reason}
                benchmark_rows.append(
                    {
                        "weights_version": fp16_onnx.name,
                        "eval_split": "export_only",
                        "runtime": f"ONNX Runtime {ort.__version__}",
                        "precision": "FP16",
                        "ep": "CPUExecutionProvider",
                        "model_size_mb": fp16_size_mb,
                        "mean_ms": np.nan,
                        "p50_ms": np.nan,
                        "p95_ms": np.nan,
                        "fps": np.nan,
                        "max_box_delta_px": np.nan,
                        "max_score_delta": np.nan,
                        "not_benchmarked_reason": reason,
                        "notes": "Exported FP16 ONNX artifact successfully; benchmark skipped on CPU EP per C4 fallback rule.",
                    }
                )
    else:
        parity_results["variants"]["fp16"] = {"status": "EXPORT_FAILED", "reason": fp16_export_error}
        benchmark_rows.append(
            {
                "weights_version": "yolo11s_fp16.onnx",
                "eval_split": "not_exported",
                "runtime": f"ONNX Runtime {ort.__version__}",
                "precision": "FP16",
                "ep": "N/A",
                "model_size_mb": np.nan,
                "mean_ms": np.nan,
                "p50_ms": np.nan,
                "p95_ms": np.nan,
                "fps": np.nan,
                "max_box_delta_px": np.nan,
                "max_score_delta": np.nan,
                "not_benchmarked_reason": fp16_export_error,
                "notes": "FP16 export failed.",
            }
        )

    # Evaluate INT8 Dynamic Quantized ONNX
    if int8_onnx.is_file():
        int8_size_mb = round(int8_onnx.stat().st_size / (1024 * 1024), 2)
        try:
            int8_parity = verify_onnx_parity(pt_cpu_model, int8_onnx, img_paths, ep="CPUExecutionProvider", input_dtype=np.float32)
            int8_bench = benchmark_ort_session(int8_onnx, img_paths, ep="CPUExecutionProvider", input_dtype=np.float32, num_warmup=num_warmup, num_runs=num_runs)
            parity_results["variants"]["int8_dynamic"] = int8_parity
            benchmark_rows.append(
                {
                    "weights_version": int8_onnx.name,
                    "eval_split": f"test_parity_32 ({num_runs} frames bench)",
                    "runtime": f"ONNX Runtime {ort.__version__}",
                    "precision": "INT8_dynamic",
                    "ep": "CPUExecutionProvider",
                    "model_size_mb": int8_size_mb,
                    "mean_ms": int8_bench["mean_ms"],
                    "p50_ms": int8_bench["p50_ms"],
                    "p95_ms": int8_bench["p95_ms"],
                    "fps": int8_bench["fps"],
                    "max_box_delta_px": int8_parity["max_box_delta_px"],
                    "max_score_delta": int8_parity["max_score_delta"],
                    "not_benchmarked_reason": "",
                    "notes": f"Parity={int8_parity['status']}; dynamic INT8 quantization on CPU EP.",
                }
            )
        except Exception as exc:
            reason = f"INT8 dynamic ONNX execution failed on CPU EP: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            parity_results["variants"]["int8_dynamic"] = {"status": "NOT_BENCHMARKED", "reason": reason}
            benchmark_rows.append(
                {
                    "weights_version": int8_onnx.name,
                    "eval_split": "export_only",
                    "runtime": f"ONNX Runtime {ort.__version__}",
                    "precision": "INT8_dynamic",
                    "ep": "CPUExecutionProvider",
                    "model_size_mb": int8_size_mb,
                    "mean_ms": np.nan,
                    "p50_ms": np.nan,
                    "p95_ms": np.nan,
                    "fps": np.nan,
                    "max_box_delta_px": np.nan,
                    "max_score_delta": np.nan,
                    "not_benchmarked_reason": reason,
                    "notes": "Quantized artifact created; inference failed due to unsupported integer operator.",
                }
            )
    else:
        parity_results["variants"]["int8_dynamic"] = {"status": "EXPORT_FAILED", "reason": int8_export_error}
        benchmark_rows.append(
            {
                "weights_version": "yolo11s_int8_dynamic.onnx",
                "eval_split": "not_exported",
                "runtime": f"ONNX Runtime {ort.__version__}",
                "precision": "INT8_dynamic",
                "ep": "CPUExecutionProvider",
                "model_size_mb": np.nan,
                "mean_ms": np.nan,
                "p50_ms": np.nan,
                "p95_ms": np.nan,
                "fps": np.nan,
                "max_box_delta_px": np.nan,
                "max_score_delta": np.nan,
                "not_benchmarked_reason": int8_export_error,
                "notes": "Dynamic quantization failed.",
            }
        )

    df_bench = pd.DataFrame(benchmark_rows)
    df_bench.to_csv(runtime_dir / "export_benchmark.csv", index=False)
    (runtime_dir / "onnx_parity.json").write_text(json.dumps(parity_results, indent=2), encoding="utf-8")

    # Render Figure 14: Model Size / Inference Latency / Numerical Parity 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), dpi=300)

    # Filter entries with valid size
    df_size = df_bench.dropna(subset=["model_size_mb"]).copy()
    labels_size = [f"{r['precision']}\n({r['runtime'].split()[0]})" for _, r in df_size.iterrows()]
    bars_sz = axes[0].bar(range(len(df_size)), df_size["model_size_mb"], color=["#2b5c8f", "#1b9e77", "#7570b3", "#d95f02"][: len(df_size)], edgecolor="black")
    axes[0].set_xticks(range(len(df_size)))
    axes[0].set_xticklabels(labels_size, fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("Model File Size (MB)", fontsize=9.5, fontweight="bold")
    axes[0].set_title("A. Artifact Footprint (YOLO11s)", fontsize=10.5, fontweight="bold")
    for b in bars_sz:
        h_val = b.get_height()
        axes[0].annotate(f"{h_val:.1f} MB", xy=(b.get_x() + b.get_width() / 2, h_val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    # Panel B: Latency (Mean & P95) for benchmarked variants
    df_lat = df_bench.dropna(subset=["mean_ms"]).copy()
    x_lat = np.arange(len(df_lat))
    w_bar = 0.34
    labels_lat = [
        f"{r['precision']}\n({'CUDA' if 'CUDA' in r['ep'] else 'CPU EP'})"
        for _, r in df_lat.iterrows()
    ]
    b_m = axes[1].bar(x_lat - w_bar / 2, df_lat["mean_ms"], w_bar, label="Mean (ms)", color="#2b5c8f", edgecolor="black")
    b_p = axes[1].bar(x_lat + w_bar / 2, df_lat["p95_ms"], w_bar, label="P95 (ms)", color="#e07b39", edgecolor="black")
    axes[1].set_xticks(x_lat)
    axes[1].set_xticklabels(labels_lat, fontsize=8.0, fontweight="bold")
    axes[1].set_ylabel("Pure Inference Latency (ms)", fontsize=9.5, fontweight="bold")
    axes[1].set_title("B. Latency by Runtime & EP\n(Trend Ref Only: GPU vs CPU EP)", fontsize=10.0, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=8, frameon=True)
    for b in list(b_m) + list(b_p):
        h_val = b.get_height()
        axes[1].annotate(f"{h_val:.1f}", xy=(b.get_x() + b.get_width() / 2, h_val), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5)

    # Panel C: Numerical Parity (max score delta & max box delta) on ONNX variants
    df_par = df_bench[(df_bench["runtime"].str.contains("ONNX")) & (df_bench["max_score_delta"].notna())].copy()
    if len(df_par) > 0:
        x_par = np.arange(len(df_par))
        ax_c2 = axes[2].twinx()
        b_bx = axes[2].bar(x_par - 0.18, df_par["max_box_delta_px"], width=0.35, color="#386cb0", edgecolor="black", label="Max Box Delta (px)")
        b_sc = ax_c2.bar(x_par + 0.18, df_par["max_score_delta"], width=0.35, color="#f0027f", edgecolor="black", label="Max Score Delta")
        axes[2].axhline(1.0, color="#386cb0", linestyle="--", linewidth=1.0, label="Box Tol (1.0 px)")
        ax_c2.axhline(0.01, color="#f0027f", linestyle=":", linewidth=1.2, label="Score Tol (0.01)")
        axes[2].set_xticks(x_par)
        axes[2].set_xticklabels([r["precision"] for _, r in df_par.iterrows()], fontsize=9, fontweight="bold")
        axes[2].set_ylabel("Max Box Delta (px)", fontsize=9.5, color="#386cb0")
        ax_c2.set_ylabel("Max Score Delta", fontsize=9.5, color="#f0027f")
        for b in b_bx:
            v = b.get_height()
            axes[2].annotate(f"{v:.2f}px", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
        for b in b_sc:
            v = b.get_height()
            ax_c2.annotate(f"{v:.4f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
    axes[2].set_title("C. PyTorch vs ONNX Parity (n=32 images)", fontsize=10.5, fontweight="bold")

    fig.suptitle(
        "C4: YOLO11s Export Size, Host Latency & Numerical Parity (CPU Execution Provider Only; Edge Hardware Not Benchmarked)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(runtime_dir / "14_export_accuracy_speed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"parity": parity_results, "benchmark_rows": benchmark_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO11s to ONNX variants, check parity, and benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Verify onnx/onnxruntime packages, source weights, and test images.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/compare_yolo11s_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing source weights: {args.weights}")
    if not (ROOT / "results/runtime/runtime.csv").is_file():
        raise FileNotFoundError("Missing baseline results/runtime/runtime.csv")

    if args.dry_run:
        import onnx
        import onnxruntime as ort

        imgs = select_parity_images(n_images=32, seed=args.seed)
        plan = {
            "script": "export_and_benchmark_v2",
            "dry_run": True,
            "weights": str(args.weights.relative_to(ROOT)),
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "parity_images_count": len(imgs),
            "output_dir": "results/v2/runtime",
        }
        print(json.dumps(plan, indent=2))
        return 0

    report = run_export_and_benchmark(weights_path=args.weights, seed=args.seed, num_warmup=args.warmup, num_runs=args.runs)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
