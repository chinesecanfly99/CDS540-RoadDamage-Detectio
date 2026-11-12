"""
Multi-Scale Feature Hierarchy (P3/P4/P5) Activation Energy Analysis (v2).
Course: CDS540 Computer Vision (C-Tier Upgrade C2)

Directly addresses Oral Defense Q2 (how fine cracks survive stride-32 downsampling)
using forward hooks to automatically discover P3 (stride 8), P4 (stride 16), and
P5 (stride 32) feature maps fed into the YOLO11 Detect head (zero hardcoded layer
indices, zero third-party CAM dependencies).

DISCLAIMER: Activation energy magnitude is correlative localization evidence of
multi-scale feature responsiveness, NOT causal proof of detection decisions.

Outputs:
  - results/v2/cam/detected_layers.json
  - results/v2/cam/cam_response_by_class.csv
  - results/v2/cam/cam_summary.json
  - results/v2/cam/sample_panels/*.png (>= 20 sample panels including >= 3 negatives)
  - results/v2/cam/12_layer_cam_response.png (300 DPI, English-only)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

CANONICAL_NAME_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
RAW_TYPO_MAP = {"D0w0": "D00"}
CLASSES_ORDER = ["D00", "D10", "D20", "D40", "BG_NEG"]


def parse_voc_targets(xml_path: Path) -> list[dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    targets: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        cls_name = RAW_TYPO_MAP.get(name_el.text.strip(), name_el.text.strip())
        if cls_name not in CANONICAL_NAME_TO_ID:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.findtext("xmin", "0"))
        y1 = float(bnd.findtext("ymin", "0"))
        x2 = float(bnd.findtext("xmax", "0"))
        y2 = float(bnd.findtext("ymax", "0"))
        if x2 <= x1 or y2 <= y1:
            continue
        targets.append(
            {
                "class_code": cls_name,
                "class_id": CANONICAL_NAME_TO_ID[cls_name],
                "box": [x1, y1, x2, y2],
            }
        )
    return targets


def discover_pyramid_layers(
    detection_model: torch.nn.Module, input_size: int = 640, device: str = "cpu"
) -> list[dict[str, Any]]:
    """
    Dynamically probe `model.model` via forward hooks with a dummy (1, 3, 640, 640)
    tensor to discover the exact stride-8 (P3), stride-16 (P4), and stride-32 (P5)
    layers feeding the Detect head, without hardcoding any module indices.
    """
    detection_model.eval()
    recorded: list[dict[str, Any]] = []
    detect_input_ids: set[int] = set()
    hooks: list[Any] = []

    top_seq = getattr(detection_model, "model", None)
    modules_to_probe = list(top_seq.named_children()) if top_seq is not None else list(detection_model.named_modules())

    def make_hook(name: str, mod: torch.nn.Module):
        def hook_fn(_module: torch.nn.Module, inputs: Any, output: Any) -> None:
            if mod.__class__.__name__ == "Detect":
                # Capture the exact feature tensor ids received by Detect
                for inp in inputs:
                    if isinstance(inp, (list, tuple)):
                        for t in inp:
                            if isinstance(t, torch.Tensor):
                                detect_input_ids.add(id(t))
                    elif isinstance(inp, torch.Tensor):
                        detect_input_ids.add(id(inp))
                return

            if isinstance(output, torch.Tensor) and output.ndim == 4:
                _, c, fh, fw = output.shape
                if fh > 0 and input_size % fh == 0 and fw > 0 and input_size % fw == 0:
                    stride_h = input_size // fh
                    stride_w = input_size // fw
                    if stride_h == stride_w and stride_h in (8, 16, 32):
                        recorded.append(
                            {
                                "layer_name": f"model.{name}" if top_seq is not None else name,
                                "module_index": int(name) if name.isdigit() else -1,
                                "module_type": mod.__class__.__name__,
                                "stride": int(stride_h),
                                "channels": int(c),
                                "feature_h": int(fh),
                                "feature_w": int(fw),
                                "tensor_id": id(output),
                            }
                        )

        return hook_fn

    for name, mod in modules_to_probe:
        hooks.append(mod.register_forward_hook(make_hook(name, mod)))

    try:
        dummy = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32, device=device)
        with torch.no_grad():
            _ = detection_model(dummy)
    finally:
        for h in hooks:
            h.remove()

    # Prefer exact tensors consumed by Detect head; fallback to last observed module per stride
    by_stride: dict[int, dict[str, Any]] = {}
    for rec in recorded:
        if rec["tensor_id"] in detect_input_ids:
            by_stride[rec["stride"]] = rec

    if set(by_stride.keys()) != {8, 16, 32}:
        by_stride = {}
        for rec in recorded:
            by_stride[rec["stride"]] = rec

    if set(by_stride.keys()) != {8, 16, 32}:
        raise RuntimeError(f"Failed to discover stride 8/16/32 layers; found strides={sorted(by_stride.keys())}")

    result = []
    for label, stride in (("P3", 8), ("P4", 16), ("P5", 32)):
        info = dict(by_stride[stride])
        info.pop("tensor_id", None)
        info["pyramid_level"] = label
        result.append(info)
    return result


def letterbox_image(img_bgr: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Standard YOLO letterbox resize matching ultralytics stride-32 preprocessing."""
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


def compute_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1e-6, area_a + area_b - inter)


def select_stratified_cam_samples(df_test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Select >= 20 stratified samples covering D00, D10, D20, D40 plus >= 3 pure negatives."""
    rng = np.random.default_rng(seed)
    obj_df = df_test[df_test["has_object"] == True].copy()  # noqa: E712
    neg_df = df_test[df_test["has_object"] == False].copy()  # noqa: E712

    selected: list[int] = []
    used: set[int] = set()
    for cls_code, quota in [("D40", 6), ("D20", 5), ("D10", 5), ("D00", 5)]:
        sub = obj_df[obj_df["canonical_classes"].astype(str).str.contains(cls_code, na=False)]
        pool = [idx for idx in sub.index.tolist() if idx not in used]
        take = min(quota, len(pool))
        if take > 0:
            chosen = rng.choice(pool, size=take, replace=False).tolist()
            selected.extend(chosen)
            used.update(chosen)

    neg_pool = neg_df.index.tolist()
    neg_chosen = rng.choice(neg_pool, size=min(4, len(neg_pool)), replace=False).tolist()
    selected.extend(neg_chosen)
    return df_test.loc[selected].reset_index(drop=True)


def render_sample_panel(
    img_bgr: np.ndarray,
    energy_maps: dict[str, np.ndarray],
    targets: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    image_id: str,
    out_path: Path,
) -> None:
    """Save 4-panel figure: Original + GT/Pred | P3 Energy | P4 Energy | P5 Energy (English-only)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)

    axes[0].imshow(img_rgb)
    for t in targets:
        x1, y1, x2, y2 = t["box"]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00e5ff", linewidth=1.6)
        axes[0].add_patch(rect)
        axes[0].text(x1, max(10, y1 - 3), f"GT:{t['class_code']}", color="#00e5ff", fontsize=7, weight="bold")
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#ff3d00", linewidth=1.4, linestyle="--")
        axes[0].add_patch(rect)
        axes[0].text(x1, min(img_rgb.shape[0] - 4, y2 + 10), f"{p['class_code']} {p['conf']:.2f}", color="#ff3d00", fontsize=7)
    axes[0].set_title(f"{image_id}\nGT (cyan) vs Pred (red)", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for ax, (lvl, stride) in zip(axes[1:], [("P3", 8), ("P4", 16), ("P5", 32)]):
        emap = energy_maps[lvl]
        emap_vis = emap / max(1e-6, float(emap.max()))
        ax.imshow(img_rgb, alpha=0.55)
        ax.imshow(emap_vis, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        for t in targets:
            x1, y1, x2, y2 = t["box"]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1.0, linestyle=":")
            ax.add_patch(rect)
        ax.set_title(f"{lvl} Activation Energy (stride {stride})", fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_cam_analysis(
    weights_path: Path = ROOT / "models/compare_yolo11s_best.pt",
    seed: int = 42,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    out_dir = ROOT / "results/v2/cam"
    panels_dir = out_dir / "sample_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(weights_path))
    det_model = yolo.model.to(device)
    det_model.eval()

    layers_info = discover_pyramid_layers(det_model, input_size=640, device=device)
    (out_dir / "detected_layers.json").write_text(json.dumps(layers_info, indent=2), encoding="utf-8")

    # Map module_index or layer_name to hook
    top_seq = getattr(det_model, "model", None)
    name_to_level = {item["layer_name"]: item["pyramid_level"] for item in layers_info}
    p3_layer_name = next(item["layer_name"] for item in layers_info if item["pyramid_level"] == "P3")
    p4_layer_name = next(item["layer_name"] for item in layers_info if item["pyramid_level"] == "P4")
    p5_layer_name = next(item["layer_name"] for item in layers_info if item["pyramid_level"] == "P5")

    df_test = pd.read_csv(ROOT / "data/manifests/test.csv")
    sampled = select_stratified_cam_samples(df_test, seed=seed)

    captured_feats: dict[str, torch.Tensor] = {}
    hooks = []
    modules_list = list(top_seq.named_children()) if top_seq is not None else list(det_model.named_modules())
    for name, mod in modules_list:
        full_name = f"model.{name}" if top_seq is not None else name
        if full_name in name_to_level:
            lvl = name_to_level[full_name]

            def _make_cap(level_key: str):
                def _cap(_m: torch.nn.Module, _i: Any, out: Any) -> None:
                    captured_feats[level_key] = out.detach()

                return _cap

            hooks.append(mod.register_forward_hook(_make_cap(lvl)))

    rows: list[dict[str, Any]] = []
    panel_paths: list[str] = []

    try:
        for _, row in sampled.iterrows():
            img_path = ROOT / str(row["image_path"])
            xml_path = ROOT / str(row["xml_path"])
            image_id = img_path.stem

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            h, w = img_bgr.shape[:2]

            targets = parse_voc_targets(xml_path) if bool(row["has_object"]) else []

            # Run Ultralytics prediction on original image for detection matching
            pred_res = yolo.predict(img_bgr, conf=0.25, imgsz=640, device=device, verbose=False)[0]
            preds: list[dict[str, Any]] = []
            if pred_res.boxes is not None and len(pred_res.boxes) > 0:
                xyxy_arr = pred_res.boxes.xyxy.cpu().numpy()
                cls_arr = pred_res.boxes.cls.cpu().numpy()
                conf_arr = pred_res.boxes.conf.cpu().numpy()
                inv_cls = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
                for bx, cid, cf in zip(xyxy_arr, cls_arr, conf_arr):
                    preds.append(
                        {
                            "box": [float(v) for v in bx],
                            "class_code": inv_cls.get(int(cid), "D00"),
                            "conf": float(cf),
                        }
                    )

            # Run deterministic letterbox forward pass to extract P3/P4/P5 energy maps aligned to original image
            lb_bgr, scale, (pad_left, pad_top) = letterbox_image(img_bgr, 640)
            lb_rgb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
            inp_tensor = torch.from_numpy(lb_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)

            captured_feats.clear()
            with torch.no_grad():
                _ = det_model(inp_tensor)

            # Compute RMS channel activation energy per level, upsample to 640x640, unpad, resize to (h, w),
            # then per-image spatial L2-normalize so energy across scales is commensurable.
            energy_maps: dict[str, np.ndarray] = {}
            unpad_h = int(round(h * scale))
            unpad_w = int(round(w * scale))
            for lvl in ("P3", "P4", "P5"):
                feat = captured_feats[lvl]  # (1, C, fh, fw)
                rms_map = torch.sqrt(torch.mean(feat.pow(2), dim=1, keepdim=True))  # (1, 1, fh, fw)
                up_640 = F.interpolate(rms_map, size=(640, 640), mode="bilinear", align_corners=False)[0, 0]
                cropped = up_640[pad_top : pad_top + unpad_h, pad_left : pad_left + unpad_w]
                up_orig = F.interpolate(
                    cropped.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu().numpy()
                # Per-image spatial L2 normalization (mean-root-square = 1 per pixel average)
                l2_norm = float(np.sqrt(np.mean(up_orig.astype(np.float64) ** 2))) + 1e-8
                energy_maps[lvl] = (up_orig / l2_norm).astype(np.float32)

            if targets:
                for b_idx, t in enumerate(targets):
                    x1, y1, x2, y2 = t["box"]
                    ix1 = max(0, min(w - 1, int(math.floor(x1))))
                    iy1 = max(0, min(h - 1, int(math.floor(y1))))
                    ix2 = max(ix1 + 1, min(w, int(math.ceil(x2))))
                    iy2 = max(iy1 + 1, min(h, int(math.ceil(y2))))

                    e3 = float(np.mean(energy_maps["P3"][iy1:iy2, ix1:ix2]))
                    e4 = float(np.mean(energy_maps["P4"][iy1:iy2, ix1:ix2]))
                    e5 = float(np.mean(energy_maps["P5"][iy1:iy2, ix1:ix2]))
                    esum = max(1e-8, e3 + e4 + e5)

                    # Match with prediction
                    best_conf = 0.0
                    detected = False
                    for p in preds:
                        if p["class_code"] == t["class_code"] and compute_iou_xyxy(t["box"], p["box"]) >= 0.5:
                            if p["conf"] > best_conf:
                                best_conf = p["conf"]
                                detected = True

                    rows.append(
                        {
                            "weights_version": weights_path.name,
                            "eval_split": "test_stratified_cam_25",
                            "image_id": image_id,
                            "class_code": t["class_code"],
                            "box_id": b_idx,
                            "p3_energy": round(e3, 6),
                            "p4_energy": round(e4, 6),
                            "p5_energy": round(e5, 6),
                            "p3_share": round(e3 / esum, 6),
                            "p4_share": round(e4 / esum, 6),
                            "p5_share": round(e5 / esum, 6),
                            "detected": bool(detected),
                            "conf": round(float(best_conf), 4),
                            "p3_layer": p3_layer_name,
                            "p4_layer": p4_layer_name,
                            "p5_layer": p5_layer_name,
                        }
                    )
            else:
                # Pure negative frame control: measure background central road region response
                e3 = float(np.mean(energy_maps["P3"]))
                e4 = float(np.mean(energy_maps["P4"]))
                e5 = float(np.mean(energy_maps["P5"]))
                esum = max(1e-8, e3 + e4 + e5)
                rows.append(
                    {
                        "weights_version": weights_path.name,
                        "eval_split": "test_stratified_cam_24",
                        "image_id": image_id,
                        "class_code": "BG_NEG",
                        "box_id": 0,
                        "p3_energy": round(e3, 6),
                        "p4_energy": round(e4, 6),
                        "p5_energy": round(e5, 6),
                        "p3_share": round(e3 / esum, 6),
                        "p4_share": round(e4 / esum, 6),
                        "p5_share": round(e5 / esum, 6),
                        "detected": bool(len(preds) > 0),
                        "conf": round(max([p["conf"] for p in preds], default=0.0), 4),
                        "p3_layer": p3_layer_name,
                        "p4_layer": p4_layer_name,
                        "p5_layer": p5_layer_name,
                    }
                )

            panel_path = panels_dir / f"{len(panel_paths)+1:02d}_{image_id}_cam.png"
            render_sample_panel(img_bgr, energy_maps, targets, preds, image_id, panel_path)
            panel_paths.append(str(panel_path.relative_to(ROOT)))
    finally:
        for h in hooks:
            h.remove()

    df_cam = pd.DataFrame(rows)
    if df_cam.isna().any().any():
        raise RuntimeError(f"NaN values detected in cam_response_by_class.csv: {df_cam.isna().sum().to_dict()}")
    df_cam.to_csv(out_dir / "cam_response_by_class.csv", index=False)

    # Class aggregation
    agg_summary: dict[str, Any] = {}
    for cls in CLASSES_ORDER:
        sub = df_cam[df_cam["class_code"] == cls]
        if len(sub) == 0:
            continue
        agg_summary[cls] = {
            "n_boxes": int(len(sub)),
            "p3_share_mean": round(float(sub["p3_share"].mean()), 4),
            "p3_share_std": round(float(sub["p3_share"].std(ddof=1)) if len(sub) > 1 else 0.0, 4),
            "p4_share_mean": round(float(sub["p4_share"].mean()), 4),
            "p4_share_std": round(float(sub["p4_share"].std(ddof=1)) if len(sub) > 1 else 0.0, 4),
            "p5_share_mean": round(float(sub["p5_share"].mean()), 4),
            "p5_share_std": round(float(sub["p5_share"].std(ddof=1)) if len(sub) > 1 else 0.0, 4),
            "p3_energy_mean": round(float(sub["p3_energy"].mean()), 4),
            "p4_energy_mean": round(float(sub["p4_energy"].mean()), 4),
            "p5_energy_mean": round(float(sub["p5_energy"].mean()), 4),
            "detection_rate": round(float(sub["detected"].mean()), 4),
        }

    summary = {
        "weights_version": weights_path.name,
        "n_images_sampled": int(len(sampled)),
        "n_negative_images": int((~sampled["has_object"].astype(bool)).sum()),
        "n_csv_rows": int(len(df_cam)),
        "detected_layers": layers_info,
        "class_aggregation": agg_summary,
        "panels_count": len(panel_paths),
    }
    (out_dir / "cam_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Render Figure 12 (300 DPI, English-only, error bars + sample counts)
    present_classes = [c for c in CLASSES_ORDER if c in agg_summary]
    x = np.arange(len(present_classes))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=300)

    p3_means = [agg_summary[c]["p3_share_mean"] * 100.0 for c in present_classes]
    p3_stds = [agg_summary[c]["p3_share_std"] * 100.0 for c in present_classes]
    p4_means = [agg_summary[c]["p4_share_mean"] * 100.0 for c in present_classes]
    p4_stds = [agg_summary[c]["p4_share_std"] * 100.0 for c in present_classes]
    p5_means = [agg_summary[c]["p5_share_mean"] * 100.0 for c in present_classes]
    p5_stds = [agg_summary[c]["p5_share_std"] * 100.0 for c in present_classes]

    b3 = ax1.bar(x - width, p3_means, width, yerr=p3_stds, capsize=3, label="P3 (stride 8, high-res)", color="#1b9e77", edgecolor="black", alpha=0.9)
    b4 = ax1.bar(x, p4_means, width, yerr=p4_stds, capsize=3, label="P4 (stride 16, mid-res)", color="#7570b3", edgecolor="black", alpha=0.9)
    b5 = ax1.bar(x + width, p5_means, width, yerr=p5_stds, capsize=3, label="P5 (stride 32, semantic)", color="#d95f02", edgecolor="black", alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(n={agg_summary[c]['n_boxes']})" for c in present_classes], fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Normalized Layer Energy Share (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 68)
    ax1.set_title("A. Multi-Scale Energy Share by Defect Class (Mean +/- SD)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    for bars, stds in ((b3, p3_stds), (b4, p4_stds), (b5, p5_stds)):
        for bar, sd in zip(bars, stds):
            val = bar.get_height()
            ax1.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, val + sd),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.2,
            )

    # Panel B: Raw L2-Normalized Regional Energy relative to Frame Mean (=1.0)
    e3_means = [agg_summary[c]["p3_energy_mean"] for c in present_classes]
    e4_means = [agg_summary[c]["p4_energy_mean"] for c in present_classes]
    e5_means = [agg_summary[c]["p5_energy_mean"] for c in present_classes]
    ax2.plot(x, e3_means, marker="o", linewidth=2.2, color="#1b9e77", label="P3 (stride 8)")
    ax2.plot(x, e4_means, marker="s", linewidth=2.2, color="#7570b3", label="P4 (stride 16)")
    ax2.plot(x, e5_means, marker="^", linewidth=2.2, color="#d95f02", label="P5 (stride 32)")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.1, label="Frame L2 Baseline (1.0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n(n={agg_summary[c]['n_boxes']})" for c in present_classes], fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Mean Regional Energy (Normalized to Frame RMS)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Defect vs Background Activation Concentration", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        f"C2: P3/P4/P5 Layer Activation Energy Response ({weights_path.name}, Correlative Evidence)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "12_layer_cam_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze YOLO11s P3/P4/P5 activation energy across defect classes.")
    parser.add_argument("--dry-run", action="store_true", help="Discover P3/P4/P5 layers via forward hook and validate sample counts.")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/compare_yolo11s_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_csv = ROOT / "data/manifests/test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Missing manifest: {test_csv}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing weights: {args.weights}")

    if args.dry_run:
        yolo = YOLO(str(args.weights))
        layers_info = discover_pyramid_layers(yolo.model, input_size=640, device="cpu")
        df_test = pd.read_csv(test_csv)
        sampled = select_stratified_cam_samples(df_test, seed=args.seed)
        plan = {
            "script": "layer_cam_analysis_v2",
            "dry_run": True,
            "weights": str(args.weights.relative_to(ROOT)),
            "discovered_layers": layers_info,
            "sampled_images": int(len(sampled)),
            "negative_samples": int((~sampled["has_object"].astype(bool)).sum()),
            "output_dir": "results/v2/cam",
        }
        print(json.dumps(plan, indent=2))
        return 0

    summary = run_cam_analysis(weights_path=args.weights, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
