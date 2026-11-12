"""
Compare 2D Pixel Area Ratio vs IPM Estimated Ground Area (v2).
Course: CDS540 Computer Vision (C-Tier Upgrade C1)

Evaluates perspective foreshortening sensitivity on >= 30 annotated images from
`data/manifests/test.csv`, comparing:
  - 2D proxy: `area_ratio_2d = box_area_px / image_area_px`
  - IPM proxy: `estimated_ground_area_m2` under assumed lane geometry

Outputs:
  - results/v2/ipm/area_comparison.csv
  - results/v2/ipm/ipm_summary.json
  - results/v2/ipm/sample_cards/*.jpg (>= 6 bird's-eye comparison cards)
  - results/v2/ipm/11_ipm_birdseye_and_area_comparison.png (300 DPI, English-only)
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
from scipy.stats import kendalltau, spearmanr
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ipm_geometry_v2 import RoadPlaneIPM, canonical_road_quad
from scripts.priority import CLASS_MAP, RoadDamagePrioritizer
from scripts.priority_v2 import RoadDamagePrioritizerV2

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

CANONICAL_NAME_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
RAW_TYPO_MAP = {"D0w0": "D00"}


def parse_voc_targets(xml_path: Path) -> list[dict[str, Any]]:
    """Parse canonical D00/D10/D20/D40 targets from Pascal VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    targets: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        raw_name = name_el.text.strip()
        cls_name = RAW_TYPO_MAP.get(raw_name, raw_name)
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


def select_stratified_test_images(
    df_test: pd.DataFrame, n_images: int = 40, seed: int = 42
) -> pd.DataFrame:
    """Select >= 30 object-bearing test images balanced across countries & classes."""
    rng = np.random.default_rng(seed)
    obj_df = df_test[df_test["has_object"] == True].copy()  # noqa: E712

    selected_indices: list[int] = []
    # Guarantee coverage across D40, D20, D10, D00 and Czech/India
    strata = [
        ("D40", 12),
        ("D20", 10),
        ("D10", 10),
        ("D00", 8),
    ]
    used = set()
    for cls_code, quota in strata:
        sub = obj_df[obj_df["canonical_classes"].astype(str).str.contains(cls_code, na=False)]
        sub_idxs = [idx for idx in sub.index.tolist() if idx not in used]
        if sub_idxs:
            take = min(quota, len(sub_idxs))
            chosen = rng.choice(sub_idxs, size=take, replace=False).tolist()
            selected_indices.extend(chosen)
            used.update(chosen)

    if len(selected_indices) < n_images:
        rem = [idx for idx in obj_df.index.tolist() if idx not in used]
        need = min(n_images - len(selected_indices), len(rem))
        if need > 0:
            chosen = rng.choice(rem, size=need, replace=False).tolist()
            selected_indices.extend(chosen)

    return obj_df.loc[selected_indices].reset_index(drop=True)


def render_comparison_card(
    img_bgr: np.ndarray,
    ipm: RoadPlaneIPM,
    targets: list[dict[str, Any]],
    image_id: str,
    out_path: Path,
) -> None:
    """Save side-by-side original road image & bird's-eye IPM card (English labels)."""
    h, w = img_bgr.shape[:2]
    left = img_bgr.copy()

    # Draw calibrated assumed road trapezoid
    if ipm.src_quad is not None:
        pts = ipm.src_quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(left, [pts], isClosed=True, color=(0, 220, 220), thickness=2)

    bev = ipm.warp_image(img_bgr)
    img_area = float(h * w)

    for t in targets:
        x1, y1, x2, y2 = [int(round(v)) for v in t["box"]]
        area_ratio_2d = max(0.0, (x2 - x1) * (y2 - y1)) / img_area
        est_m2, conf = ipm.estimated_ground_area_m2(t["box"])
        poly = ipm.box_to_ground_polygon(t["box"])

        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(
            left,
            f"{t['class_code']} 2D:{area_ratio_2d:.3f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if poly is not None and math.isfinite(est_m2):
            bev_pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(bev, [bev_pts], isClosed=True, color=(0, 220, 0), thickness=2)
            cx = int(np.clip(np.mean(poly[:, 0]), 8, ipm.bev_width - 140))
            cy = int(np.clip(np.mean(poly[:, 1]), 22, ipm.bev_height - 12))
            cv2.putText(
                bev,
                f"{t['class_code']} est:{est_m2:.2f}m2 (c={conf:.2f})",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (120, 255, 120),
                1,
                cv2.LINE_AA,
            )

    # Resize BEV to match left height for clean side-by-side layout
    bev_resized = cv2.resize(bev, (int(round(ipm.bev_width * (h / ipm.bev_height))), h))
    header_h = 52
    canvas_w = left.shape[1] + bev_resized.shape[1]
    canvas = np.full((h + header_h, canvas_w, 3), 24, dtype=np.uint8)
    canvas[header_h:, : left.shape[1]] = left
    canvas[header_h:, left.shape[1] :] = bev_resized

    cv2.putText(
        canvas,
        f"{image_id} | Original (Assumed Road Quad, 2D ratio)",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"IPM Bird's-Eye View (estimated; lane_w={ipm.lane_width_m}m, L={ipm.lane_length_m}m, h={ipm.camera_height_m}m)",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (180, 230, 255),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_analysis(
    n_images: int = 40,
    seed: int = 42,
    weights_path: Path = ROOT / "models/compare_yolo11s_best.pt",
) -> dict[str, Any]:
    test_csv = ROOT / "data/manifests/test.csv"
    df_test = pd.read_csv(test_csv)
    sampled = select_stratified_test_images(df_test, n_images=n_images, seed=seed)

    out_dir = ROOT / "results/v2/ipm"
    cards_dir = out_dir / "sample_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    v1_prio = RoadDamagePrioritizer(conf_threshold=0.25)

    lane_widths = [3.0, 3.5, 4.0]
    default_lw = 3.5
    lane_length_m = 20.0
    camera_height_m = 1.2

    gt_rows: list[dict[str, Any]] = []
    img_level_rows: list[dict[str, Any]] = []
    sensitivity_rows: dict[float, list[dict[str, Any]]] = {lw: [] for lw in lane_widths}
    vp_attempts = {"ok": 0, "failed": 0, "reasons": {}}
    saved_cards: list[str] = []
    cards_per_country: dict[str, int] = {}

    for row_idx, row in sampled.iterrows():
        img_rel = Path(str(row["image_path"]))
        xml_rel = Path(str(row["xml_path"]))
        img_path = ROOT / img_rel
        xml_path = ROOT / xml_rel
        image_id = img_path.stem

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        h, w = img_bgr.shape[:2]
        img_area = float(h * w)

        targets = parse_voc_targets(xml_path)
        if not targets:
            continue

        # Test vanishing-point calibration diagnostics on each sample (without fallback)
        vp_ipm = RoadPlaneIPM(lane_width_m=default_lw, lane_length_m=lane_length_m, camera_height_m=camera_height_m, seed=seed)
        H_vp, vp_diag = vp_ipm.calibrate_from_vanishing_point(img_bgr)
        if H_vp is not None:
            vp_attempts["ok"] += 1
        else:
            vp_attempts["failed"] += 1
            rsn = vp_diag.get("reason", "unknown")
            vp_attempts["reasons"][rsn] = vp_attempts["reasons"].get(rsn, 0) + 1

        # Canonical assumed trapezoid for consistent cross-image comparison
        quad = canonical_road_quad(img_bgr.shape)
        ipm_default = RoadPlaneIPM(lane_width_m=default_lw, lane_length_m=lane_length_m, camera_height_m=camera_height_m, seed=seed)
        ipm_default.calibrate_from_quad(quad, img_bgr.shape)

        for box_idx, t in enumerate(targets):
            x1, y1, x2, y2 = t["box"]
            box_area_px = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio_2d = box_area_px / img_area
            est_m2, conf = ipm_default.estimated_ground_area_m2(t["box"])
            y_center_norm = (0.5 * (y1 + y2)) / float(h)

            gt_rows.append(
                {
                    "image_id": image_id,
                    "country": str(row["country"]),
                    "weights_version": "GT_XML_canonical",
                    "eval_split": "test_stratified_40",
                    "box_id": box_idx,
                    "class_code": t["class_code"],
                    "y_center_norm": round(y_center_norm, 4),
                    "box_area_px": round(box_area_px, 2),
                    "area_ratio_2d": round(area_ratio_2d, 6),
                    "estimated_ground_area_m2": np.nan if math.isnan(est_m2) else round(est_m2, 6),
                    "confidence": round(conf, 4),
                    "lane_width_m": default_lw,
                    "lane_length_m": lane_length_m,
                    "camera_height_m": camera_height_m,
                }
            )

        # Model inference for priority-level stability comparison (v1 vs v2)
        pred = model.predict(img_bgr, conf=0.25, imgsz=640, verbose=False)[0]
        p_boxes = pred.boxes.xyxy.cpu().numpy()
        p_cls = pred.boxes.cls.cpu().numpy()
        p_conf = pred.boxes.conf.cpu().numpy()

        triage_v1 = v1_prio.evaluate_detections(p_boxes, p_cls, p_conf, w, h)
        per_lw_triage: dict[float, dict[str, Any]] = {}
        for lw in lane_widths:
            ipm_lw = RoadPlaneIPM(lane_width_m=lw, lane_length_m=lane_length_m, camera_height_m=camera_height_m, seed=seed)
            ipm_lw.calibrate_from_quad(quad, img_bgr.shape)
            v2_prio = RoadDamagePrioritizerV2(conf_threshold=0.25, use_ipm=True, ipm=ipm_lw)
            triage_lw = v2_prio.evaluate_detections(p_boxes, p_cls, p_conf, w, h)
            per_lw_triage[lw] = triage_lw
            sensitivity_rows[lw].append(
                {
                    "image_id": image_id,
                    "score": triage_lw["total_score"],
                    "level": triage_lw["priority_level"],
                }
            )

        triage_v2_def = per_lw_triage[default_lw]
        img_level_rows.append(
            {
                "image_id": image_id,
                "country": str(row["country"]),
                "weights_version": "compare_yolo11s_best.pt",
                "eval_split": "test_stratified_40",
                "n_preds": triage_v1["total_distress_count"],
                "v1_score_2d": triage_v1["total_score"],
                "v1_level_2d": triage_v1["priority_level"],
                "v2_score_ipm": triage_v2_def["total_score"],
                "v2_level_ipm": triage_v2_def["priority_level"],
                "level_changed": triage_v1["priority_level"] != triage_v2_def["priority_level"],
                "v2_score_lw30": per_lw_triage[3.0]["total_score"],
                "v2_level_lw30": per_lw_triage[3.0]["priority_level"],
                "v2_score_lw40": per_lw_triage[4.0]["total_score"],
                "v2_level_lw40": per_lw_triage[4.0]["priority_level"],
            }
        )

        # Save >= 8 sample comparison cards across Czech and India
        if len(saved_cards) < 8 and cards_per_country.get(str(row["country"]), 0) < 4:
            card_path = cards_dir / f"{len(saved_cards)+1:02d}_{image_id}_ipm.jpg"
            render_comparison_card(img_bgr, ipm_default, targets, image_id, card_path)
            saved_cards.append(str(card_path.relative_to(ROOT)))
            cards_per_country[str(row["country"])] = cards_per_country.get(str(row["country"]), 0) + 1

    df_area = pd.DataFrame(gt_rows)
    df_area.to_csv(out_dir / "area_comparison.csv", index=False)

    df_img = pd.DataFrame(img_level_rows)
    df_img.to_csv(out_dir / "priority_level_comparison.csv", index=False)

    # Box-level rank correlation on valid IPM boxes
    valid_boxes = df_area.dropna(subset=["estimated_ground_area_m2"]).copy()
    rho_box, p_rho_box = spearmanr(valid_boxes["area_ratio_2d"], valid_boxes["estimated_ground_area_m2"])
    tau_box, p_tau_box = kendalltau(valid_boxes["area_ratio_2d"], valid_boxes["estimated_ground_area_m2"])

    # Top-10 rank difference table (by 2D area vs IPM estimated area)
    valid_boxes = valid_boxes.reset_index(drop=True)
    valid_boxes["rank_2d"] = valid_boxes["area_ratio_2d"].rank(ascending=False, method="min").astype(int)
    valid_boxes["rank_ipm"] = valid_boxes["estimated_ground_area_m2"].rank(ascending=False, method="min").astype(int)
    valid_boxes["rank_shift"] = valid_boxes["rank_2d"] - valid_boxes["rank_ipm"]

    top10_2d_ids = set(valid_boxes.nsmallest(10, "rank_2d")[["image_id", "box_id"]].apply(tuple, axis=1))
    top10_ipm_ids = set(valid_boxes.nsmallest(10, "rank_ipm")[["image_id", "box_id"]].apply(tuple, axis=1))
    top10_overlap = len(top10_2d_ids & top10_ipm_ids)

    top10_diff_df = valid_boxes[
        (valid_boxes["rank_2d"] <= 10) | (valid_boxes["rank_ipm"] <= 10)
    ].sort_values("rank_ipm")[
        ["weights_version", "eval_split", "image_id", "box_id", "class_code", "y_center_norm", "area_ratio_2d", "estimated_ground_area_m2", "rank_2d", "rank_ipm", "rank_shift"]
    ]
    top10_diff_df.to_csv(out_dir / "top10_rank_differences.csv", index=False)

    # Image-level priority ranking & level changes
    rho_img, _ = spearmanr(df_img["v1_score_2d"], df_img["v2_score_ipm"])
    tau_img, _ = kendalltau(df_img["v1_score_2d"], df_img["v2_score_ipm"])
    n_level_changed = int(df_img["level_changed"].sum())
    pct_level_changed = round(100.0 * n_level_changed / max(1, len(df_img)), 2)

    # Sensitivity across lane_width_m = 3.0 / 3.5 / 4.0
    sens_summary = {}
    for lw in lane_widths:
        s_df = pd.DataFrame(sensitivity_rows[lw])
        rho_vs_def, _ = spearmanr(df_img["v2_score_ipm"], s_df["score"])
        tau_vs_def, _ = kendalltau(df_img["v2_score_ipm"], s_df["score"])
        lvl_diff = int((df_img["v2_level_ipm"] != s_df["level"]).sum())
        scale_factor = lw / default_lw
        sens_summary[str(lw)] = {
            "mean_estimated_area_m2": round(float(valid_boxes["estimated_ground_area_m2"].mean() * scale_factor), 4),
            "mean_priority_score": round(float(s_df["score"].mean()), 4),
            "spearman_vs_3.5m": round(float(rho_vs_def), 4),
            "kendall_vs_3.5m": round(float(tau_vs_def), 4),
            "priority_level_changes_vs_3.5m": lvl_diff,
        }

    summary = {
        "n_sampled_images": int(len(df_img)),
        "n_gt_boxes_total": int(len(df_area)),
        "n_gt_boxes_in_calibrated_zone": int(len(valid_boxes)),
        "n_gt_boxes_outside_zone": int(len(df_area) - len(valid_boxes)),
        "box_level_spearman_rho": round(float(rho_box), 4),
        "box_level_kendall_tau": round(float(tau_box), 4),
        "top10_overlap_count": int(top10_overlap),
        "top10_jaccard": round(top10_overlap / max(1, len(top10_2d_ids | top10_ipm_ids)), 4),
        "image_priority_spearman_rho": round(float(rho_img), 4),
        "image_priority_kendall_tau": round(float(tau_img), 4),
        "priority_level_changes_v1_vs_v2": n_level_changed,
        "priority_level_change_pct": pct_level_changed,
        "lane_width_sensitivity": sens_summary,
        "vanishing_point_hough_attempts": vp_attempts,
        "saved_cards": saved_cards,
    }
    (out_dir / "ipm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Render Figure 11 (300 DPI, English-only, no emoji)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=300)

    # Panel A: Sample Bird's-Eye Card embedded
    if saved_cards:
        sample_card_bgr = cv2.imread(str(ROOT / saved_cards[0]))
        sample_card_rgb = cv2.cvtColor(sample_card_bgr, cv2.COLOR_BGR2RGB)
        axes[0].imshow(sample_card_rgb)
    axes[0].set_title("A. Canonical IPM Bird's-Eye Example\n(Assumed Lane Geometry)", fontsize=10.5, fontweight="bold")
    axes[0].axis("off")

    # Panel B: 2D Area Ratio vs IPM Estimated Ground Area (colored by vertical position y_center_norm)
    sc = axes[1].scatter(
        valid_boxes["area_ratio_2d"],
        valid_boxes["estimated_ground_area_m2"],
        c=valid_boxes["y_center_norm"],
        cmap="viridis_r",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.4,
        s=42,
    )
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Vertical Position (y/H)", fontsize=8.5)
    axes[1].set_title(
        f"B. 2D Ratio vs IPM Estimated Area (n={len(valid_boxes)} boxes)\nSpearman rho={rho_box:.3f}, Kendall tau={tau_box:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].set_xlabel("2D Pixel Area Ratio (box_area / image_area)", fontsize=9.5)
    axes[1].set_ylabel("IPM Estimated Ground Area (m^2, assumed)", fontsize=9.5)

    # Panel C: Lane Width Sensitivity Analysis (3.0 / 3.5 / 4.0 m)
    lws = [str(lw) for lw in lane_widths]
    mean_areas = [sens_summary[k]["mean_estimated_area_m2"] for k in lws]
    mean_scores = [sens_summary[k]["mean_priority_score"] for k in lws]
    x_pos = np.arange(len(lws))
    ax_c = axes[2]
    bars = ax_c.bar(x_pos - 0.18, mean_areas, width=0.35, color="#2b5c8f", edgecolor="black", label="Mean Est. Area (m^2)")
    ax_c2 = ax_c.twinx()
    bars2 = ax_c2.bar(x_pos + 0.18, mean_scores, width=0.35, color="#e07b39", edgecolor="black", label="Mean Priority Score")

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([f"{k} m" for k in lws], fontsize=9.5, fontweight="bold")
    ax_c.set_xlabel("Assumed Lane Width (m)", fontsize=9.5)
    ax_c.set_ylabel("Mean Estimated Area (m^2)", fontsize=9.5, color="#2b5c8f")
    ax_c2.set_ylabel("Mean Priority Score (v2)", fontsize=9.5, color="#e07b39")
    ax_c.set_title(
        f"C. Assumption Sensitivity & Triage Stability\nPriority Level Changes (v1 vs v2): {n_level_changed}/{len(df_img)} ({pct_level_changed:.1f}%)",
        fontsize=10.5,
        fontweight="bold",
    )
    for b in bars:
        val = b.get_height()
        ax_c.annotate(f"{val:.2f}", xy=(b.get_x() + b.get_width() / 2, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for b in bars2:
        val = b.get_height()
        ax_c2.annotate(f"{val:.2f}", xy=(b.get_x() + b.get_width() / 2, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle(
        "C1: Uncalibrated Road-Plane IPM Area Proxy vs 2D Bounding Box Ratio (Test Split Subset)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "11_ipm_birdseye_and_area_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare 2D pixel area ratio vs IPM estimated ground area.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths, data manifest, and weights without executing full inference.")
    parser.add_argument("--n-images", type=int, default=40, help="Number of stratified test images (>= 30).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/compare_yolo11s_best.pt")
    args = parser.parse_args()

    if args.n_images < 30:
        raise ValueError("--n-images must be >= 30 per C1 specification.")

    test_csv = ROOT / "data/manifests/test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Missing manifest: {test_csv}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing weights: {args.weights}")

    if args.dry_run:
        df_test = pd.read_csv(test_csv)
        sampled = select_stratified_test_images(df_test, n_images=args.n_images, seed=args.seed)
        plan = {
            "script": "compare_area_proxy_v2",
            "dry_run": True,
            "test_csv_rows": int(len(df_test)),
            "sampled_images": int(len(sampled)),
            "countries": sampled["country"].value_counts().to_dict(),
            "weights": str(args.weights.relative_to(ROOT)),
            "output_dir": "results/v2/ipm",
        }
        print(json.dumps(plan, indent=2))
        return 0

    summary = run_analysis(n_images=args.n_images, seed=args.seed, weights_path=args.weights)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

