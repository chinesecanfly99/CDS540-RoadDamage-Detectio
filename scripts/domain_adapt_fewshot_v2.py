"""
Labelled US Subset Few-Shot Domain Adaptation with Strict Holdout Integrity (v2).
Course: CDS540 Computer Vision (C-Tier Upgrade C3)

Strictly enforces R4 / DECISIONS.md D08 & D09:
  - The 4,805 US frames are already labelled and previously viewed; we NEVER claim
    "unseen US", "blind test", or "never-seen target".
  - Deterministically partitions 4,805 US images (seed=42) into:
      * fewshot_val: 200 images (stratified, for early stopping / LR only)
      * fewshot_train: nested stratified subsets of 50 / 100 / 200 images (with D40)
      * adapt_holdout: remaining 4,405 images (>= 4,400), strictly isolated from training/tuning
  - Re-measures `compare_yolo11s_best.pt` on `adapt_holdout` (never copying old 4,805 CSV numbers).
  - Also evaluates all 3 adapted weights on in-domain `test` (1,579 images) to measure
    catastrophic forgetting.

Outputs:
  - data/manifests/v2/fewshot_train_{50,100,200}.csv, fewshot_val.csv, adapt_holdout.csv
  - configs/v2/data_fewshot*.yaml, data_adapt_holdout.yaml, data_in_domain_test.yaml
  - models/v2/yolo11s_fewshot_{50,100,200}.pt
  - results/v2/domain_adapt/split_integrity.json
  - results/v2/domain_adapt/domain_adapt.csv
  - results/v2/domain_adapt/13_fewshot_domain_recovery.png (300 DPI, English-only)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def stratified_sample_indices(
    df_pool: pd.DataFrame, n_samples: int, rng: np.random.Generator, min_d40: int = 8
) -> list[int]:
    """
    Stratified sampling by `canonical_classes` while guaranteeing D40 representation.
    Returns index labels from `df_pool`.
    """
    if n_samples > len(df_pool):
        raise ValueError(f"Requested {n_samples} samples from pool of {len(df_pool)}.")

    d40_mask = df_pool["canonical_classes"].astype(str).str.contains("D40", na=False)
    d40_indices = df_pool[d40_mask].index.tolist()
    non_d40_df = df_pool[~d40_mask]

    take_d40 = min(min_d40, len(d40_indices), n_samples)
    chosen_d40 = rng.choice(d40_indices, size=take_d40, replace=False).tolist() if take_d40 > 0 else []

    rem_need = n_samples - len(chosen_d40)
    if rem_need <= 0:
        return chosen_d40

    # Proportional allocation across canonical_classes groups in non-D40 pool
    counts = non_d40_df["canonical_classes"].fillna("NONE").value_counts()
    raw_alloc = (counts / counts.sum()) * rem_need
    alloc = np.floor(raw_alloc).astype(int)
    shortfall = rem_need - int(alloc.sum())
    if shortfall > 0:
        remainders = (raw_alloc - alloc).sort_values(ascending=False)
        for grp in remainders.index[:shortfall]:
            alloc[grp] += 1

    chosen_non_d40: list[int] = []
    for grp, k in alloc.items():
        if k <= 0:
            continue
        grp_pool = non_d40_df[non_d40_df["canonical_classes"].fillna("NONE") == grp].index.tolist()
        k_actual = min(int(k), len(grp_pool))
        if k_actual > 0:
            chosen_non_d40.extend(rng.choice(grp_pool, size=k_actual, replace=False).tolist())

    # Top up if rounding caused minor deficit
    if len(chosen_non_d40) < rem_need:
        used_set = set(chosen_d40) | set(chosen_non_d40)
        leftover = [idx for idx in df_pool.index.tolist() if idx not in used_set]
        need = rem_need - len(chosen_non_d40)
        chosen_non_d40.extend(rng.choice(leftover, size=need, replace=False).tolist())

    return chosen_d40 + chosen_non_d40


def build_us_splits(seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """
    Partition 4,805 US images into:
      - fewshot_val (200)
      - fewshot_train_200 (200, with nested 50 and 100 subsets)
      - adapt_holdout (4,405)
    """
    df_us = pd.read_csv(ROOT / "data/manifests/domain_test.csv")
    rng = np.random.default_rng(seed)

    # 1. Carve out fixed 200-image validation split first
    val_idx = stratified_sample_indices(df_us, n_samples=200, rng=rng, min_d40=8)
    df_val = df_us.loc[val_idx].reset_index(drop=True)

    df_rem = df_us.drop(index=val_idx)

    # 2. Carve out nested 50 -> 100 -> 200 fewshot train subsets from remaining pool
    idx_50 = stratified_sample_indices(df_rem, n_samples=50, rng=rng, min_d40=5)
    df_rem_after_50 = df_rem.drop(index=idx_50)
    idx_add_50 = stratified_sample_indices(df_rem_after_50, n_samples=50, rng=rng, min_d40=4)
    idx_100 = idx_50 + idx_add_50

    df_rem_after_100 = df_rem.drop(index=idx_100)
    idx_add_100 = stratified_sample_indices(df_rem_after_100, n_samples=100, rng=rng, min_d40=6)
    idx_200 = idx_100 + idx_add_100

    df_train_50 = df_us.loc[idx_50].reset_index(drop=True)
    df_train_100 = df_us.loc[idx_100].reset_index(drop=True)
    df_train_200 = df_us.loc[idx_200].reset_index(drop=True)
    df_holdout = df_us.drop(index=val_idx + idx_200).reset_index(drop=True)

    # Verify strict zero-overlap integrity
    s_val = set(df_val["image_path"].tolist())
    s_t50 = set(df_train_50["image_path"].tolist())
    s_t100 = set(df_train_100["image_path"].tolist())
    s_t200 = set(df_train_200["image_path"].tolist())
    s_hold = set(df_holdout["image_path"].tolist())

    inter_train_val = len(s_t200 & s_val)
    inter_train_hold = len(s_t200 & s_hold)
    inter_val_hold = len(s_val & s_hold)
    total_unique = len(s_t200 | s_val | s_hold)

    if inter_train_val != 0 or inter_train_hold != 0 or inter_val_hold != 0 or total_unique != len(df_us):
        raise RuntimeError(
            f"Split integrity check failed! intersections=({inter_train_val},{inter_train_hold},{inter_val_hold}), total={total_unique}"
        )

    integrity = {
        "seed": seed,
        "source_manifest": "data/manifests/domain_test.csv",
        "source_total": int(len(df_us)),
        "counts": {
            "fewshot_train_50": int(len(df_train_50)),
            "fewshot_train_100": int(len(df_train_100)),
            "fewshot_train_200": int(len(df_train_200)),
            "fewshot_val": int(len(df_val)),
            "adapt_holdout": int(len(df_holdout)),
        },
        "d40_image_counts": {
            "fewshot_train_50": int(df_train_50["canonical_classes"].astype(str).str.contains("D40").sum()),
            "fewshot_train_100": int(df_train_100["canonical_classes"].astype(str).str.contains("D40").sum()),
            "fewshot_train_200": int(df_train_200["canonical_classes"].astype(str).str.contains("D40").sum()),
            "fewshot_val": int(df_val["canonical_classes"].astype(str).str.contains("D40").sum()),
            "adapt_holdout": int(df_holdout["canonical_classes"].astype(str).str.contains("D40").sum()),
        },
        "pairwise_path_intersections": {
            "train50_vs_val": len(s_t50 & s_val),
            "train50_vs_adapt_holdout": len(s_t50 & s_hold),
            "train100_vs_val": len(s_t100 & s_val),
            "train100_vs_adapt_holdout": len(s_t100 & s_hold),
            "train200_vs_val": inter_train_val,
            "train200_vs_adapt_holdout": inter_train_hold,
            "val_vs_adapt_holdout": inter_val_hold,
        },
        "nested_subset_verified": bool(s_t50.issubset(s_t100) and s_t100.issubset(s_t200)),
        "disjoint_sum_equals_source": bool(len(df_train_200) + len(df_val) + len(df_holdout) == len(df_us)),
    }

    splits = {
        "fewshot_train_50": df_train_50,
        "fewshot_train_100": df_train_100,
        "fewshot_train_200": df_train_200,
        "fewshot_val": df_val,
        "adapt_holdout": df_holdout,
    }
    return splits, integrity


def link_yolo_split(df_split: pd.DataFrame, split_name: str) -> None:
    """Create NTFS hardlinks under data/yolo/{images,labels}/<split_name> without copying bytes."""
    src_img_dir = ROOT / "data/yolo/images/domain_test"
    src_lbl_dir = ROOT / "data/yolo/labels/domain_test"
    dst_img_dir = ROOT / "data/yolo/images" / split_name
    dst_lbl_dir = ROOT / "data/yolo/labels" / split_name

    for d in (dst_img_dir, dst_lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for rel_img in df_split["image_path"]:
        stem = Path(str(rel_img)).stem
        ext = Path(str(rel_img)).suffix
        s_img = src_img_dir / f"{stem}{ext}"
        s_lbl = src_lbl_dir / f"{stem}.txt"
        if not s_img.is_file():
            raise FileNotFoundError(f"Missing YOLO domain_test image: {s_img}")
        if not s_lbl.is_file():
            raise FileNotFoundError(f"Missing YOLO domain_test label: {s_lbl}")
        os.link(s_img, dst_img_dir / s_img.name)
        os.link(s_lbl, dst_lbl_dir / s_lbl.name)


def write_v2_yaml(path: Path, train_split: str, val_split: str, test_split: str | None = None) -> None:
    cfg: dict[str, Any] = {
        "path": "data/yolo",
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "names": CLASS_NAMES,
    }
    if test_split is not None:
        cfg["test"] = f"images/{test_split}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def extract_eval_metrics(metrics: Any) -> dict[str, float]:
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    class_map50 = metrics.box.ap50.tolist() if hasattr(metrics.box, "ap50") else []
    class_r = metrics.box.r.tolist() if hasattr(metrics.box, "r") else []
    return {
        "Precision": round(mp, 4),
        "Recall": round(mr, 4),
        "mAP@50": round(map50, 4),
        "mAP@50-95": round(map50_95, 4),
        "D00_AP50": round(float(class_map50[0]) if len(class_map50) > 0 else 0.0, 4),
        "D10_AP50": round(float(class_map50[1]) if len(class_map50) > 1 else 0.0, 4),
        "D20_AP50": round(float(class_map50[2]) if len(class_map50) > 2 else 0.0, 4),
        "D40_AP50": round(float(class_map50[3]) if len(class_map50) > 3 else 0.0, 4),
        "D40_Recall": round(float(class_r[3]) if len(class_r) > 3 else 0.0, 4),
    }


def eval_weights(weights_path: Path, data_yaml: Path, exp_name: str, batch: int = 16) -> dict[str, float]:
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        batch=batch,
        imgsz=640,
        device=0,
        workers=0,
        project=str(ROOT / "runs/v2/eval"),
        name=exp_name,
        exist_ok=True,
        verbose=False,
    )
    return extract_eval_metrics(metrics)


def run_fewshot_adaptation(
    base_weights: Path = ROOT / "models/compare_yolo11s_best.pt",
    seed: int = 42,
    epochs: int = 25,
    lr0: float = 1e-4,
    batch: int = 8,
) -> dict[str, Any]:
    os.chdir(ROOT)
    if epochs > 30:
        raise ValueError("C3 requires epochs <= 30.")
    if lr0 > 1e-4:
        raise ValueError("C3 requires lr0 <= 1e-4.")

    manifests_dir = ROOT / "data/manifests/v2"
    configs_dir = ROOT / "configs/v2"
    models_v2_dir = ROOT / "models/v2"
    out_dir = ROOT / "results/v2/domain_adapt"
    runs_v2_dir = ROOT / "runs/v2"
    for d in (manifests_dir, configs_dir, models_v2_dir, out_dir, runs_v2_dir):
        d.mkdir(parents=True, exist_ok=True)

    splits, integrity = build_us_splits(seed=seed)
    for name, df_sp in splits.items():
        df_sp.to_csv(manifests_dir / f"{name}.csv", index=False)

    (out_dir / "split_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    # Build NTFS hardlinks for YOLO splits
    split_folder_map = {
        "fewshot_train_50": "v2_fewshot_50",
        "fewshot_train_100": "v2_fewshot_100",
        "fewshot_train_200": "v2_fewshot_200",
        "fewshot_val": "v2_fewshot_val",
        "adapt_holdout": "v2_adapt_holdout",
    }
    for sp_key, folder_name in split_folder_map.items():
        link_yolo_split(splits[sp_key], folder_name)

    # Write YAML configs (including required canonical names data_fewshot.yaml & data_adapt_holdout.yaml)
    write_v2_yaml(configs_dir / "data_fewshot_50.yaml", "v2_fewshot_50", "v2_fewshot_val")
    write_v2_yaml(configs_dir / "data_fewshot_100.yaml", "v2_fewshot_100", "v2_fewshot_val")
    write_v2_yaml(configs_dir / "data_fewshot_200.yaml", "v2_fewshot_200", "v2_fewshot_val")
    write_v2_yaml(configs_dir / "data_fewshot.yaml", "v2_fewshot_200", "v2_fewshot_val", "v2_adapt_holdout")
    write_v2_yaml(configs_dir / "data_adapt_holdout.yaml", "v2_fewshot_val", "v2_adapt_holdout")
    write_v2_yaml(configs_dir / "data_in_domain_test.yaml", "train", "test")

    records: list[dict[str, Any]] = []

    # 1. Re-measure zero-shot baseline on adapt_holdout (4,405 images) & in-domain test (1,579 images)
    holdout_n = len(splits["adapt_holdout"])
    base_holdout_m = eval_weights(base_weights, configs_dir / "data_adapt_holdout.yaml", "holdout_baseline")
    records.append(
        {
            "weights_version": base_weights.name,
            "eval_split": f"adapt_holdout ({holdout_n})",
            "n_images": holdout_n,
            **base_holdout_m,
            "notes": "re-measured on holdout (zero-shot baseline state; never copied from 4805 CSV)",
        }
    )

    base_indom_m = eval_weights(base_weights, configs_dir / "data_in_domain_test.yaml", "indomain_baseline")
    records.append(
        {
            "weights_version": base_weights.name,
            "eval_split": "in_domain_test (1579)",
            "n_images": 1579,
            **base_indom_m,
            "notes": "re-measured in-domain reference baseline before adaptation",
        }
    )

    # 2. Fine-tune 50 / 100 / 200 independently from base_weights
    shots_list = [50, 100, 200]
    for shot in shots_list:
        yaml_path = configs_dir / f"data_fewshot_{shot}.yaml"
        exp_name = f"fewshot_{shot}"
        model = YOLO(str(base_weights))
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            lr0=lr0,
            lrf=0.1,
            optimizer="AdamW",
            warmup_epochs=2,
            patience=10,
            seed=seed,
            deterministic=True,
            device=0,
            workers=0,
            project=str(runs_v2_dir),
            name=exp_name,
            exist_ok=True,
            verbose=True,
        )
        run_weights_dir = runs_v2_dir / exp_name / "weights"
        best_pt = run_weights_dir / "best.pt"
        src_pt = best_pt if best_pt.is_file() else (run_weights_dir / "last.pt")
        out_pt = models_v2_dir / f"yolo11s_fewshot_{shot}.pt"
        shutil.copy2(src_pt, out_pt)

        # Evaluate on adapt_holdout (4,405)
        hold_m = eval_weights(out_pt, configs_dir / "data_adapt_holdout.yaml", f"holdout_fewshot_{shot}")
        records.append(
            {
                "weights_version": out_pt.name,
                "eval_split": f"adapt_holdout ({holdout_n})",
                "n_images": holdout_n,
                **hold_m,
                "notes": f"few-shot adaptation on a labelled US subset (n={shot}, epochs={epochs}, lr0={lr0})",
            }
        )

        # Evaluate on in-domain test (1,579) for catastrophic forgetting check
        indom_m = eval_weights(out_pt, configs_dir / "data_in_domain_test.yaml", f"indomain_fewshot_{shot}")
        delta_indom = round(indom_m["mAP@50"] - base_indom_m["mAP@50"], 4)
        records.append(
            {
                "weights_version": out_pt.name,
                "eval_split": "in_domain_test (1579)",
                "n_images": 1579,
                **indom_m,
                "notes": f"catastrophic forgetting check vs baseline: delta_mAP50={delta_indom:+.4f}",
            }
        )

    df_res = pd.DataFrame(records)
    df_res.to_csv(out_dir / "domain_adapt.csv", index=False)

    # Render Figure 13 (300 DPI, English-only)
    df_hold = df_res[df_res["eval_split"].str.startswith("adapt_holdout")].copy()
    df_indom = df_res[df_res["eval_split"].str.startswith("in_domain_test")].copy()
    x_labels = ["0 (Baseline)", "50", "100", "200"]
    x_pos = np.arange(len(x_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=300)

    # Panel A: Held-out Labelled US Subset Recovery (mAP@50, Precision, Recall, D40 Recall)
    ax1.plot(x_pos, df_hold["mAP@50"] * 100.0, marker="o", linewidth=2.3, color="#2b5c8f", label="US Holdout mAP@50 (%)")
    ax1.plot(x_pos, df_hold["Recall"] * 100.0, marker="s", linewidth=2.0, color="#1b9e77", label="US Holdout Recall (%)")
    ax1.plot(x_pos, df_hold["D40_Recall"] * 100.0, marker="^", linewidth=2.0, linestyle="--", color="#d95f02", label="US Holdout D40 Recall (%)")
    for idx_pt, val in enumerate(df_hold["mAP@50"] * 100.0):
        ax1.annotate(f"{val:.2f}%", xy=(x_pos[idx_pt], val), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#2b5c8f")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Labelled US Few-Shot Training Subset Size (Images)", fontsize=9.5)
    ax1.set_ylabel("Performance (%)", fontsize=9.5, fontweight="bold")
    ax1.set_title(f"A. Held-Out Labelled US Subset (n={holdout_n})\nFew-Shot Domain Adaptation Curve", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="best", fontsize=8.5, frameon=True)

    # Panel B: In-Domain Test Retention / Catastrophic Forgetting Check (n=1,579)
    ax2.plot(x_pos, df_indom["mAP@50"] * 100.0, marker="o", linewidth=2.3, color="#7570b3", label="In-Domain Test mAP@50 (%)")
    ax2.plot(x_pos, df_indom["D40_Recall"] * 100.0, marker="^", linewidth=2.0, linestyle="--", color="#e7298a", label="In-Domain D40 Recall (%)")
    for idx_pt, val in enumerate(df_indom["mAP@50"] * 100.0):
        ax2.annotate(f"{val:.2f}%", xy=(x_pos[idx_pt], val), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#7570b3")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Labelled US Few-Shot Training Subset Size (Images)", fontsize=9.5)
    ax2.set_ylabel("In-Domain Performance (%)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. In-Domain Test Set (Czech + India, n=1,579)\nCatastrophic Forgetting Diagnostic", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="best", fontsize=8.5, frameon=True)

    fig.suptitle(
        "C3: Few-Shot Adaptation on a Labelled US Subset vs In-Domain Retention (YOLO11s)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "13_fewshot_domain_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {"integrity": integrity, "records": records}
    (out_dir / "domain_adapt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="US domain few-shot adaptation with strict holdout split.")
    parser.add_argument("--dry-run", action="store_true", help="Validate split counts and zero-overlap integrity without training.")
    parser.add_argument("--base-weights", type=Path, default=ROOT / "models/compare_yolo11s_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    if not args.base_weights.is_file():
        raise FileNotFoundError(f"Missing base weights: {args.base_weights}")
    if not (ROOT / "data/manifests/domain_test.csv").is_file():
        raise FileNotFoundError("Missing data/manifests/domain_test.csv")

    if args.dry_run:
        _, integrity = build_us_splits(seed=args.seed)
        plan = {
            "script": "domain_adapt_fewshot_v2",
            "dry_run": True,
            "base_weights": str(args.base_weights.relative_to(ROOT)),
            "epochs": args.epochs,
            "lr0": args.lr0,
            "batch": args.batch,
            "split_integrity_preview": integrity,
        }
        print(json.dumps(plan, indent=2))
        return 0

    summary = run_fewshot_adaptation(
        base_weights=args.base_weights,
        seed=args.seed,
        epochs=args.epochs,
        lr0=args.lr0,
        batch=args.batch,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

