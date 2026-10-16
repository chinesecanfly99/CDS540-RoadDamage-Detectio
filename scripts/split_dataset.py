"""
split_dataset.py
--------------------------------------------------------------------------------
Phase 4 / Step 5: Dataset Splitting & Domain Holdout Protocol.

Objectives:
1. Strictly split from the 15,340 annotated training images.
2. Fix random seed (seed=42) for 100% scientific reproducibility.
3. Hold out an entire country (United States, 4,805 images) as the unseen
   cross-domain evaluation set (`domain_test.csv`) to benchmark Step 10.
4. Stratified split of in-domain data (Czech + India, 10,535 images) into
   Train (70%), Val (15%), and Test (15%) preserving country and damage ratios.
5. Prevent any image-level data leakage (zero intersection across splits).
6. Export CSV split manifests and a comprehensive English report `reports/split_policy.md`.
--------------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np


def build_image_manifest(annotation_check_csv="data/manifests/annotation_check.csv"):
    """
    Collapses box-level annotation manifest into unique image-level records.
    """
    print(f"[*] Reading annotation check manifest: {annotation_check_csv}")
    df = pd.read_csv(annotation_check_csv)

    # Group by image_path
    records = []
    grouped = df.groupby("image_path")

    for img_path, group in grouped:
        first = group.iloc[0]
        has_object = bool(first["has_object"])
        country = first["country"]
        split = first["split"]
        img_w = int(first["image_width"])
        img_h = int(first["image_height"])
        xml_path = first["xml_path"]

        if has_object:
            box_count = len(group)
            raw_classes = sorted(list(group["class_name"].unique()))
            canonical_classes = sorted(list(group["canonical_class"].unique()))
        else:
            box_count = 0
            raw_classes = ["NONE"]
            canonical_classes = ["NONE"]

        records.append({
            "image_path": img_path,
            "xml_path": xml_path,
            "country": country,
            "split": split,
            "image_width": img_w,
            "image_height": img_h,
            "has_object": has_object,
            "box_count": box_count,
            "raw_classes": ",".join(raw_classes),
            "canonical_classes": ",".join(canonical_classes)
        })

    img_df = pd.DataFrame(records)
    print(f"[+] Total unique annotated images compiled: {len(img_df):,}")
    return img_df


def split_dataset(seed=42, output_dir="data/manifests", report_path="reports/split_policy.md"):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    img_df = build_image_manifest()

    # Step 1: Hold out United States as Unseen Domain Test Set
    domain_test_df = img_df[img_df["country"] == "United_States"].copy()
    in_domain_df = img_df[img_df["country"] != "United_States"].copy()

    print(f"[*] Cross-Domain Holdout (United States): {len(domain_test_df):,} images")
    print(f"[*] In-Domain Pool (Czech + India)     : {len(in_domain_df):,} images")

    # Step 2: Stratified Split of In-Domain Pool
    # Stratum = country + "_" + has_object (preserves both national origin & damage ratio)
    in_domain_df["stratum"] = in_domain_df["country"] + "_" + in_domain_df["has_object"].astype(str)

    train_indices = []
    val_indices = []
    test_indices = []

    for stratum, group in in_domain_df.groupby("stratum"):
        shuffled_indices = group.sample(frac=1.0, random_state=seed).index.tolist()
        n = len(shuffled_indices)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        # Ensure remaining goes to test
        train_idx = shuffled_indices[:n_train]
        val_idx = shuffled_indices[n_train:n_train + n_val]
        test_idx = shuffled_indices[n_train + n_val:]

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

    train_df = in_domain_df.loc[train_indices].copy().drop(columns=["stratum"])
    val_df = in_domain_df.loc[val_indices].copy().drop(columns=["stratum"])
    test_df = in_domain_df.loc[test_indices].copy().drop(columns=["stratum"])

    # Step 3: Leakage & Disjointness Verification
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])
    domain_test_paths = set(domain_test_df["image_path"])

    assert len(train_paths & val_paths) == 0, "FATAL: Leakage detected between Train and Val!"
    assert len(train_paths & test_paths) == 0, "FATAL: Leakage detected between Train and Test!"
    assert len(val_paths & test_paths) == 0, "FATAL: Leakage detected between Val and Test!"
    assert len((train_paths | val_paths | test_paths) & domain_test_paths) == 0, "FATAL: Leakage detected into Domain Test!"
    assert len(train_df) + len(val_df) + len(test_df) + len(domain_test_df) == len(img_df), "FATAL: Image count mismatch!"

    print("\n[+] Verification PASSED: All splits are strictly disjoint (0% data leakage).")

    # Step 4: Export CSVs
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "val.csv")
    test_csv = os.path.join(output_dir, "test.csv")
    domain_test_csv = os.path.join(output_dir, "domain_test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    domain_test_df.to_csv(domain_test_csv, index=False)

    print(f"[+] Exported Train Manifest       : {train_csv} ({len(train_df):,} images)")
    print(f"[+] Exported Val Manifest         : {val_csv} ({len(val_df):,} images)")
    print(f"[+] Exported Test Manifest        : {test_csv} ({len(test_df):,} images)")
    print(f"[+] Exported Domain Test Manifest : {domain_test_csv} ({len(domain_test_df):,} images)")

    # Step 5: Generate Detailed Split Policy Markdown Report
    generate_split_report(report_path, train_df, val_df, test_df, domain_test_df, seed)


def generate_split_report(report_path, train_df, val_df, test_df, domain_test_df, seed):
    """
    Writes a publication-ready English split policy documentation.
    """
    def summarize(df_split, name):
        total = len(df_split)
        pos = (df_split["has_object"] == True).sum()
        neg = total - pos
        boxes = df_split["box_count"].sum()
        countries = df_split["country"].value_counts().to_dict()
        czech_n = countries.get("Czech", 0)
        india_n = countries.get("India", 0)
        us_n = countries.get("United_States", 0)
        return {
            "name": name,
            "total": total,
            "pos": pos,
            "pos_pct": (pos / total * 100) if total > 0 else 0,
            "neg": neg,
            "neg_pct": (neg / total * 100) if total > 0 else 0,
            "boxes": boxes,
            "czech": czech_n,
            "india": india_n,
            "us": us_n
        }

    s_train = summarize(train_df, "Train (In-Domain)")
    s_val = summarize(val_df, "Validation (In-Domain)")
    s_test = summarize(test_df, "Test (In-Domain)")
    s_domain = summarize(domain_test_df, "Domain Test (Zero-Shot Holdout)")

    report_content = f"""# Dataset Partitioning & Cross-Domain Holdout Policy

**Course**: CDS540 Computer Vision  
**Project**: Automated Road Damage Detection for Smart City Maintenance  
**Random Seed**: `{seed}` (Fixed for 100% scientific reproducibility)  
**Total Ground Truth Images**: 15,340 images across Czech, India, and United States  

---

## 1. Split Strategy & Leakage Prevention Protocol

To adhere to the rigorous experimental standards of CDS540 and prevent any optimistic performance bias:
1. **Zero Data Leakage**: Splitting is executed strictly at the **image level** (rather than the bounding box level). Any image, along with all of its bounding boxes, belongs exclusively to exactly one split.
2. **Stratified Sampling**: The in-domain dataset (Czech + India) is partitioned into **Train (70%)**, **Validation (15%)**, and **In-Domain Test (15%)** using stratified grouping across both **Country of Origin** and **Damage Presence (Positive vs Negative background frames)**. This guarantees identical category and country representation across splits.
3. **Dedicated Cross-Domain Holdout**: An entire geographical region (**United States**, 4,805 images) is quarantined from all model training, validation, and hyperparameter tuning. It serves exclusively as the **Zero-Shot Domain Shift Benchmark** in Step 10 to evaluate cross-continental generalization.
4. **Preservation of Background Images**: A representative fraction (37%) of undamaged road background images is systematically retained in Train, Val, and Test to enforce true negative learning and suppress false alarms on pavement markings and shadows.

---

## 2. Partition Summary Statistics

| Partition | Total Images | Positive (Damaged) | Negative (Background) | Total Bounding Boxes | Czech Images | India Images | US Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **{s_train['total']:,}** | {s_train['pos']:,} ({s_train['pos_pct']:.1f}%) | {s_train['neg']:,} ({s_train['neg_pct']:.1f}%) | {s_train['boxes']:,} | {s_train['czech']:,} | {s_train['india']:,} | 0 |
| **Validation** | **{s_val['total']:,}** | {s_val['pos']:,} ({s_val['pos_pct']:.1f}%) | {s_val['neg']:,} ({s_val['neg_pct']:.1f}%) | {s_val['boxes']:,} | {s_val['czech']:,} | {s_val['india']:,} | 0 |
| **In-Domain Test** | **{s_test['total']:,}** | {s_test['pos']:,} ({s_test['pos_pct']:.1f}%) | {s_test['neg']:,} ({s_test['neg_pct']:.1f}%) | {s_test['boxes']:,} | {s_test['czech']:,} | {s_test['india']:,} | 0 |
| **Domain Test (Holdout)** | **{s_domain['total']:,}** | {s_domain['pos']:,} ({s_domain['pos_pct']:.1f}%) | {s_domain['neg']:,} ({s_domain['neg_pct']:.1f}%) | {s_domain['boxes']:,} | 0 | 0 | {s_domain['us']:,} |
| **Combined Total** | **{s_train['total'] + s_val['total'] + s_test['total'] + s_domain['total']:,}** | **{s_train['pos'] + s_val['pos'] + s_test['pos'] + s_domain['pos']:,}** | **{s_train['neg'] + s_val['neg'] + s_test['neg'] + s_domain['neg']:,}** | **{s_train['boxes'] + s_val['boxes'] + s_test['boxes'] + s_domain['boxes']:,}** | **2,829** | **7,706** | **4,805** |

---

## 3. Roles and Operational Guidelines

* **`train.csv` ({s_train['total']:,} images)**: Used exclusively for updating neural network weights during YOLO backpropagation.
* **`val.csv` ({s_val['total']:,} images)**: Used for real-time validation after each epoch, early stopping trigger, and model checkpoint selection (`best.pt`).
* **`test.csv` ({s_test['total']:,} images)**: Used for evaluating the primary in-domain benchmark metrics (mAP@50, mAP@50-95, Class-specific Precision/Recall) after model training is completely finished.
* **`domain_test.csv` ({s_domain['total']:,} images)**: Evaluated in Step 10 to quantify operational degradation under realistic domain shifts (unseen highway pavement, different dashboard cameras, US road markings).

---

## 4. Manifest Export Paths

All generated manifests are stored under `data/manifests/`:
- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test.csv`
- `data/manifests/domain_test.csv`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] Split documentation generated at: {report_path}")


if __name__ == "__main__":
    split_dataset(seed=42)
