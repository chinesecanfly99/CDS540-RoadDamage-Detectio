"""
annotation_qa_inspector.py
--------------------------------------------------------------------------------
Task 4.4 (Bonus / Quality Assurance):
Automated Road Damage Dataset Annotation QA & Outlier Inspector.

Features:
1. Automated outlier detection across 20,962 bounding boxes:
   - Tiny boxes (area < 0.05% or min dimension < 8px)
   - Giant boxes (area > 50% of the entire frame)
   - Extreme aspect ratio boxes (w/h > 12 or h/w > 12)
   - Crowded scenes (> 8 boxes per image)
   - Non-standard / historical categories (D44, D01, D43, D11, D50, typos)
2. Export audit findings to `data/manifests/annotation_anomalies.csv`.
3. Render high-resolution visual inspection cards to `results/eda/qa_inspections/`.
4. CLI support for auditing or inspecting specific images/classes.

Usage:
  python scripts/annotation_qa_inspector.py --mode audit --sample_per_issue 5
  python scripts/annotation_qa_inspector.py --image Czech_000010
--------------------------------------------------------------------------------
"""

import os
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CANONICAL_CLASSES = {"D00", "D10", "D20", "D40"}

BOX_COLORS = {
    "D00": (52, 152, 219),    # Blue: Longitudinal Crack
    "D10": (46, 204, 113),    # Green: Transverse Crack
    "D20": (241, 196, 15),    # Yellow: Alligator Crack
    "D40": (231, 76, 60),     # Red: Pothole
    "OTHER": (155, 89, 182),  # Purple: Non-standard Subtype
    "ANOMALY": (255, 0, 128)  # Bright Magenta: QA Flagged
}


def load_font(size=14):
    font_candidates = [
        "arial.ttf",
        "tahoma.ttf",
        "seguiemj.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def inspect_dataset(csv_path="data/manifests/annotation_check.csv",
                    output_audit_csv="data/manifests/annotation_anomalies.csv",
                    output_img_dir="results/eda/qa_inspections",
                    samples_per_issue=4):
    """
    Scans the annotation manifest for geometric and categorical anomalies.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation manifest not found at {csv_path}")

    print(f"[*] Loading annotations from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter positive bounding box entries
    boxes_df = df[df["has_object"] == True].copy()
    print(f"[*] Total bounding boxes to audit: {len(boxes_df)}")

    # Calculate aspect ratio
    boxes_df["aspect_ratio"] = boxes_df["box_width"] / boxes_df["box_height"].replace(0, 1)
    boxes_df["inv_aspect_ratio"] = boxes_df["box_height"] / boxes_df["box_width"].replace(0, 1)

    # Calculate image box counts
    img_counts = boxes_df["image_path"].value_counts().to_dict()
    boxes_df["box_count_in_image"] = boxes_df["image_path"].map(img_counts)

    # Identify Anomaly Criteria
    # 1. Extreme Aspect Ratio
    boxes_df["issue_extreme_aspect"] = (boxes_df["aspect_ratio"] > 12.0) | (boxes_df["inv_aspect_ratio"] > 12.0)

    # 2. Extremely Tiny Box
    boxes_df["issue_tiny_box"] = (boxes_df["rel_box_area"] < 0.0005) | (boxes_df["box_width"] < 8) | (boxes_df["box_height"] < 8)

    # 3. Giant Box
    boxes_df["issue_giant_box"] = boxes_df["rel_box_area"] > 0.50

    # 4. Non-Standard / Rare Class
    boxes_df["issue_non_standard_class"] = ~boxes_df["class_name"].isin(CANONICAL_CLASSES)

    # 5. Crowded Image
    boxes_df["issue_crowded_image"] = boxes_df["box_count_in_image"] >= 8

    # Combine flags
    issue_cols = [
        "issue_extreme_aspect",
        "issue_tiny_box",
        "issue_giant_box",
        "issue_non_standard_class",
        "issue_crowded_image"
    ]

    def get_issues(row):
        issues = []
        if row["issue_extreme_aspect"]:
            issues.append(f"extreme_aspect_ratio(w/h={row['aspect_ratio']:.1f})")
        if row["issue_tiny_box"]:
            issues.append(f"tiny_box({row['box_width']}x{row['box_height']})")
        if row["issue_giant_box"]:
            issues.append(f"giant_box(area={row['rel_box_area']*100:.1f}%)")
        if row["issue_non_standard_class"]:
            issues.append(f"non_standard_class({row['class_name']})")
        if row["issue_crowded_image"]:
            issues.append(f"crowded_image({row['box_count_in_image']}_boxes)")
        return "; ".join(issues)

    boxes_df["detected_issues"] = boxes_df.apply(get_issues, axis=1)
    anomalies_df = boxes_df[boxes_df["detected_issues"] != ""].copy()

    # Save CSV
    os.makedirs(os.path.dirname(output_audit_csv), exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    anomalies_df.to_csv(output_audit_csv, index=False)
    print(f"[+] Found {len(anomalies_df)} boxes with QA flags ({len(anomalies_df) / len(boxes_df) * 100:.2f}% of all boxes)")
    print(f"[+] Anomaly audit log exported to: {output_audit_csv}")

    # Summary Statistics of Issues
    print("\n--- QA Issue Breakdown ---")
    print(f"  * Extreme Aspect Ratio (>12x)    : {boxes_df['issue_extreme_aspect'].sum():,}")
    print(f"  * Tiny Bounding Boxes (<0.05%)  : {boxes_df['issue_tiny_box'].sum():,}")
    print(f"  * Giant Bounding Boxes (>50%)   : {boxes_df['issue_giant_box'].sum():,}")
    print(f"  * Non-standard / Sub-classes    : {boxes_df['issue_non_standard_class'].sum():,}")
    print(f"  * Boxes in Crowded Images (>=8) : {boxes_df['issue_crowded_image'].sum():,}")
    print("--------------------------\n")

    # Render Sample Images for each issue
    font_bold = load_font(14)
    font_small = load_font(12)

    rendered_samples = 0
    sampled_image_paths = set()

    for col_name, tag in [
        ("issue_giant_box", "giant_box"),
        ("issue_extreme_aspect", "extreme_aspect"),
        ("issue_tiny_box", "tiny_box"),
        ("issue_crowded_image", "crowded_scene"),
        ("issue_non_standard_class", "non_standard_class"),
    ]:
        subset = anomalies_df[anomalies_df[col_name] == True]
        unique_imgs = subset["image_path"].unique()
        chosen_imgs = [img for img in unique_imgs if img not in sampled_image_paths][:samples_per_issue]
        
        for img_rel_path in chosen_imgs:
            sampled_image_paths.add(img_rel_path)
            if not os.path.exists(img_rel_path):
                continue

            img = Image.open(img_rel_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Get all boxes for this image from original boxes_df
            img_boxes = boxes_df[boxes_df["image_path"] == img_rel_path]

            for _, box in img_boxes.iterrows():
                cls = box["canonical_class"] if box["canonical_class"] in BOX_COLORS else "OTHER"
                has_flag = box["detected_issues"] != ""
                color = BOX_COLORS["ANOMALY"] if has_flag else BOX_COLORS[cls]
                width = 3 if has_flag else 2

                draw.rectangle(
                    [(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])],
                    outline=color,
                    width=width
                )

                label_text = f"{box['class_name']}"
                if has_flag:
                    label_text += f" [QA!]"

                # Background banner for text
                t_bbox = draw.textbbox((box["xmin"], max(0, box["ymin"] - 16)), label_text, font=font_small)
                draw.rectangle(t_bbox, fill=color)
                draw.text((box["xmin"] + 2, max(0, box["ymin"] - 16)), label_text, fill=(255, 255, 255), font=font_small)

            # Top Header Banner
            base_fname = os.path.basename(img_rel_path)
            country = img_boxes.iloc[0]["country"]
            banner_text = f"QA Inspection: {base_fname} ({country}) | Type: {tag} | Boxes: {len(img_boxes)}"
            draw.rectangle([(0, 0), (img.width, 24)], fill=(20, 20, 20))
            draw.text((6, 4), banner_text, fill=(241, 196, 15), font=font_bold)

            out_filename = f"qa_{tag}_{base_fname}"
            out_filepath = os.path.join(output_img_dir, out_filename)
            img.save(out_filepath, quality=92)
            rendered_samples += 1

    print(f"[+] Rendered {rendered_samples} QA visualization images into: {output_img_dir}")
    return anomalies_df


def inspect_single_image(image_query, csv_path="data/manifests/annotation_check.csv", output_dir="results/eda/qa_inspections"):
    """
    Renders QA overlay for a specific image filename or query.
    """
    df = pd.read_csv(csv_path)
    matched = df[df["image_path"].str.contains(image_query, case=False, na=False)]
    if len(matched) == 0:
        print(f"[!] No image matching query '{image_query}' found.")
        return

    img_rel_path = matched.iloc[0]["image_path"]
    if not os.path.exists(img_rel_path):
        print(f"[!] Image file does not exist: {img_rel_path}")
        return

    img = Image.open(img_rel_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(14)

    boxes = matched[matched["has_object"] == True]
    for _, box in boxes.iterrows():
        cls = box["canonical_class"] if box["canonical_class"] in BOX_COLORS else "OTHER"
        color = BOX_COLORS[cls]
        draw.rectangle([(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])], outline=color, width=3)
        draw.text((box["xmin"], max(0, box["ymin"] - 16)), f"{box['class_name']}", fill=color, font=font)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"inspect_{os.path.basename(img_rel_path)}")
    img.save(out_path)
    print(f"[+] Saved visual inspection to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDD2022 Annotation QA Inspector")
    parser.add_argument("--mode", type=str, default="audit", choices=["audit", "single"], help="Inspection mode")
    parser.add_argument("--image", type=str, default=None, help="Image filename substring to inspect (single mode)")
    parser.add_argument("--samples", type=int, default=3, help="Sample images to render per issue category")

    args = parser.parse_args()

    if args.image or args.mode == "single":
        query = args.image if args.image else "Czech_000010"
        inspect_single_image(query)
    else:
        inspect_dataset(samples_per_issue=args.samples)
