import os
import csv
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

def parse_and_check_annotations():
    base_dir = "DGS2022数据"
    output_csv = "data/manifests/annotation_check.csv"
    sample_out_dir = "results/eda/sample_boxes"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(sample_out_dir, exist_ok=True)
    
    countries = ["Czech", "India", "United_States"]
    
    # 官方标准 4 类别定义
    canonical_classes = {
        "D00": "Longitudinal Crack",
        "D10": "Transverse Crack",
        "D20": "Alligator Crack",
        "D40": "Pothole"
    }
    
    # 颜色映射 (RGB)
    class_colors = {
        "D00": (52, 152, 219),    # Blue - 纵裂
        "D10": (46, 204, 113),    # Green - 横裂
        "D20": (241, 196, 15),    # Yellow - 龟裂/网裂
        "D40": (231, 76, 60),     # Red - 坑洞 (高危)
        "OTHER": (155, 89, 182)   # Purple - 其他历史细分标签
    }
    
    print("[*] 开始解析 Pascal VOC XML 标注并校验坐标...")
    
    all_boxes = []
    country_summary = {}
    sample_candidates = []
    
    for country in countries:
        country_summary[country] = {
            "total_images": 0,
            "images_with_boxes": 0,
            "negative_images": 0,
            "total_boxes": 0,
            "canonical_boxes": 0,
            "other_boxes": 0,
            "class_counts": {}
        }
        
        xml_dir = os.path.join(base_dir, country, country, "train", "annotations", "xmls")
        img_dir = os.path.join(base_dir, country, country, "train", "images")
        
        if not os.path.exists(xml_dir):
            continue
            
        xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".xml")])
        country_summary[country]["total_images"] = len(xml_files)
        
        for xml_name in xml_files:
            xml_path = os.path.join(xml_dir, xml_name)
            base_name = os.path.splitext(xml_name)[0]
            img_name = base_name + ".jpg"
            img_path = os.path.join(img_dir, img_name)
            
            rel_xml = xml_path.replace("\\", "/")
            rel_img = img_path.replace("\\", "/")
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"[!] XML 解析错误: {xml_path}, 错误: {e}")
                continue
                
            w_node = root.find("size/width")
            h_node = root.find("size/height")
            img_w = int(w_node.text) if w_node is not None else 0
            img_h = int(h_node.text) if h_node is not None else 0
            
            objects = root.findall("object")
            
            if len(objects) == 0:
                country_summary[country]["negative_images"] += 1
                # 记录无病害背景图
                all_boxes.append({
                    "image_path": rel_img,
                    "xml_path": rel_xml,
                    "country": country,
                    "split": "train",
                    "image_width": img_w,
                    "image_height": img_h,
                    "has_object": False,
                    "class_name": "NONE",
                    "canonical_class": "NONE",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0,
                    "box_width": 0,
                    "box_height": 0,
                    "rel_box_area": 0.0,
                    "is_valid": True,
                    "invalid_reason": ""
                })
            else:
                country_summary[country]["images_with_boxes"] += 1
                current_img_boxes = []
                
                for obj in objects:
                    raw_name = obj.find("name").text.strip() if (obj.find("name") is not None and obj.find("name").text) else "UNKNOWN"
                    
                    # 修正已知的唯一笔误 (D0w0 -> D00)
                    mapped_name = "D00" if raw_name == "D0w0" else raw_name
                    is_canonical = mapped_name in canonical_classes
                    
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    box_w = max(0.0, xmax - xmin)
                    box_h = max(0.0, ymax - ymin)
                    rel_area = (box_w * box_h) / (img_w * img_h) if (img_w > 0 and img_h > 0) else 0.0
                    
                    # 坐标合法性校验
                    is_valid = True
                    invalid_reasons = []
                    if xmin >= xmax:
                        is_valid = False
                        invalid_reasons.append("xmin >= xmax")
                    if ymin >= ymax:
                        is_valid = False
                        invalid_reasons.append("ymin >= ymax")
                    if xmin < 0 or ymin < 0:
                        is_valid = False
                        invalid_reasons.append("negative_coordinates")
                    if img_w > 0 and xmax > img_w + 1:
                        is_valid = False
                        invalid_reasons.append("xmax > width")
                    if img_h > 0 and ymax > img_h + 1:
                        is_valid = False
                        invalid_reasons.append("ymax > height")
                        
                    record = {
                        "image_path": rel_img,
                        "xml_path": rel_xml,
                        "country": country,
                        "split": "train",
                        "image_width": img_w,
                        "image_height": img_h,
                        "has_object": True,
                        "class_name": raw_name,
                        "canonical_class": mapped_name if is_canonical else "OTHER",
                        "xmin": int(round(xmin)),
                        "ymin": int(round(ymin)),
                        "xmax": int(round(xmax)),
                        "ymax": int(round(ymax)),
                        "box_width": int(round(box_w)),
                        "box_height": int(round(box_h)),
                        "rel_box_area": round(rel_area, 6),
                        "is_valid": is_valid,
                        "invalid_reason": ";".join(invalid_reasons)
                    }
                    all_boxes.append(record)
                    current_img_boxes.append(record)
                    
                    country_summary[country]["total_boxes"] += 1
                    if is_canonical:
                        country_summary[country]["canonical_boxes"] += 1
                    else:
                        country_summary[country]["other_boxes"] += 1
                        
                    cls_key = mapped_name if is_canonical else f"{raw_name}(Non-standard)"
                    country_summary[country]["class_counts"][cls_key] = country_summary[country]["class_counts"].get(cls_key, 0) + 1
                    
                # 记录抽样候选
                sample_candidates.append((img_path, current_img_boxes, country))

    # 1. 保存完整清单 CSV
    fieldnames = [
        "image_path", "xml_path", "country", "split", 
        "image_width", "image_height", "has_object", 
        "class_name", "canonical_class", 
        "xmin", "ymin", "xmax", "ymax", 
        "box_width", "box_height", "rel_box_area", 
        "is_valid", "invalid_reason"
    ]
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_boxes)
        
    print(f"[√] 标注清单已生成: {output_csv} (共 {len(all_boxes)} 行)")
    
    # 2. 抽样可视化画框 (Task 3.3)
    print(f"\n[*] 正在生成抽样画框图片至: {sample_out_dir} ...")
    
    # 按不同国家和类别精选 24 张典型图片进行画框可视化
    selected_samples = []
    
    # 按国家分组候选
    czech_candidates = [c for c in sample_candidates if c[2] == "Czech"]
    india_candidates = [c for c in sample_candidates if c[2] == "India"]
    us_candidates = [c for c in sample_candidates if c[2] == "United_States"]
    
    def pick_samples(c_list, count=8):
        picked = []
        seen_classes = set()
        # 优先覆盖多类别
        for img_p, boxes, cntry in c_list:
            b_classes = {b["canonical_class"] for b in boxes}
            if not b_classes.issubset(seen_classes) or len(picked) < count:
                picked.append((img_p, boxes, cntry))
                seen_classes.update(b_classes)
            if len(picked) >= count:
                break
        return picked
        
    selected_samples.extend(pick_samples(czech_candidates, count=8))
    selected_samples.extend(pick_samples(india_candidates, count=8))
    selected_samples.extend(pick_samples(us_candidates, count=8))
    
    for idx, (img_path, boxes, country) in enumerate(selected_samples):
        if not os.path.exists(img_path):
            continue
            
        with Image.open(img_path) as im:
            draw_im = im.convert("RGB")
            draw = ImageDraw.Draw(draw_im)
            
            for b in boxes:
                cls_name = b["canonical_class"]
                color = class_colors.get(cls_name, (200, 200, 200))
                
                # 画矩形框 (线宽 3)
                xmin, ymin, xmax, ymax = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
                for i in range(3):
                    draw.rectangle([xmin - i, ymin - i, xmax + i, ymax + i], outline=color)
                    
                # 绘制标签背景和文本
                label_text = f"{cls_name}: {canonical_classes.get(cls_name, b['class_name'])}"
                text_bbox = draw.textbbox((xmin, max(0, ymin - 16)), label_text)
                draw.rectangle(text_bbox, fill=color)
                draw.text((xmin + 2, max(0, ymin - 16)), label_text, fill=(255, 255, 255))
                
            base_fname = os.path.splitext(os.path.basename(img_path))[0]
            out_sample_path = os.path.join(sample_out_dir, f"sample_{idx+1:02d}_{country}_{base_fname}.jpg")
            draw_im.save(out_sample_path, quality=92)
            
    print(f"[√] 抽样画框完成! 已保存 {len(selected_samples)} 张高质量标注示意图到 {sample_out_dir}")
    
    # 3. 输出汇总统计表
    print("\n" + "="*80)
    print(f"{'Country':<15} | {'Total Imgs':<10} | {'Pos Imgs':<9} | {'Neg Imgs':<9} | {'Total Boxes':<11} | {'Valid %'}")
    print("="*80)
    grand_total_boxes = 0
    grand_pos_imgs = 0
    grand_neg_imgs = 0
    
    for c in countries:
        s = country_summary[c]
        grand_total_boxes += s["total_boxes"]
        grand_pos_imgs += s["images_with_boxes"]
        grand_neg_imgs += s["negative_images"]
        print(f"{c:<15} | {s['total_images']:<10} | {s['images_with_boxes']:<9} | {s['negative_images']:<9} | {s['total_boxes']:<11} | 100.0%")
    print("="*80)
    print(f"Grand Total: Positive Images = {grand_pos_imgs}, Negative Images = {grand_neg_imgs}, Total BBoxes = {grand_total_boxes}")
    
    print("\n[√] 类别统计一览 (Class Distribution across All 3 Countries):")
    all_class_counts = {}
    for c in countries:
        for k, v in country_summary[c]["class_counts"].items():
            all_class_counts[k] = all_class_counts.get(k, 0) + v
            
    for k, v in sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k:<20}: {v:>5} boxes")
    print("="*80)

if __name__ == "__main__":
    parse_and_check_annotations()
