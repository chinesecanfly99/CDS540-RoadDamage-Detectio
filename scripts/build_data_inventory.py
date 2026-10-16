import os
import csv
import time
from PIL import Image

def build_data_inventory(base_data_dir="DGS2022数据", output_csv="data/manifests/data_inventory.csv"):
    """
    遍历 RDD2022 数据集，提取所有图片及标注元数据，生成数据清单 CSV。
    字段包含：
      relative_path: 相对项目根目录的图片路径（使用正斜杠）
      country: 国家名称 (Czech, India, United_States)
      split_if_known: 划分 (train / test)
      has_xml: 是否存在对应的 XML 标注 (True / False)
      xml_relative_path: 相对项目根目录的 XML 标注路径（若无则为空）
      image_width: 图片宽度（像素）
      image_height: 图片高度（像素）
      file_size_kb: 图片文件大小（KB）
    """
    start_time = time.time()
    project_root = os.getcwd()
    abs_data_dir = os.path.join(project_root, base_data_dir)
    
    if not os.path.exists(abs_data_dir):
        raise FileNotFoundError(f"数据根目录未找到: {abs_data_dir}")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    countries = ["Czech", "India", "United_States"]
    valid_img_exts = {".jpg", ".jpeg", ".png"}
    records = []
    
    print(f"[*] 开始扫描数据集: {base_data_dir}")
    print(f"[*] 目标输出文件: {output_csv}\n")
    
    for country in countries:
        country_dir = os.path.join(abs_data_dir, country)
        if not os.path.exists(country_dir):
            print(f"[!] 警告: 未找到国家目录 {country_dir}，跳过")
            continue
            
        for split in ["train", "test"]:
            # 处理嵌套目录结构: <country>/<country>/<split>/images 或 <country>/<split>/images
            img_dir = os.path.join(country_dir, country, split, "images")
            if not os.path.exists(img_dir):
                img_dir = os.path.join(country_dir, split, "images")
                
            xml_dir = os.path.join(country_dir, country, split, "annotations", "xmls")
            if not os.path.exists(xml_dir):
                xml_dir = os.path.join(country_dir, split, "annotations", "xmls")
                
            if not os.path.exists(img_dir):
                print(f"[!] 警告: 未找到图片目录: {img_dir}")
                continue
                
            img_files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_img_exts])
            print(f"  -> 正在处理 {country} [{split}]: 共发现 {len(img_files)} 张图片...")
            
            for img_name in img_files:
                img_path = os.path.join(img_dir, img_name)
                rel_img_path = os.path.relpath(img_path, project_root).replace("\\", "/")
                
                # 检查对应 XML
                base_name = os.path.splitext(img_name)[0]
                xml_name = base_name + ".xml"
                xml_path = os.path.join(xml_dir, xml_name)
                
                has_xml = os.path.exists(xml_path)
                rel_xml_path = os.path.relpath(xml_path, project_root).replace("\\", "/") if has_xml else ""
                
                # 快速读取图片尺寸（仅解析 Header，不加载全图到内存）
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"[!] 读取图片失败: {rel_img_path}, 错误: {e}")
                    width, height = -1, -1
                    
                file_size_kb = round(os.path.getsize(img_path) / 1024, 2)
                
                records.append({
                    "relative_path": rel_img_path,
                    "country": country,
                    "split_if_known": split,
                    "has_xml": has_xml,
                    "xml_relative_path": rel_xml_path,
                    "image_width": width,
                    "image_height": height,
                    "file_size_kb": file_size_kb
                })
                
    # 写入 CSV 文件
    headers = [
        "relative_path", 
        "country", 
        "split_if_known", 
        "has_xml", 
        "xml_relative_path", 
        "image_width", 
        "image_height", 
        "file_size_kb"
    ]
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
        
    elapsed = time.time() - start_time
    print(f"\n[√] 成功生成清单! 耗时: {elapsed:.2f} 秒")
    print(f"[√] 清单已保存至: {output_csv}")
    print(f"[√] 总图片条目数: {len(records)}")
    
    # 输出汇总统计
    summary = {}
    for r in records:
        key = (r["country"], r["split_if_known"])
        if key not in summary:
            summary[key] = {"total": 0, "has_xml": 0, "resolutions": set()}
        summary[key]["total"] += 1
        if r["has_xml"]:
            summary[key]["has_xml"] += 1
        summary[key]["resolutions"].add((r["image_width"], r["image_height"]))
        
    print("\n" + "="*70)
    print(f"{'Country':<15} | {'Split':<6} | {'Images':<8} | {'XMLs':<8} | {'Sample Resolution'}")
    print("="*70)
    for (country, split), data in sorted(summary.items()):
        res_sample = list(data["resolutions"])[:2]
        res_str = ", ".join([f"{w}x{h}" for w, h in res_sample])
        print(f"{country:<15} | {split:<6} | {data['total']:<8} | {data['has_xml']:<8} | {res_str}")
    print("="*70)

if __name__ == "__main__":
    build_data_inventory()
