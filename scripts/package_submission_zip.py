"""
Automated Project Packaging Script
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Creates a self-contained, lightweight sharing archive:
  - Archive Name: CDS540_RoadDamage_Submission_Pack.zip
  - Includes: Source code, final PDF report, PowerPoint PPTX deck, evaluation CSVs,
              300 DPI publication figures, and trained model weights (models/).
  - Excludes: Massive 14GB raw images (DGS2022数据/), 8GB .venv, and intermediate runs/.
  - Total Size: ~100 MB (Easily shareable via WeChat, Email, OneDrive, or Moodle).
"""

import os
import sys
import zipfile
from pathlib import Path


def create_zip_package():
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    
    zip_filename = "CDS540_RoadDamage_Submission_Pack.zip"
    zip_path = root_dir / zip_filename
    
    # Directories and files to include
    include_paths = [
        "submission",
        "demo",
        "reports",
        "results",
        "models",
        "scripts",
        "configs",
        "docs",
        "README.md",
        "requirements.txt",
        "RDD2022_任务执行清单.md",
        "RDD2022_道路损坏检测_项目指南.md",
        ".gitignore"
    ]
    
    # Extensions/folders to exclude
    exclude_patterns = [
        "__pycache__",
        ".pyc",
        ".git",
        ".venv",
        "runs",
        "DGS2022数据",
        "data/yolo/images",
        "data/robustness_test"
    ]
    
    print(f"\n=======================================================")
    print(f"Creating Lightweight Shareable Submission Package")
    print(f"Target: {zip_path}")
    print(f"=======================================================")
    
    total_files = 0
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_paths:
            p = root_dir / item
            if not p.exists():
                continue
                
            if p.is_file():
                zipf.write(p, arcname=item)
                total_files += 1
            elif p.is_dir():
                for root, dirs, files in os.walk(p):
                    # Filter directories in-place
                    dirs[:] = [d for d in dirs if not any(ex in d for ex in exclude_patterns)]
                    
                    for f in files:
                        if any(f.endswith(ex) for ex in [".pyc", ".tmp", ".cache"]):
                            continue
                        file_path = Path(root) / f
                        rel_path = file_path.relative_to(root_dir)
                        
                        # Skip if in exclude patterns
                        if any(ex in str(rel_path).replace("\\", "/") for ex in exclude_patterns):
                            continue
                            
                        zipf.write(file_path, arcname=str(rel_path))
                        total_files += 1

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Packaged {total_files} files into {zip_filename}")
    print(f"[INFO] Archive Size: {zip_size_mb:.1f} MB (Easily shareable via WeChat/OneDrive!)\n")
    return zip_path


if __name__ == '__main__':
    create_zip_package()
