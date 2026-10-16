# Technical Milestone Reports & Experimental Findings

**Course**: Lingnan University CDS540 Computer Vision  
**Project**: Automated Road Damage Detection for Smart City Infrastructure  

This directory contains intermediate engineering reports and technical analyses produced across Steps 1 through 13 of the project pipeline. For the final consolidated Moodle submission, please refer to the `submission/` directory.

---

## 📑 Milestone Reports Index

| Step | Report File | Technical Scope | Key Finding / Deliverable |
|:---:|---|---|---|
| **Step 1** | [data_status.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/data_status.md) | Raw data inventory & verification | Verified 19,208 raw images across Czech, India, and US |
| **Step 4** | [eda_findings.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/eda_findings.md) | Exploratory Data Analysis | 15,340 VOC XMLs parsed, class imbalance & aspect ratio findings |
| **Step 5** | [split_policy.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/split_policy.md) | Dataset partitioning policy | Image-level splitting with fixed seed=42; 100% US held out |
| **Step 8** | [baseline_evaluation_report.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/baseline_evaluation_report.md) | Baseline model performance | YOLO11n evaluated on 1,579 test images (mAP50=0.2169, D40 Recall=33.3%) |
| **Step 9** | [model_comparison_report.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/model_comparison_report.md) | Model scale comparison | YOLO11s delivers +31.2% mAP50 (0.2846) & +19.2% D40 Recall (39.7%) |
| **Step 10** | [domain_shift_and_robustness.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/domain_shift_and_robustness.md) | Cross-domain & robustness | Longitudinal cracks decay by only -4.98%; motion blur preserves potholes |
| **Step 11** | [priority_rules.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/priority_rules.md) | Municipal priority heuristics | Multi-factor risk formula (D40 10x weight, area penalty, clustering) |
| **Step 13** | [runtime_deployment_report.md](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/reports/runtime_deployment_report.md) | Latency & edge deployment | Precision RTX 5070 timing (P95 29.6ms, 38.6 FPS); Jetson Orin Nano selection |
