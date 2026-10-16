# 数据集说明 (Dataset & Manifests)

本项目使用 **Road Damage Dataset 2022 (RDD2022)**，涵盖捷克 (Czech)、印度 (India) 与美国 (United States) 三国公路图像数据。

为了避免数十 GB 的图像文件膨胀 Git 仓库，图像文件已通过 `.gitignore` 排除，仓库中保留了**完整的数据集划分清单与校验清单 (`data/manifests/`)**，确保所有小组成员环境下的训练、验证与测试完全对齐。

---

## 1. 原始数据获取与目录结构规范

请下载三国原始压缩包并统一解压到 `data/raw/` 或项目根目录下的数据目录：
- `Czech.zip`
- `India.zip`
- `United_States.zip`

推荐统一解压路径结构如下（避免嵌套多层同名国家目录）：
```text
data/
├── manifests/            # Git 跟踪：全量划分与校验清单
│   ├── train.csv         # 训练集样本清单 (捷克 + 印度)
│   ├── val.csv           # 验证集样本清单
│   ├── test.csv          # 源域独立测试集清单
│   ├── domain_test.csv   # 美国跨域外评估清单
│   ├── data_inventory.csv# 全量原始样本盘点
│   └── annotation_check.csv # XML 标注检查
└── raw/                  # 本地存放，不上传 Git
    ├── Czech/
    │   └── train/ (images & annotations)
    ├── India/
    │   └── train/ (images & annotations)
    └── United_States/
        └── train/ (images & annotations)
```

---

## 2. 目标损伤四分类定义 (Target Classes)

模型聚焦国际道路损伤评测定义的 4 类核心表面病害：

| 类别编码 | 中文名称 | 英文描述 | 备注 |
| :---: | :---: | :---: | :--- |
| **D00** | 纵向裂缝 | Longitudinal Crack | 含 `D0w0` 笔误显式映射 |
| **D10** | 横向裂缝 | Transverse Crack | 结构性横向裂纹 |
| **D20** | 龟裂/网状裂缝 | Alligator Crack | 疲劳龟裂 |
| **D40** | 坑洞 | Pothole | 重点复核对象 |

> **数据清洗说明**：
> 印度原始数据中的非目标标注类别（如 D44 减速带/人行道、D01、D11、D50 等共 1,371 个非目标框）在生成 YOLO 标注时被过滤剔除；不含上述 4 类的图片作为**负样本（背景图）**参与训练以抑制误报率，但不可随意称作“经人工确认的完好道路”。

---

## 3. 快速生成 YOLO 格式数据

配置好原始数据路径后，运行转换脚本即可利用本目录下的 `manifests/*.csv` 快速生成对齐的 YOLO 格式：
```bash
python scripts/voc_to_yolo.py
```
生成的 `data/yolo/images/` 与 `data/yolo/labels/` 会自动与 `configs/data.yaml` 对应。
