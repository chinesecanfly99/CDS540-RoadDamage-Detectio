# RDD2022 数据集下载与结构状态报告 (Data Status Report)

- **扫描时间**：2026-10-14
- **本地数据目录**：`DGS2022数据/`
- **清单产出文件**：[`data/manifests/data_inventory.csv`](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/data/manifests/data_inventory.csv)
- **总图片量**：19,208 张
- **总标注量**：15,340 个 XML 文件（覆盖 100% 训练集）

---

## 1. 国家与划分数据统计 (Summary Table)

| 国家 (Country) | 划分 (Split) | 图片数 (Images) | XML 标注数 (XMLs) | 图像分辨率 (Resolution) | 标注状态说明 |
|:---|:---:|:---:|:---:|:---:|:---|
| **Czech (捷克)** | `train` | 2,829 | 2,829 | 600 × 600 | 100% 具备 Pascal VOC XML 标注 |
| **Czech (捷克)** | `test` | 709 | 0 | 600 × 600 | 官方打榜测试集（无公开 GT） |
| **India (印度)** | `train` | 7,706 | 7,706 | 720 × 720 | 100% 具备 Pascal VOC XML 标注 |
| **India (印度)** | `test` | 1,959 | 0 | 720 × 720 | 官方打榜测试集（无公开 GT） |
| **United States (美国)** | `train` | 4,805 | 4,805 | 640 × 640 | 100% 具备 Pascal VOC XML 标注 |
| **United States (美国)** | `test` | 1,200 | 0 | 640 × 640 | 官方打榜测试集（无公开 GT） |
| **总计 (Total)** | — | **19,208** | **15,340** | — | **训练集全部配对无缺失** |

---

## 2. 数据目录结构说明

解压后的实际物理路径结构如下：

```text
DGS2022数据/
├── Czech/
│   ├── Czech/
│   │   ├── train/
│   │   │   ├── images/              # 2,829 张 JPG
│   │   │   └── annotations/xmls/    # 2,829 个 XML
│   │   └── test/
│   │       └── images/              # 709 张 JPG
│   └── Czech.zip                    # 原始压缩包 (245 MB)
├── India/
│   ├── India/
│   │   ├── train/
│   │   │   ├── images/              # 7,706 张 JPG
│   │   │   └── annotations/xmls/    # 7,706 个 XML
│   │   └── test/
│   │       └── images/              # 1,959 张 JPG
│   └── India.zip                    # 原始压缩包 (502 MB)
└── United_States/
    ├── United_States/
    │   ├── train/
    │   │   ├── images/              # 4,805 张 JPG
    │   │   └── annotations/xmls/    # 4,805 个 XML
    │   └── test/
    │       └── images/              # 1,200 张 JPG
    └── United_States.zip            # 原始压缩包 (424 MB)
```

---

## 3. 核心结论与后续实验建议

1. **训练标注完备度 100%**：
   三大国家的所有训练集图像（共 15,340 张）均拥有严格 1:1 同名的 Pascal VOC 格式 XML 文件，不存在图片丢失或空标问题。
2. **测试集无标注属于正常设计**：
   测试集的 3,868 张图像是 CRDDC 挑战赛打榜专用盲测集，没有真值标签。在后续实验中：
   - 模型评估与消融必须在有标注的 `train/` 中划分为 `train/val/test`；
   - 官方 `test/` 图像主要用于最终可视化与视频推理 Demo 演示。
3. **分辨率统一性**：
   各国家内部分辨率高度统一（捷克 600×600、印度 720×720、美国 640×640），后续直接统一缩放到 YOLO 常规输入尺寸（`imgsz=640`）几乎无损纵横比，非常适合训练。
