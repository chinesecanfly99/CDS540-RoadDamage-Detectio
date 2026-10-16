# CDS540: 跨国家道路损伤检测与人工复核决策辅助系统
> **Lingnan University - CDS540 Computer Vision Group Project**  
> **面向跨国家道路图像的损坏检测、泛化性评估与人工复核辅助系统 (Road Damage Detection & Human Review Support)**

---

## 📌 项目简介 (Project Overview)

本项目基于国际公路病害评测基准 **RDD2022**（包含捷克、印度与美国真实道路行车记录图像），构建了一套面向道路巡检场景的轻量化目标检测与人工复核辅助系统。

针对巡检人员在多变自然光照、跨国家路面材质以及低对比度条件下的排查需求，项目重点探讨：
1. **模型轻量化与规模权衡**：计划统一协议比较 YOLO11n 与 YOLO11s 的精度（mAP50、mAP50-95）、推理延迟与资源开销；旧实际训练协议尚待核对。
2. **跨域泛化性（Domain Shift）**：以捷克和印度为源域训练，评估模型在美国公路留出数据上的泛化表现与性能衰减边界。
3. **可解释的人工复核决策规则**：结合预设的类别复核权重（如 D40 坑洞重点监控）、检测置信度与几何特征，输出透明、可追踪的复核优先级建议。
4. **跨平台轻量交互演示 (Demo)**：已支持离线单图与媒体处理和 JSON 导出；上传网页和真实车载视频验收仍待实现。

---

## 📂 仓库目录结构 (Directory Layout)

```text
├── configs/              # YOLO 数据集与跨域评估配置文件
│   ├── data.yaml         # 源域 (捷克+印度) 训练/验证/测试集配置
│   └── data_domain_test.yaml # 美国跨域留出测试配置
├── data/                 # 数据集规范与划分依据
│   ├── manifests/        # 【核心依据】训练/验证/测试/跨域清单 CSV
│   └── README.md         # 原始数据获取与目录层级规范
├── demo/                 # 离线与交互演示程序
│   ├── demo.py           # 视频/图像序列推理及 HUD 叠框导出
│   └── README.md         # 演示使用说明
├── docs/                 # 项目文档与官方指引
│   ├── course_info/      # 课程原始 PDF、评分标准、选题指南
│   └── system_architecture.html # 端到端技术架构交互图
├── models/               # 模型权重说明 (权重文件通过网盘共享)
│   └── README.md         # 权重哈希校验码 (SHA256) 与下载说明
├── planning/             # 【小组成员执行规划与审计】
│   ├── README.md         # 审计规划总览
│   ├── STATUS.md         # 项目全局进度与验收看板
│   ├── DECISIONS.md      # 技术决策与变更日志
│   ├── 01_现状审计.md    # 历史实现审计与 16 项修复重点
│   ├── 02_项目定义与总流程.md
│   ├── 03_技术架构与环境.md
│   ├── 04_实验设计与指标.md
│   ├── 05_跨平台部署验收.md
│   ├── 06_评分映射与团队排期.md
│   ├── 07_AI执行约定.md
│   ├── 08_来源与待确认事项.md
│   ├── 09_跨电脑运行问题与AI交接.md # 换电脑运行与 AI 交接专用指南
│   └── stages/           # 阶段 01 ~ 08 独立推进文档
├── reports/              # 各模块详细分析报告 (Markdown 源文件)
│   ├── CDS540_Final_Project_Report.md # 主报告源文
│   ├── baseline_evaluation_report.md
│   ├── model_comparison_report.md
│   ├── domain_shift_and_robustness.md
│   ├── priority_rules.md
│   ├── presentation_slides_deck.md    # 答辩 PPT 结构草稿
│   └── defense_qa_guide.md            # 答辩问答预演指南
├── results/              # 实验结果数据、混淆矩阵与可视化图表
│   ├── eda/              # 探索性数据分析分布图
│   ├── evaluations/      # PR 曲线、混淆矩阵
│   ├── runtime/          # 运行时延迟测量日志
│   └── *.csv             # 模型对比、跨域及鲁棒性指标汇总
├── scripts/              # 全流程核心 Python 脚本
│   ├── voc_to_yolo.py    # XML 标注清洗与 YOLO 格式构建
│   ├── train_and_compare_models.py # 旧对比训练入口（需统一预算）
│   ├── evaluate_baseline.py        # 基线评估
│   ├── evaluate_domain_shift.py    # 跨域泛化评估
│   ├── test_robustness.py          # 扰动鲁棒性评测
│   ├── priority.py                 # 人工复核规则引擎
│   └── generate_final_report_pdf.py# 报告与交付物编译脚本
├── env_check.txt         # 原始训练环境软硬件记录
├── requirements.txt      # 项目依赖列表 (已包含 ultralytics, gradio)
└── .gitignore            # 忽略虚拟环境、大图像、权重及大视频
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境安装

推荐使用 Python 3.10 或 3.11 创建独立虚拟环境：

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2. 安装项目依赖
pip install --upgrade pip
pip install -r requirements.txt
```

> **跨平台加速说明**：
> - **Windows (NVIDIA GPU)**：若需使用 CUDA，请确保按 [PyTorch 官网](https://pytorch.org/) 安装与本机驱动匹配的 CUDA 版 PyTorch。若无 GPU，本地演示入口默认使用 CPU；确认 CUDA 可用后可显式传入 `--device 0`。
> - **macOS (Apple Silicon)**：支持 MPS 加速后端，可在兼容环境中显式传入 `--device mps`；先用 `--device cpu` 验证基本运行，无需安装 CUDA。

---

### 2. 准备模型权重

从小组成员共享网盘或 Release 中获取 `baseline_best.pt` (YOLO11n) 和 `compare_yolo11s_best.pt` (YOLO11s)，放入 `models/` 目录。
校验 SHA256 确保完整性（详见 [models/README.md](models/README.md)）。

---

### 3. 运行最小单图推理验证

在目标电脑上验证环境是否跑通：

```bash
python scripts/predict_image.py --source "path/to/road.jpg" --output ".local/single-image" --device cpu
```

将 `path/to/road.jpg` 替换为一张实际道路图片。命令会保存叠框图片和 JSON；输出目录应为新目录。默认加载已有 `models/baseline_best.pt`，可通过 `--model` 指定对照模型。

最新问题核查、已修复项和后续工作见 [planning/11_全量复核与继续实施.md](planning/11_全量复核与继续实施.md)。

2026-10-16 的本机 CPU 运行记录见 [docs/LOCAL_RUNTIME.md](docs/LOCAL_RUNTIME.md)。

---

### 4. 运行人工复核 Demo

```bash
python demo/demo.py --source "path/to/images-or-video" --output ".local/demo.mp4" --device cpu --max-frames 120
```

将 `path/to/images-or-video` 替换为实际图片目录或短视频路径。图片目录生成的是图片序列演示；真实视频默认沿用源 FPS。不同尺寸图片会统一到首帧尺寸，并在 JSON 中记录。可用 `--fps` 修改输出播放速度，输出文件已存在时会提示更换名称。

当前入口是离线脚本，不包含上传网页。完整数据训练与指标复现仍按 `planning/` 的阶段文档推进。

---

## 👥 团队分工与协作规范 (Collaboration Workflow)

项目采用阶段推进（Stage-by-Stage）与特性分支（Feature Branch）协作模式，避免多版本冲突。

### 1. 五人模块分工 (待组内认领)
- **A 模块 (基线与接口)**：跨平台运行环境适配、YOLO11n 基准跑通、统一推理与评估接口。
- **B 模块 (数据准备)**：四类标签口径核对、过滤异常非目标框、防相邻帧泄漏的分组划分清单。
- **C 模块 (模型实验)**：YOLO11s 对比受控实验、针对性图像增强创新点（如抗模糊/暗光）。
- **D 模块 (评估与规则)**：固定阈值评估、美国跨域测试、失败案例 (FP/FN) IoU 匹配、复核规则。
- **E 模块 (交互与交付)**：Gradio 交互式网页界面开发、Windows/Mac 离线验收、报告与答辩演练。

### 2. Git 分支管理约定
1. **主分支 (`main`)**：保持随时可运行状态，所有合并必须经过至少一名小组成员交叉审核。
2. **功能分支 (`feature/<task_name>`)**：每项任务从 `main` 切出新分支（如 `feature/gradio-ui`、`feature/data-v1`）。
3. **提交规范 (Commit Message)**：
   ```text
   [模块代号] 简要说明变更内容 (如: [B-Data] 修正 voc_to_yolo.py 路径解析并生成 data v1 清单)
   ```
4. **禁止上传大文件**：大图、原始 ZIP、`.pt` 模型权重、`.mp4` 大视频严禁直接 commit 到仓库，请利用 `data/manifests/` 或网盘链接共享。

---

## 🤖 AI 辅助开发约定 (AI Handoff)

换到新电脑或交给新的 AI 助手协助开发时，请优先阅读：
👉 **[planning/09_跨电脑运行问题与AI交接.md](planning/09_跨电脑运行问题与AI交接.md)**

该文档包含直接复制给 AI 的精准 Prompt 和阶梯式排错步骤，能大幅减少因环境差异导致的反复重跑。


本轮已经完成原划分的全量转换、两模型完整评估及阈值试验，发现同场景跨集合和训练预算差异；最新状态见仓库 planning/11_全量复核与继续实施.md。跨电脑验收按用户要求暂缓；本目录此前记录为历史快照。
