# CDS540: 跨国家道路损伤检测与人工复核决策辅助系统
> **Lingnan University - CDS540 Computer Vision Group Project**  
> **面向跨国家道路图像的损坏检测、泛化性评估与人工复核辅助系统 (Road Damage Detection & Human Review Support)**

---

## 📌 项目简介 (Project Overview)

本项目基于国际公路病害评测基准 **RDD2022**（包含捷克、印度与美国真实道路行车记录图像），构建了一套面向道路巡检场景的轻量化目标检测与人工复核辅助系统。

针对巡检人员在多变自然光照、跨国家路面材质以及低对比度条件下的排查需求，项目重点探讨：
1. **模型轻量化与规模权衡**：计划统一协议比较 YOLO11n 与 YOLO11s 的精度（mAP50、mAP50-95）、推理延迟与资源开销；已有权重的实际训练预算不同，正式比较需统一条件重训。
2. **跨域泛化性（Domain Shift）**：以捷克和印度为源域训练，评估模型在美国公路留出数据上的泛化表现与性能衰减边界。
3. **可解释的人工复核决策规则**：结合预设的类别复核权重（如 D40 坑洞重点监控）、检测置信度与几何特征，输出透明、可追踪的复核优先级建议。
4. **跨平台轻量交互演示 (Demo)**：已支持离线单图与媒体处理、图片上传网页及 JSON 导出；真实车载视频验收仍待实现。

> 以上为项目立项时的目标清单；**当前实际进展与最终交付物以下一节为准**。

---

## 英文答辩 PPT 与字幕演示（2026-11-12）

原答辩 PPT 已在 [同一路径](submission/CDS540_Final_Presentation_Deck.pptx) 更新为 **21 页英文版**。结构按“选题原因 → 设计目标与原理 → 总流程 → 功能关联 → 各步骤的方法、数据与实际结果 → 验收与结论”组织，正文统一浅蓝紫背景，含 7 个可编辑图表；不再单列 Questions 过渡页或附录页组。

[85 秒英文字幕演示](submission/RoadInsight_Demo_EN.mp4)、[SRT 字幕](submission/RoadInsight_Demo_EN.srt)、[讲稿与交接说明](submission/presentation_support/README.md) 已随 PPT 提供。PPT 和视频应下载到同一目录，第 19 页可打开视频。正式演示前仍需在答辩电脑上试播。

演示界面对应 [工作台 PR #4](https://github.com/pico0v0/CDS540-RoadDamage-Detectio/pull/4)，运行该分支或包含其改动的 main 才会得到相同界面。本次演示材料更新没有修改应用代码、模型、数据划分或最终报告；Render 仍是待确认方案。

---

## 🆕 最新进展：v2 技术升级（2026-11-12）

提交 `4d2ccbf` 完成 C 档技术升级（C1–C5）并合并进正式交付物。**本次没有更换检测模型**，而是把原报告中列为 limitations 的四条 "future work" 变成有数据、有边界、也如实记录负结果的实证：

| 编号 | 内容 | 结果（含负结果） |
|---|---|---|
| C1 | IPM 逆透视映射，把检测框面积换算为 `estimated_ground_area_m2` | 与 2D 面积排序 Spearman `0.8957`；优先级仅在 `2/40` 帧变化（有检测帧 `2/18`）；自动消失点标定仅 `9/40` 帧成功 |
| C2 | P3/P4/P5 层级激活能量分析 | 属相关性定位证据，非因果证明；D40 的 P5 占比低于背景，各类检出率偏低 |
| C3 | 美国域 few-shot 适应（`seed=42`，严格 holdout） | holdout mAP@50 `0.0863 → 0.1957`（100-shot）；但原域 `0.2845 → 0.1420`，出现灾难性遗忘与 D40 负迁移 |
| C4 | ONNX / FP16 / INT8 导出与 CPU 数值一致性实测 | FP32 parity PASS（`0.0002 px`）；FP16 / INT8 FAIL（漏框、分数偏差）；Jetson / TensorRT 未实测 |

**最终交付物（编译版已纳入本仓库）**：`submission/CDS540_Final_Project_Report.pdf`（7 页，课程上限 8 页）、`submission/CDS540_Final_Presentation_Deck.pptx`（21 页）、`submission/CDS540_Final_Project_Report.md`、`submission/defense_qa_guide.md`。

新代码、配置、清单与结果统一放在 `v2` 命名空间：`scripts/*_v2.py`、`configs/v2/`、`data/manifests/v2/`、`results/v2/`（含 11–14 号 300 DPI 主图）。旧文件只增不改，旧权重与新实验指标不混排；训练权重与 ONNX 导出因体积限制仍走共享存储。完整变更见 [CHANGELOG_C.md](CHANGELOG_C.md)，口径修订见 [submission/v2_candidates/discrepancy_log.md](submission/v2_candidates/discrepancy_log.md)。

---

## 📂 仓库目录结构 (Directory Layout)

```text
├── run_demo.py           # 统一启动，使用安装器选定的项目环境
├── start_demo.cmd / .sh  # 本机启动入口
├── requirements/        # 演示、CPU/CUDA、分析与报告依赖
├── configs/              # YOLO 数据集、跨域评估与本机配置示例
│   ├── data.yaml         # 源域 (捷克+印度) 训练/验证/测试集配置
│   └── data_domain_test.yaml # 美国跨域留出测试配置
├── data/                 # 数据集规范与划分依据
│   ├── manifests/        # 【核心依据】训练/验证/测试/跨域清单 CSV
│   └── README.md         # 原始数据获取与目录层级规范
├── demo/                 # 离线与交互演示程序
│   ├── demo.py           # 视频/图像序列推理及 HUD 叠框导出
│   ├── web_app.py        # 单图、双模型对比、样例交互页面
│   ├── inference.py      # 推理与导出服务（按模型/设备隔离缓存）
│   ├── cases.py          # 本机样例库与原标注参考
│   └── README.md         # 演示使用说明
├── docs/                 # 项目文档与官方指引
│   ├── DEMO_GUIDE.md     # 安装、使用与 AI 交接
│   ├── course_info/      # 课程原始 PDF、评分标准、选题指南
│   └── system_architecture.html # 端到端技术架构交互图
├── models/               # 两份已纳入 Git 的模型权重及校验说明
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
│   ├── setup_runtime.py / check_environment.py # 安装和原生算子检查
│   ├── demo_runtime.py / runtime_bootstrap.py   # 端口、设备、解释器
│   ├── prepare_demo_samples.py / evaluate_demo_cases.py # 样例与逐图回归
│   ├── voc_to_yolo.py    # XML 标注清洗与 YOLO 格式构建
│   ├── train_and_compare_models.py # 旧对比训练入口（需统一预算）
│   ├── evaluate_baseline.py        # 基线评估
│   ├── evaluate_domain_shift.py    # 跨域泛化评估
│   ├── test_robustness.py          # 扰动鲁棒性评测
│   ├── priority.py                 # 人工复核规则引擎
│   └── generate_final_report_pdf.py# 报告与交付物编译脚本
├── submission/           # 【最终交付物，以本目录为准】
│   ├── CDS540_Final_Project_Report.pdf / .md  # 7 页正式报告（编译版 + 源文）
│   ├── CDS540_Final_Presentation_Deck.pptx    # 21 页英文答辩 PPT
│   ├── RoadInsight_Demo_EN.mp4 / .srt        # 85 秒英文字幕演示
│   ├── presentation_support/                # 讲稿、来源、预览与交接
│   ├── defense_qa_guide.md                   # 答辩 Q&A 口径
│   └── v2_candidates/    # 候选补充章节与口径差异登记
├── configs/v2/           # few-shot / holdout 评测配置
├── data/manifests/v2/    # 新划分清单（三集合零交集）
├── results/v2/           # C1-C4 数据表与 11-14 号主图
├── CHANGELOG_C.md        # v2 升级变更记录
├── env_check.txt         # 原始训练环境软硬件记录
├── requirements.txt      # 项目依赖列表 (已包含 ultralytics, gradio)
└── .gitignore            # 忽略环境、数据和输出；放行最终 PDF/PPTX 与两份权重
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境安装与网页启动

推荐 Python 3.11，在项目根目录执行：

```bash
# 通用安装（CPU；Apple Silicon 也使用此入口）
python scripts/setup_runtime.py --backend cpu

# 有兼容 NVIDIA 显卡的 Windows / Linux 电脑可改用：
# python scripts/setup_runtime.py --backend cuda118

python run_demo.py
```

以后 Windows 可双击 `start_demo.cmd`；PyCharm 右键根目录 `run_demo.py`。
安装器会记录选定环境，页面自动选择可用设备。端口在本机 `.local/demo.json` 修改，
默认占用时自动顺延，控制台显示实际地址。旧环境混入全局包时，用 `--env .local/runtime-cpu` 新建。

网页现支持单模型检测、同图双模型并排对比、按国家/类别筛选样例、原始标注参考与 JSON 下载。
从三个国家 ZIP 可生成 60 张分层演示样例；程序运行不要求先训练。
完整的安装、样例准备、端口和 AI 交接见 [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)。
最小依赖与分析、报告依赖已分开；后两项按需安装，见该指南。

---

### 2. 准备模型权重

两份已有权重 `models/baseline_best.pt` (YOLO11n) 和 `models/compare_yolo11s_best.pt` (YOLO11s) 已直接纳入 `main`。新克隆或更新到包含 `76c155b` 的版本即可获得，无需另找网盘链接；如果使用历史代码压缩包，请重新获取当前仓库。
校验 SHA256 确保完整性（详见 [models/README.md](models/README.md)）。

---

### 3. 运行最小单图推理验证

在目标电脑上验证环境是否跑通：

```bash
python scripts/predict_image.py --source "path/to/road.jpg" --output ".local/single-image" --device cpu
```

将 `path/to/road.jpg` 替换为一张实际道路图片。命令会保存叠框图片和 JSON；输出目录应为新目录。默认加载已有 `models/baseline_best.pt`，可通过 `--model` 指定对照模型。

最新变更、已修复项与口径差异见 [CHANGELOG_C.md](CHANGELOG_C.md) 及 [submission/v2_candidates/discrepancy_log.md](submission/v2_candidates/discrepancy_log.md)。

2026-10-16 的本机 CPU 运行记录见 [docs/LOCAL_RUNTIME.md](docs/LOCAL_RUNTIME.md)。

两个模型可分别检测同一张图，无需二选一。示例命令（替换实际图片路径，使用未存在的输出目录）：

```bash
python scripts/predict_image.py --source "path/to/road.jpg" --model models/baseline_best.pt --output .local/compare-n-01 --device cpu
python scripts/predict_image.py --source "path/to/road.jpg" --model models/compare_yolo11s_best.pt --output .local/compare-s-01 --device cpu
```

这是依次运行两个模型并保存各自结果。在交互式网页中，已原生支持**双模型并排同图对比**与综合结果导出；预测级融合（Ensemble）仍需统一数据与训练后进一步探索。

---

### 4. 运行人工复核交互演示 (Web App)

推荐在项目根目录运行统一启动入口 `python run_demo.py`（Windows 也可直接双击 `start_demo.cmd`）。启动器会自动识别项目环境并探测空闲端口（默认 7860，被占用时自动顺延），避免多进程冲突。

网页支持：
- **单模型快速排查**：选择模型（YOLO11n / YOLO11s）与阈值，查看缺陷叠框、四类统计、优先级建议并导出单次 JSON。
- **双模型同图并排对比**：同图同推理配置并行查看两个模型的检出框差异与统计表格，一键下载对比诊断 JSON。
- **60 张跨国分层样例库**：支持按捷克、印度、美国及 4 类病害筛选，展开可实时核对原始标注参考（青色框）。

完整参数与交接指南见 [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) 及 [demo/README.md](demo/README.md)。

```bash
# 启动 Gradio 交互式网页
python run_demo.py

# 自动化验证网页接口、叠框与导出授权
python scripts/smoke_demo.py --device auto

# 离线批量图片或短视频生成
python demo/demo.py --source "path/to/images-or-video" --output ".local/demo.mp4" --device cpu --max-frames 120
```

将 `path/to/images-or-video` 替换为实际图片目录或短视频路径。图片目录生成的是图片序列演示；真实视频默认沿用源 FPS。不同尺寸图片会统一到首帧尺寸，并在 JSON 中记录。可用 `--fps` 修改输出播放速度，输出文件已存在时会提示更换名称。

除 `demo/demo.py` 外，媒体演示、训练指标复现与真实视频验收仍按 `planning/` 的阶段文档推进。

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
4. **数据与产物管理**：两份指定已有权重，以及最终报告 PDF 与答辩 PPTX（2026-11-12 起）已纳入仓库；其他训练 checkpoint、原始 ZIP、大批图片、运行环境和大视频继续通过共享存储分发，不随普通提交上传。不要覆盖已有权重来代替新实验版本。

---

## 🤖 AI 辅助开发约定 (AI Handoff)

换到新电脑或交给新的 AI 助手协助开发时，请优先阅读：
👉 **[planning/09_跨电脑运行问题与AI交接.md](planning/09_跨电脑运行问题与AI交接.md)**

该文档包含直接复制给 AI 的精准 Prompt 和阶梯式排错步骤，能大幅减少因环境差异导致的反复重跑。


当前状态以本 README 的「最新进展」与 [CHANGELOG_C.md](CHANGELOG_C.md) 为准，提交物以 `submission/` 为准；`planning/` 与 `reports/` 下的阶段文档属早期规划与历史副本，其进度口径可能滞后。跨电脑验收按组内要求暂缓。
