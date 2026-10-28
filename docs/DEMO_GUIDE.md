# 演示安装、运行与 AI 交接

## 第一次运行

先克隆完整仓库，并安装 Python 3.11。两份道路检测权重已经在 `models/` 中。
在项目根目录打开终端，二选一安装：

```bash
# 普通电脑：CPU；Apple Silicon 也用此安装入口，再由环境判断 MPS 是否可用
python scripts/setup_runtime.py --backend cpu

# 有 NVIDIA 显卡且驱动支持 CUDA 11.8 的 Windows / Linux 电脑
python scripts/setup_runtime.py --backend cuda118
```

安装器创建独立 `.venv`、安装匹配的 Torch / Torchvision、检查真实算子和模型加载，
通过后将解释器写入本机 `.local/runtime-env.json`。不修改全局 Python。
旧环境如果继承了全局软件包，安装器会要求换一个位置，例如：

```bash
python scripts/setup_runtime.py --backend cuda118 --env .local/runtime-gpu
```

下载较慢时可原命令重试。`--index-url https://你的可信镜像/simple` 只替换一般依赖来源，
Windows / Linux 的 Torch 包仍使用官方 CPU / CUDA 11.8 索引。
这里的 CUDA 版本指 PyTorch 自带运行库；通常不需要另装 CUDA Toolkit。
如 GPU 检查失败，先根据错误核对驱动与解释器；CPU 安装方案可独立使用。

## 以后点哪个文件

- Windows：双击项目根目录 `start_demo.cmd`。保留运行窗口；关闭窗口会停止服务。
- 终端：在项目根目录执行 `python run_demo.py`，停止时按 Ctrl+C。
- PyCharm：打开项目根目录，选择已有环境的 Python 解释器，右键根目录 `run_demo.py` 运行。
  新入口会使用安装器选定的项目环境；控制台显示实际解释器路径。
  直接运行 `demo/web_app.py` 也会转到选定环境；无需靠 IDE 名称猜测用了哪个 Python。

本机的环境路径不提交 Git，也不能把一台电脑的 `.venv` 复制给另一台电脑。
其他成员应执行安装步骤。首次无网络安装、远程公网部署不在此入口覆盖范围内。

## 端口与设备配置

第一次启动自动创建 `.local/demo.json`，可参考 `configs/demo.example.json`：

```json
{"port": 7860, "auto_port": true, "device": "auto"}
```

只改此本机配置即可，不必修改 Python 源码。端口占用时默认查找后续 20 个端口中的空闲项，
以控制台打印的地址为准。命令行覆盖配置：

```bash
python run_demo.py --port 7870
python run_demo.py --port 7870 --strict-port
python run_demo.py --device cpu
```

`--strict-port` 要求端口可用，否则报错退出。启动器不会关闭占用端口的其他程序。
`auto` 优先 CUDA，再 MPS，最后 CPU；下拉框只列出当前环境可识别的设备。
要检查实际解释器，在选定环境中运行 `python scripts/check_environment.py`。
CUDA 方案增加 `--require-gpu`，会执行 GPU 张量计算和 torchvision NMS，避免只看显卡名称。

## 样例准备与使用

原始图片受数据集来源约束且体积较大，不随本 PR 重复上传。三个国家 ZIP 放在同一个目录后执行：

```bash
python scripts/prepare_demo_samples.py --archive-root "你的三个国家ZIP所在目录" --per-country 20
```

只读取现有 `test.csv` / `domain_test.csv` 的图片，从每国按四类目标和无四类标注的场景挑选，
默认共 60 张。结果在 `.local/demo_samples/`，包含原图、来源成员名、原划分、SHA256 与标注框。
选择过程不读取模型预测，移除完全重复图；不修改任何训练/验证/测试清单或标注。
“无四类目标标注”不等于道路正常；样例分层挑选，不能据此推算整个数据集准确率。

重启网页后即可使用：

1. 按国家、类别筛选样例，点击“载入所选样例”，也可拖入自己的原始道路图片。
2. 展开“原始标注参考”查看青色标注框。上传自己的图片时会清除样例标注，避免错配。
3. “单模型检测”：选择模型和阈值，查看叠框、四类统计、复核优先级与 JSON。
4. “双模型对比”：同图同推理设置依次运行两个模型，并排显示结果和类别统计，下载组合 JSON。
5. 可调低阈值观察误报/漏检变化，再结合原始标注逐个解释；不要用“框更多”判定模型更好。

样例回传页面使用 PNG 无损传输，避免二次有损压缩改变细裂缝。
切换图片、阈值或设备后会清空旧结果；等待样例载入完成，再点击检测。

两模型对比不是融合。现有权重训练预算不同，仍需统一数据版本与训练条件后正式比较。
单次调用时间含可能的首次初始化，只供观察，不作为速度基准。

## 更多功能性回归

```bash
python scripts/evaluate_demo_cases.py --device auto
python scripts/smoke_demo.py --device auto
python -m unittest discover -s tests -v
```

第一条用两个模型遍历全部样例，保存逐图 CSV、完整 JSON、IoU=0.5 同类匹配 TP/FP/FN。
这是固定阈值的个例诊断，不是 mAP，也不是独立代表性评估。输出默认在 `.local/case-regression/`。
第二条启动临时网页，从 `demo/` 工作目录实测双模型叠框、JSON 下载和目录授权，结束后关闭自己启动的服务。
无样例库时增加 `--source "原始道路图片路径"`。将输出的 `.local/demo-smoke/summary.json` 交回小组，
才构成该台电脑的实际运行记录。
正式评估继续使用原有 `evaluate_models.py` 等工具，保留统一数据清单与模型哈希。

## 功能分层与可选依赖

```text
run_demo.py / start_demo.*       统一启动入口
scripts/setup_runtime.py        独立环境安装
scripts/runtime_bootstrap.py    解释器选择与转交
scripts/demo_runtime.py         本机配置、设备与端口
scripts/check_environment.py    原生算子和权重检查
demo/web_app.py                 页面组件和事件
demo/inference.py               两模型推理、缓存、JSON 导出
demo/cases.py                   样例筛选、标注参考
scripts/prepare_demo_samples.py 样例生成与来源清单
scripts/evaluate_demo_cases.py  双模型样例回归
scripts/smoke_demo.py           真实网页接口与导出验收
```

最小演示依赖在 `requirements/runtime.txt`。数据分析另装 `pip install -r requirements/analysis.txt`；
生成报告另装 `pip install -r requirements/reports.txt`，避免仅演示时被文档工具阻塞。
CLI 脚本应使用所选环境的 Python；终端里普通 `python` 不一定与启动器选中环境一致。

## 给下一位 AI 的验收要求

检查 Git 分支、未提交改动、`.local/runtime-env.json` 实际解释器、依赖检查、两个权重。
先确认服务使用的端口和进程归属，再测试，禁止为腾端口结束无关进程。
至少验证两模型检测、CPU/GPU切换（硬件可用时）、双模型对比、样例过滤和原标注、JSON 下载。
从项目根目录和 `demo/` 工作目录分别测试，确认下载不再报 `InvalidPathError`。
导出授权仅限 `.local/web_exports` 与 `.local/demo_samples`，不可扩大到整个磁盘。
回归结果写入本机 `.local/`，只提交必要的代码、文档与简要验证结论。
没有在另一台真实电脑运行就保留“跨电脑待验收”，不可用模拟设备测试冒充实际验收。
