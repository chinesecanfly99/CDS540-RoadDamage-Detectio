# 本地演示与验收

## 图片上传网页

首次执行 `python scripts/setup_runtime.py --backend cpu`，需要 NVIDIA GPU 时选 `--backend cuda118`。
以后从根目录运行 `python run_demo.py` 或双击 `start_demo.cmd`。
PyCharm 运行根目录 `run_demo.py`；直接运行 `demo/web_app.py` 也会使用已选项目环境。

页面分为检测工作台、实验分析和案例库。工作台左侧设置图片和参数，右侧显示彩色统计卡片和检测框；案例库按国家/类别筛选缩略图，点击即可载入；实验页展示仓库已记录的 v2 图表与条件。

双模型支持同步缩放、拖动和逐框差异核对，原始标注可展开查看。
阈值、设备为两模型共用，结果可下载 JSON。两模型依次推理，不做融合；已有权重训练预算不同。
无检出不表示道路完好，复核优先级不代表实际严重程度。首次调用耗时不能作为稳定速度排名。

本机端口、设备在 `.local/demo.json` 配置；端口占用默认自动顺延。
网页仅监听本机。导出目录授权已经覆盖从 `demo/` 工作目录启动的情形，避免 JSON 下载路径错误。
完整安装、60 张样例生成和操作步骤见 [演示指南](../docs/DEMO_GUIDE.md)。

## 命令行单图对照

用同一张原图和同一模型核对网页结果。输出目录每次使用新名称：

```powershell
python scripts/predict_image.py --source "C:\path\road.jpg" --model models/baseline_best.pt --output .local/single-image-01 --device cpu --conf 0.25
```

应生成 `prediction.jpg` 和 `prediction.json`。网页使用类别彩色检测框，复核建议集中显示在统计卡片中；模型、阈值、NMS IoU、输入尺寸和检测类别应一致。

## 离线图片序列或视频

```powershell
python demo/demo.py --source "C:\path\road_images" --output .local/demo-01.mp4 --device cpu --max-frames 10
```

图片目录输出的是幻灯片式视频。真实视频请将 `--source` 改为视频路径，并单独检查导出帧数和播放情况。完整数据训练与指标评估需要原始数据 ZIP；网页和单图验收不需要全量数据。
