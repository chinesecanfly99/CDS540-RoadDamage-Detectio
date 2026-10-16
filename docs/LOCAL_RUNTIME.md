# 本机运行记录

日期：2026-10-16。分支：`feature/local-runtime`。以下结论来自实际运行，未重新训练或复算 mAP。

## 结果

- 两份已有道路检测权重均成功加载，四类映射为 D00/D10/D20/D40。
- CPU 单图推理成功，叠框图与 JSON 均可读取。一个捷克样例两模型均零检出；印度样例 n/s 分别检出 3/5 个框。它们只是功能样例，不代表精度比较。
- 三个国家各取 4 张原始图片，共 12 张，成功导出 12 帧、4 秒的图片序列视频；混合分辨率处理通过。
- 使用上述原图合成一个无检测框的视频输入，成功处理前 8 帧并导出视频；验证了解码、处理、编码链路，尚未验证真实连续行车视频。
- 空目录、缺失输入、重复输出会给出错误；不会静默覆盖已有结果。
- `python -m pip check` 通过。视频输出已逐帧解码核对帧数、尺寸和 FPS。

## 本轮修正

原 `python demo/demo.py` 因默认目录 `data/yolo/images/test` 不存在而失败。本轮新增单图推理入口，演示入口增加 model/source/output/device/max-frames/fps 参数，增加图像与视频读写检查、混合尺寸处理和输出防覆盖。画面显示实际设备，预测调用耗时与整个处理循环耗时分别记录，不再写死 RTX 5070 或宣称完整实时性能。

模型与原始数据保持原样。优先级模块沿用原启发式规则，尚未验证其实际维护建议的有效性；不得将无检出解释成道路完好。网页界面、训练和全量评估不在本轮通过范围内。

## 实际环境

Python 3.11.1；PyTorch 2.14.0+cpu；torchvision 0.29.0+cpu；Ultralytics 8.4.177。完整版本记录位于 `.local/runtime/environment.json`。

本机 `.venv` 采用 `--system-site-packages`，复用原有 CPU PyTorch，并将新增库安装在项目环境。这是本次的本机运行方案，不是可直接拷给其他电脑的完整独立环境；新机器应重新创建适配环境。

## 本机重复运行

在项目根目录，用现有环境执行，输出名称每次换新：

```powershell
.\.venv\Scripts\python.exe scripts/predict_image.py --source .local/runtime/samples/India_000209.jpg --output .local/manual-image-01 --device cpu
.\.venv\Scripts\python.exe demo/demo.py --source .local/runtime/samples --output .local/manual-demo-01.mp4 --device cpu --max-frames 12 --fps 3
```

原始图片样例、模型权重与 `.venv` 都不提交到 Git。新克隆的电脑需要按说明获取权重和至少一张实际道路图，再将 source 改为实际位置；不必为单图演示准备全部数据。

## 证据位置

本机 `.local/runtime/` 保存 original_demo.log、sample_manifest.json、smoke_n/s、positive_n/s、demo_slideshow.mp4/json、demo_video.mp4/json、verification.json 和运行命令记录。

这些视频是受控功能测试材料，包含图片合成的视频输入，不能写成真实巡检视频测试或部署实时性结论。几张样例的计时也不能替代正式性能评估。
