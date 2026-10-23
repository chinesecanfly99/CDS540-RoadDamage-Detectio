# 本地演示与验收

## 图片上传网页

从项目根目录运行。先在 VS Code 中选择安装了 PyTorch、Ultralytics、OpenCV、Pandas 和 Gradio 的 Python 解释器；Windows 可用 `Python: Select Interpreter`。确认当前终端与所选解释器一致：

```powershell
python -c "import sys, torch, ultralytics, gradio, cv2; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
python demo/web_app.py
```

若 pip 报清华镜像 SSL 错误，可显式使用 `python -m pip install -i https://pypi.org/simple -r requirements.txt`。此电脑测试时需要绕过系统代理访问本机地址，网页入口已为 `127.0.0.1` 和 `localhost` 设置代理例外。

浏览器打开 `http://127.0.0.1:7860`。上传一张**未经叠框的原始道路图片**，选 YOLO11n 或 YOLO11s、阈值和 CPU/CUDA:0，点击“开始检测”。页面返回叠框图、复核优先级、完整 JSON 和可下载的 JSON。程序只监听本机，不生成公开分享链接。可用 `--port 7861` 更改端口，按 `Ctrl+C` 停止。

两份项目权重必须位于 `models/`。默认阈值 0.25；0.10 是偏向召回的实验候选，会增加误报。网页中的优先级来自项目启发式规则，不表示道路完好与否、实际损坏严重程度或维修时限。首次点击包括模型加载，耗时不能当作稳定运行帧率。

## 命令行单图对照

用同一张原图和同一模型核对网页结果。输出目录每次使用新名称：

```powershell
python scripts/predict_image.py --source "C:\path\road.jpg" --model models/baseline_best.pt --output .local/single-image-01 --device cpu --conf 0.25
```

应生成 `prediction.jpg` 和 `prediction.json`。网页叠框图使用人工复核 HUD，因此画面样式与命令行输出不同；模型、阈值、NMS IoU、输入尺寸和检测类别应一致。

## 离线图片序列或视频

```powershell
python demo/demo.py --source "C:\path\road_images" --output .local/demo-01.mp4 --device cpu --max-frames 10
```

图片目录输出的是幻灯片式视频。真实视频请将 `--source` 改为视频路径，并单独检查导出帧数和播放情况。完整数据训练与指标评估需要原始数据 ZIP；网页和单图验收不需要全量数据。
