"""Local Gradio image upload demo for the existing road-damage checkpoints."""

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

# Gradio checks its own loopback endpoint during launch. Some Windows proxy
# settings otherwise route that request through a proxy and return HTTP 502.
_no_proxy = {entry.strip() for entry in (os.environ.get("NO_PROXY") or os.environ.get("no_proxy", "")).split(",") if entry.strip()}
_no_proxy.update({"127.0.0.1", "localhost"})
os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(sorted(_no_proxy))

import cv2
import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SETTINGS = ROOT / ".local" / "ultralytics"
LOCAL_SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_SETTINGS))
sys.path.insert(0, str(ROOT))

from scripts.priority import RoadDamagePrioritizer  # noqa: E402
from ultralytics import YOLO  # noqa: E402


MODELS = {
    "YOLO11n (baseline)": ROOT / "models" / "baseline_best.pt",
    "YOLO11s (comparison)": ROOT / "models" / "compare_yolo11s_best.pt",
}
EXPECTED_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
EXPORT_DIR = ROOT / ".local" / "web_exports"


@lru_cache(maxsize=2)
def load_model(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    path = MODELS[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Model weight is missing: {path}")
    model = YOLO(str(path))
    if model.names != EXPECTED_NAMES:
        raise ValueError(f"Unexpected class mapping in {path.name}: {model.names}")
    return model


def analyze_image(image_rgb, model_name, confidence, device):
    """Return an RGB overlay and a serializable review record for one image."""
    if image_rgb is None:
        raise ValueError("请先上传一张原始道路图片。")
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("图片必须是 RGB 彩色图片。")
    if image_rgb.dtype != np.uint8 or min(image_rgb.shape[:2]) == 0:
        raise ValueError("无法读取上传的图片。")
    if model_name not in MODELS:
        raise ValueError("请选择列表中的模型。")
    if not np.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("置信度阈值必须在 0 到 1 之间。")
    if device not in {"cpu", "0"}:
        raise ValueError("请选择 CPU 或 CUDA:0。")
    if device == "0" and not torch.cuda.is_available():
        raise ValueError("当前 Python 环境未识别 CUDA，请先选 CPU 并检查 PyTorch。")

    image_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    model = load_model(model_name)
    started = time.perf_counter()
    result = model.predict(
        image_bgr, device=device, conf=float(confidence), iou=0.7,
        imgsz=640, rect=False, max_det=300, verbose=False,
    )[0]
    predict_ms = round((time.perf_counter() - started) * 1000, 2)
    boxes = result.boxes
    triage = RoadDamagePrioritizer(conf_threshold=float(confidence)).evaluate_detections(
        boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), width, height,
    )
    overlay_bgr = RoadDamagePrioritizer().render_overlay(image_bgr, triage)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    checkpoint = MODELS[model_name]
    report = {
        "model": model_name,
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "device": str(boxes.data.device),
        "image_size": {"width": width, "height": height},
        "confidence_threshold": float(confidence),
        "nms_iou": 0.7,
        "imgsz": 640,
        "rect": False,
        "max_det": 300,
        "predict_call_ms_including_first_call": predict_ms,
        "review": triage,
        "note": "复核优先级是启发式建议；无检出不代表道路完好，也不代表实际损坏严重程度。",
    }
    return overlay_rgb, report


def run_web_prediction(image_rgb, model_name, confidence, device):
    try:
        overlay, report = analyze_image(image_rgb, model_name, confidence, device)
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix="road-review-",
        dir=EXPORT_DIR, delete=False,
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        export_path = handle.name
    review = report["review"]
    status = (
        f"**复核优先级：{review['priority_level']}** · "
        f"检测框：{review['total_distress_count']} · "
        f"设备：{report['device']} · 预测调用：{report['predict_call_ms_including_first_call']} ms\n\n"
        "此优先级仅用于人工复核排序；无检出不表示道路完好。"
    )
    return overlay, status, report, export_path


def build_app():
    with gr.Blocks(title="道路损伤检测与人工复核") as app:
        gr.Markdown(
            "# 道路损伤检测与人工复核\n"
            "上传原始道路图片，查看检测框与人工复核建议。默认阈值 0.25；降低阈值可能增加召回，也会增加误报。"
        )
        with gr.Row():
            image = gr.Image(label="原始道路图片", type="numpy", image_mode="RGB")
            overlay = gr.Image(label="检测与复核结果", type="numpy", interactive=False)
        with gr.Row():
            model = gr.Dropdown(choices=list(MODELS), value=list(MODELS)[0], label="模型")
            confidence = gr.Slider(minimum=0.05, maximum=0.60, value=0.25, step=0.05, label="置信度阈值")
            device = gr.Dropdown(choices=[("CPU", "cpu"), ("NVIDIA GPU (CUDA:0)", "0")], value="cpu", label="运行设备")
        run = gr.Button("开始检测", variant="primary")
        status = gr.Markdown()
        with gr.Accordion("检测与复核详情", open=False):
            details = gr.JSON(label="完整 JSON")
        download = gr.File(label="下载 JSON")
        run.click(run_web_prediction, [image, model, confidence, device], [overlay, status, details, download])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    build_app().launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
