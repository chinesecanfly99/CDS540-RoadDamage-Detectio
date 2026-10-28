"""Road inference and export service, independent from Gradio components."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.local/ultralytics'))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scripts.demo_runtime import resolve_device
from scripts.priority import RoadDamagePrioritizer

MODELS = {'YOLO11n (baseline)': ROOT / 'models/baseline_best.pt',
          'YOLO11s (comparison)': ROOT / 'models/compare_yolo11s_best.pt'}
EXPECTED_NAMES = {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}
EXPORT_DIR = ROOT / '.local/web_exports'
# A predictor stores mutable state. Serialize inference across the UI's tabs.
INFERENCE_LOCK = threading.RLock()


@lru_cache(maxsize=6)
def load_model(model_name, device='cpu'):
    if model_name not in MODELS:
        raise ValueError(f'未知模型：{model_name}')
    path = MODELS[model_name]
    if not path.is_file():
        raise FileNotFoundError(f'缺少模型权重：{path.name}，请更新完整仓库。')
    model = YOLO(str(path))
    if model.names != EXPECTED_NAMES:
        raise ValueError(f'{path.name} 的类别映射不匹配：{model.names}')
    return model


@lru_cache(maxsize=2)
def model_hash(model_name):
    with MODELS[model_name].open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def analyze_image(image_rgb, model_name, confidence, device='auto'):
    if image_rgb is None:
        raise ValueError('请先上传一张原始道路图片。')
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError('图片必须是 RGB 彩色图片。')
    if image_rgb.dtype != np.uint8 or min(image_rgb.shape[:2]) == 0:
        raise ValueError('无法读取上传的图片。')
    if model_name not in MODELS:
        raise ValueError('请选择列表中的模型。')
    if not np.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError('置信度阈值必须在 0 到 1 之间。')
    actual_device = resolve_device(device, torch)
    image_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    with INFERENCE_LOCK:
        # Separate predictors per device prevent stale CPU/CUDA state on switching.
        model = load_model(model_name, actual_device)
        started = time.perf_counter()
        result = model.predict(image_bgr, device=actual_device, conf=float(confidence), iou=.7,
                               imgsz=640, rect=False, max_det=300, verbose=False)[0]
        if result.boxes.data.is_cuda:
            torch.cuda.synchronize()
        predict_ms = round((time.perf_counter() - started) * 1000, 2)
        boxes = result.boxes
        triage = RoadDamagePrioritizer(conf_threshold=float(confidence)).evaluate_detections(
            boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), width, height)
        actual = str(boxes.data.device)
    overlay = RoadDamagePrioritizer().render_overlay(image_bgr, triage)
    report = {'model': model_name, 'model_sha256': model_hash(model_name),
              'requested_device': device, 'device': actual,
              'image_sha256': hashlib.sha256(image_rgb.tobytes()).hexdigest(),
              'image_size': {'width': width, 'height': height}, 'confidence_threshold': float(confidence),
              'nms_iou': .7, 'imgsz': 640, 'rect': False, 'max_det': 300,
              'predict_call_ms_including_first_call': predict_ms, 'review': triage,
              'note': '复核优先级是启发式建议；无检出不代表道路完好。单次调用耗时包含首次初始化，不能作为正式速度排名。'}
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), report


def export_json(report, prefix='road-review-'):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', prefix=prefix,
                                     dir=EXPORT_DIR, delete=False) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        return handle.name


def compare_images(image_rgb, confidence, device):
    outputs = [analyze_image(image_rgb, model, confidence, device) for model in MODELS]
    reports = [report for _, report in outputs]
    rows = [[report['model'], report['device'], report['review']['total_distress_count'],
             *[report['review']['defect_summary'][name] for name in EXPECTED_NAMES.values()],
             report['review']['priority_level'], report['predict_call_ms_including_first_call']]
            for report in reports]
    combined = {'comparison_type': 'same image and inference settings; sequential inference, no ensemble',
                'note': '现有两份权重训练预算不同；检出数量和个例不能替代统一条件下的正式评估。',
                'reports': reports}
    return outputs[0][0], outputs[1][0], rows, combined, export_json(combined, 'road-comparison-')
