"""Live diagnostic experiments sharing the workbench's model and inference protocol."""
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from demo.inference import INFERENCE_LOCK, MODELS, analyze_image, load_model
from demo.presentation import render_boxes
from scripts.demo_runtime import resolve_device
from scripts.ipm_geometry_v2 import RoadPlaneIPM, canonical_road_quad
from scripts.priority_v2 import RoadDamagePrioritizerV2
from scripts.rdd_quality import box_iou

ASSETS = Path(__file__).parent / 'assets/lab_cases'
CODES = ('D00', 'D10', 'D20', 'D40')


def cases():
    path = ASSETS / 'manifest.json'
    return json.loads(path.read_text(encoding='utf-8'))['cases'] if path.is_file() else []


def load_case(case_id):
    item = next((item for item in cases() if item['id'] == case_id), None)
    if item is None:
        raise ValueError('Select a valid diagnostic case.')
    path = (ASSETS / item['image']).resolve()
    if not path.is_relative_to(ASSETS.resolve()) or not path.is_file():
        raise ValueError('Diagnostic source image is missing.')
    with Image.open(path) as image:
        return np.array(image.convert('RGB')), item


def transform(image, brightness=1., blur=0):
    if image is None:
        raise ValueError('Load a diagnostic case or send an image from the workspace first.')
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('An RGB image is required.')
    if not .2 <= float(brightness) <= 1.5 or int(blur) not in (0, 3, 7, 11):
        raise ValueError('Invalid image challenge settings.')
    result = image.copy() if float(brightness) == 1. else np.clip(np.rint(image.astype(np.float32) * float(brightness)), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(result, (int(blur), int(blur)), 0) if int(blur) else result


def perspective(image, model, confidence, device, lane_width, brightness=1., blur=0):
    image = transform(image, brightness, blur)
    if not 3 <= float(lane_width) <= 4:
        raise ValueError('Assumed lane width must be 3–4 m.')
    _, report = analyze_image(image, model, confidence, device)
    h, w = image.shape[:2]
    ipm = RoadPlaneIPM(lane_width_m=float(lane_width), lane_length_m=20., camera_height_m=1.2)
    quad = canonical_road_quad(image.shape)
    ipm.calibrate_from_quad(quad, image.shape)
    original = render_boxes(image, report)
    cv2.polylines(original, [quad.astype(np.int32)], True, (255, 210, 0), 2)
    bird = ipm.warp_image(image)
    items = report['review']['items']
    boxes = np.array([i['box'] for i in items], dtype=float).reshape(-1, 4)
    classes = np.array([CODES.index(i['class_code']) for i in items])
    scores = np.array([i['confidence'] for i in items])
    triage = RoadDamagePrioritizerV2(conf_threshold=float(confidence), use_ipm=True, ipm=ipm).evaluate_detections(boxes, classes, scores, w, h)
    rows = []
    for item in triage['items']:
        area = item['estimated_ground_area_m2']
        rows.append([item['class_code'], round(item['confidence'], 3), round(item['area_ratio'] * 100, 3),
                     area, item['ipm_fallback_to_2d']])
        polygon = ipm.box_to_ground_polygon(item['box'])
        if polygon is not None and area is not None:
            cv2.polylines(bird, [polygon.astype(np.int32)], True, (80, 230, 120), 2)
    return original, bird, {'kind': 'perspective', 'prediction': report, 'rows': rows,
                            'ipm_review': triage, 'lane_width_m': float(lane_width),
                            'brightness': float(brightness), 'blur_kernel': int(blur)}


def feature_maps(image, model, confidence, device, brightness=1., blur=0):
    image = transform(image, brightness, blur)
    actual = resolve_device(device, torch)
    with INFERENCE_LOCK:
        detector = load_model(model, actual)
        if detector.predictor is None:
            analyze_image(image, model, confidence, device)  # initialize/fuse before installing hooks
        # AutoBackend can own a fused copy distinct from YOLO.model. Hook the
        # network actually executed by this cached predictor.
        sequence = detector.predictor.model.model.model
        indices = sequence[-1].f
        if not isinstance(indices, (list, tuple)) or len(indices) != 3:
            raise ValueError('This model does not expose three detection-pyramid inputs.')
        captured, hooks = {}, []
        def capture(level):
            def hook(_module, _inputs, output):
                captured[level] = output.detach().float().clone()
            return hook
        try:
            for level, index in zip(('P3', 'P4', 'P5'), indices):
                hooks.append(sequence[index].register_forward_hook(capture(level)))
            _, report = analyze_image(image, model, confidence, device)
        finally:
            for hook in hooks:
                hook.remove()
        h, w = image.shape[:2]
        scale = min(640 / h, 640 / w)
        nh, nw = round(h * scale), round(w * scale)
        top, left = round((640 - nh) / 2 - .1), round((640 - nw) / 2 - .1)
        maps = {}
        for level in ('P3', 'P4', 'P5'):
            feat = captured[level]
            expected = {'P3': 80, 'P4': 40, 'P5': 20}[level]
            if feat.shape[-2:] != (expected, expected):
                raise ValueError('Feature strides do not match the 640×640 inference protocol.')
            rms = torch.sqrt(torch.mean(feat.square(), dim=1, keepdim=True))
            padded = F.interpolate(rms, (640, 640), mode='bilinear', align_corners=False)[0, 0]
            cropped = padded[top:top + nh, left:left + nw]
            energy = F.interpolate(cropped[None, None], (h, w), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            maps[level] = energy / (float(np.sqrt(np.mean(energy.astype(np.float64) ** 2))) + 1e-8)
    overlays = []
    for energy in maps.values():
        heat = cv2.applyColorMap(np.uint8(np.clip(energy / max(float(energy.max()), 1e-8), 0, 1) * 255), cv2.COLORMAP_JET)
        overlays.append(cv2.addWeighted(image, .55, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), .45, 0))
    rows = []
    regions = [(i['class_code'], i['confidence'], i['box']) for i in report['review']['items']]
    if not regions:
        regions = [('FRAME', None, [0, 0, w, h])]
    for code, confidence_value, box in regions:
        x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
        x2, y2 = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
        values = [float(m[y1:y2, x1:x2].mean()) for m in maps.values()]
        shares = [round(100 * v / max(sum(values), 1e-8), 2) for v in values]
        rows.append([code, None if confidence_value is None else round(confidence_value, 3), *shares])
    return render_boxes(image, report), *overlays, {'kind': 'features', 'prediction': report,
            'rows': rows, 'layers': dict(zip(('P3', 'P4', 'P5'), map(int, indices))),
            'brightness': float(brightness), 'blur_kernel': int(blur)}


def match_targets(items, targets):
    """Confidence-ordered, one-to-one, same-class IoU ≥ .5 diagnostic counts."""
    used, tp = set(), 0
    for item in sorted(items, key=lambda item: -item['confidence']):
        matches = [(box_iou(item['box'], target['xyxy']), index) for index, target in enumerate(targets)
                   if index not in used and item['class_code'] == target['class_name']]
        overlap, index = max(matches, default=(0., -1))
        if overlap >= .5:
            tp += 1
            used.add(index)
    return tp, len(items) - tp, len(targets) - tp


def country_check(model, confidence, device, brightness=1., blur=0, progress=None):
    catalog = cases()
    if not catalog:
        raise ValueError('The bundled diagnostic cases are missing.')
    results = []
    for index, item in enumerate(catalog):
        image, _ = load_case(item['id'])
        image = transform(image, brightness, blur)
        _, report = analyze_image(image, model, confidence, device)
        tp, fp, fn = match_targets(report['review']['items'], item['targets'])
        results.append({'id': item['id'], 'country': item['country'], 'tp': tp, 'fp': fp, 'fn': fn,
                        'report': report})
        if progress:
            progress((index + 1) / len(catalog))
    rows = []
    for country in ('Czech', 'India', 'United_States'):
        group = [r for r in results if r['country'] == country]
        tp, fp, fn = (sum(r[k] for r in group) for k in ('tp', 'fp', 'fn'))
        rows.append([country, len(group), tp, fp, fn, round(tp / (tp + fp), 4) if tp + fp else 0.,
                     round(tp / (tp + fn), 4) if tp + fn else 0.])
    return {'kind': 'countries', 'rows': rows, 'cases': results, 'model': model,
            'confidence_threshold': float(confidence), 'brightness': float(brightness), 'blur_kernel': int(blur),
            'scope': 'fixed diagnostic subset; stratified by class; not overall accuracy or mAP'}


def benchmark(image, model_names, devices, confidence, repeats, brightness=1., blur=0):
    image = transform(image, brightness, blur)
    if not model_names or any(name not in MODELS for name in model_names) or not devices:
        raise ValueError('Select at least one registered model and device.')
    if isinstance(repeats, bool) or int(repeats) != repeats or not 3 <= int(repeats) <= 20:
        raise ValueError('Use 3–20 measured repetitions.')
    rows, traces = [], []
    with INFERENCE_LOCK:
        for device in devices:
            for model in model_names:
                for _ in range(2):
                    analyze_image(image, model, confidence, device)
                calls, full, counts = [], [], []
                for _ in range(int(repeats)):
                    started = time.perf_counter()
                    _, report = analyze_image(image, model, confidence, device)
                    full.append((time.perf_counter() - started) * 1000)
                    calls.append(report['predict_call_ms_including_first_call'])
                    counts.append(report['review']['total_distress_count'])
                rows.append([model, report['device'], int(repeats), round(float(np.mean(calls)), 2),
                             round(float(np.percentile(calls, 95)), 2), round(float(np.mean(full)), 2),
                             round(1000 / float(np.mean(full)), 2)])
                traces.append({'model': model, 'device': report['device'], 'predict_call_ms': calls,
                               'full_call_ms': full, 'detections': counts, 'model_sha256': report['model_sha256']})
    return {'kind': 'benchmark', 'rows': rows, 'traces': traces, 'warmup_calls_per_model_device': 2,
            'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'confidence_threshold': float(confidence),
            'torch_version': torch.__version__, 'settings': {'imgsz': 640, 'rect': False, 'nms_iou': .7, 'max_det': 300},
            'scope': 'current-machine sequential single-image calls; derived FPS is not a video throughput guarantee'}
