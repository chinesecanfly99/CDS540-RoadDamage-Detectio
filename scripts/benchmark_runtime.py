"""Measure prediction calls and a named local image-processing task on the actual backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
config_dir = ROOT / '.local/ultralytics'
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(config_dir))
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
from scripts.priority import RoadDamagePrioritizer


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {'mean_ms': float(array.mean()), 'p50_ms': float(np.percentile(array, 50)),
            'p95_ms': float(np.percentile(array, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, nargs='+', default=[ROOT / 'models/baseline_best.pt', ROOT / 'models/compare_yolo11s_best.pt'])
    parser.add_argument('--source', type=Path, required=True, help='Local image directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.7)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.imgsz < 1 or not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error('Invalid run count or inference parameters')
    if not args.source.is_dir():
        parser.error('source must be an existing image directory')
    paths = sorted(p.resolve() for p in args.source.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    if not paths:
        parser.error('No supported images in source directory')
    if args.output.exists():
        parser.error('Choose a new output directory')
    if len({p.stem for p in args.models}) != len(args.models):
        parser.error('Models must have unique filename stems')
    for path in args.models:
        if not path.is_file():
            parser.error(f'Model missing: {path}')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in args.models:
        loaded_at = time.perf_counter()
        model = YOLO(str(path.resolve()))
        load_ms = (time.perf_counter() - loaded_at) * 1000
        review = RoadDamagePrioritizer(conf_threshold=args.conf)
        params = dict(device=args.device, conf=args.conf, iou=args.iou, imgsz=args.imgsz, rect=False, max_det=300, verbose=False)
        first = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f'Cannot decode: {paths[0]}')
        first_at = time.perf_counter()
        result = model.predict(first, **params)[0]
        result.boxes.data.cpu()  # Materialize results before stopping the wall clock.
        cold_ms = (time.perf_counter() - first_at) * 1000
        for _ in range(args.warmup):
            model.predict(first, **params)[0].boxes.data.cpu()
        prediction_times, task_times, per_run = [], [], []
        cards = args.output / path.stem
        cards.mkdir()
        for i in range(args.runs):
            source = paths[i % len(paths)]
            task_at = time.perf_counter()
            image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode: {source}')
            prediction_at = time.perf_counter()
            result = model.predict(image, **params)[0]
            boxes = result.boxes
            coords, classes, confidences = boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()
            prediction_ms = (time.perf_counter() - prediction_at) * 1000
            triage = review.evaluate_detections(coords, classes, confidences, image.shape[1], image.shape[0])
            overlay = review.render_overlay(image, triage)
            ok, encoded = cv2.imencode('.jpg', overlay)
            if not ok:
                raise RuntimeError('Failed to encode output image')
            encoded.tofile(cards / f'{i:04d}.jpg')
            task_ms = (time.perf_counter() - task_at) * 1000
            prediction_times.append(prediction_ms)
            task_times.append(task_ms)
            per_run.append({'source': str(source), 'predict_call_ms': prediction_ms, 'local_task_ms': task_ms})
        records.append({'model': str(path.resolve()), 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'checkpoint_bytes': path.stat().st_size, 'backend': str(boxes.data.device),
                        'parameters': {'imgsz': args.imgsz, 'conf': args.conf, 'nms_iou': args.iou, 'rect': False, 'max_det': 300, 'runs': args.runs, 'warmup': args.warmup},
                        'python': sys.version, 'torch': torch.__version__, 'model_load_ms': load_ms,
                        'first_predict_ms': cold_ms, 'predict_call': summarize(prediction_times),
                        'local_decode_predict_review_overlay_jpeg_export': summarize(task_times),
                        'local_task_frames_per_second': 1000 / float(np.mean(task_times)), 'runs': per_run,
                        'limits': 'Local still-image task; excludes model load, network, webpage, streaming and video encoding. Not pure tensor forward. Repeated inputs and small run counts are not deployment guarantees.'})
    (args.output / 'benchmark.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'models': len(records), 'backend': records[0]['backend']}))


if __name__ == '__main__':
    main()
