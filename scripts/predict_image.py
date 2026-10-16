"""Run an existing road-damage checkpoint on one local image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
LOCAL_SETTINGS = ROOT / '.local/ultralytics'
LOCAL_SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(LOCAL_SETTINGS))
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original road image')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/baseline_best.pt')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', default='cpu', help='cpu, mps, or a CUDA index such as 0')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU, distinct from GT matching IoU')
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1 or args.imgsz <= 0:
        parser.error('conf must be in [0, 1], iou in (0, 1], and imgsz positive')
    model_path, source, output = [p.expanduser().resolve() for p in (args.model, args.source, args.output)]
    for path in (model_path, source):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    if output.exists():
        parser.error(f'Choose a new output directory: {output}')
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f'Cannot decode image: {source}')

    model = YOLO(str(model_path))
    expected = {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}
    if model.names != expected:
        raise ValueError(f'Unexpected road-damage class mapping: {model.names}')
    started = time.perf_counter()
    result = model.predict(image, device=args.device, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
    elapsed = (time.perf_counter() - started) * 1000
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({'class_id': class_id, 'class_name': model.names[class_id],
                           'confidence': float(box.conf.item()), 'xyxy': box.xyxy[0].cpu().tolist()})
    output.mkdir(parents=True, exist_ok=False)
    ok, encoded = cv2.imencode('.jpg', result.plot())
    if not ok:
        raise RuntimeError('Could not encode prediction image')
    encoded.tofile(output / 'prediction.jpg')
    report = {'source': str(source), 'model': str(model_path),
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'device': str(result.boxes.data.device), 'imgsz': args.imgsz, 'conf': args.conf, 'nms_iou': args.iou,
              'class_mapping': model.names,
              'image_size': {'width': image.shape[1], 'height': image.shape[0]},
              'predict_call_ms_including_first_call': round(elapsed, 2),
              'detections': detections,
              'note': 'No detection does not establish that a road is undamaged.'}
    (output / 'prediction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'detections': len(detections), 'device': report['device']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
