"""Offline image-sequence/video demo using an existing road-damage checkpoint.

An image directory produces a slideshow; it is not evidence of continuous driving.
Timing covers the prediction call separately from the complete processing loop.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
LOCAL_SETTINGS = ROOT / '.local/ultralytics'
LOCAL_SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(LOCAL_SETTINGS))
from ultralytics import YOLO
sys.path.insert(0, str(ROOT))
from scripts.priority import RoadDamagePrioritizer


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def run_video_demo(model_path, input_source, output_video_path, max_frames=120, fps=None, device='cpu', conf=0.25, nms_iou=0.7):
    model_path = Path(model_path).expanduser().resolve()
    source = Path(input_source).expanduser().resolve()
    output = Path(output_video_path).expanduser().resolve()
    summary_path = output.with_suffix('.json')
    if not model_path.is_file():
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not source.exists():
        raise FileNotFoundError(f'Input not found: {source}. Pass --source with a local image directory or video.')
    if output.exists() or summary_path.exists():
        raise FileExistsError(f'Choose a new output filename: {output}')
    if max_frames < 1 or not 0 <= conf <= 1 or not 0 < nms_iou <= 1 or (fps is not None and (not np.isfinite(fps) or fps <= 0)):
        raise ValueError('Invalid frame count, FPS, confidence or NMS IoU')

    cap, writer = None, None
    predictions, priorities = [], []
    try:
        if source.is_dir():
            paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
            if not paths:
                raise ValueError(f'No supported images found in: {source}')
            paths = paths[:max_frames]
            first = read_image(paths[0])
            output_fps = fps or 15.0
            source_fps = None
            input_kind = 'image_slideshow'
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise ValueError(f'Cannot open video: {source}; for one image use scripts/predict_image.py')
            ok, first = cap.read()
            if not ok:
                raise ValueError(f'Cannot read first video frame: {source}')
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(source_fps) or source_fps <= 0:
                source_fps = None
            output_fps = fps or source_fps or 15.0
            paths = None
            input_kind = 'video'

        height, width = first.shape[:2]
        model = YOLO(str(model_path))
        prioritizer = RoadDamagePrioritizer(conf_threshold=conf)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create video output: {output}')

        count, resized = 0, 0
        started = time.perf_counter()
        while count < max_frames:
            if paths is not None:
                if count >= len(paths):
                    break
                frame = first if count == 0 else read_image(paths[count])
            elif count == 0:
                frame = first
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
                resized += 1

            tick = time.perf_counter()
            result = model.predict(frame, conf=conf, iou=nms_iou, imgsz=640, device=device, verbose=False)[0]
            predict_ms = (time.perf_counter() - tick) * 1000
            predictions.append(predict_ms)
            boxes = result.boxes
            triage = prioritizer.evaluate_detections(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                                                     boxes.conf.cpu().numpy(), width, height)
            annotated = prioritizer.render_overlay(frame, triage)
            # Actual backend, not a hard-coded GPU model. This is not full-pipeline latency.
            backend = str(boxes.data.device)
            badge = f'{backend} | predict: {predict_ms:.1f}ms | frame: {count + 1} | {input_kind}'
            cv2.rectangle(annotated, (0, height - 25), (width, height), (0, 0, 0), -1)
            cv2.putText(annotated, badge, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            writer.write(annotated)
            priorities.append({'frame': count, 'detections': triage['total_distress_count'],
                               'review_priority': triage['priority_level'], 'detection_status': triage['detection_status']})
            count += 1
        elapsed = time.perf_counter() - started
        writer.release()
        writer = None
        if count == 0:
            raise RuntimeError('No frames processed')
        summary = {'input': str(source), 'input_kind': input_kind, 'model': str(model_path),
                   'device': backend, 'candidate_conf': conf, 'nms_iou': nms_iou, 'rule_version': 'human-review-v1', 'frames_written': count, 'max_frames': max_frames,
                   'source_fps': source_fps, 'output_fps': output_fps,
                   'output_duration_s': count / output_fps, 'frames_resized_to_first_frame': resized,
                   'processing_seconds': round(elapsed, 3), 'processing_frames_per_second': round(count / elapsed, 2),
                   'mean_predict_ms_including_first_call': round(float(np.mean(predictions)), 2),
                   'frames': priorities,
                   'note': 'Offline processing. Image directories are slideshows. Review priorities are heuristic.'}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: summary[k] for k in ['input_kind', 'device', 'frames_written', 'processing_seconds']}, ensure_ascii=True))
        print(f'Video: {output}\nSummary: {summary_path}')
        return summary
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/yolo/images/test')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/compare_yolo11s_best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'demo/output.mp4')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU')
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--fps', type=float, default=None, help='Output FPS; defaults to source video FPS or 15 for images')
    args = parser.parse_args()
    try:
        run_video_demo(args.model, args.source, args.output, args.max_frames, args.fps, args.device, args.conf, args.iou)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
