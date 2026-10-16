"""
End-to-End Real-Time Road Damage Detection & Municipal Triage Demo
Course: CDS540 Computer Vision
Project: Automated Road Damage Detection for Smart City Maintenance

Step 12: Task 12.1 & Task 12.2
Demonstrates a production-ready edge dashcam pipeline:
  1. Ingests dashcam video frames or image stream.
  2. Runs real-time YOLO inference on GPU.
  3. Evaluates municipal prioritization rules (scripts.priority.RoadDamagePrioritizer).
  4. Overlays a real-time Telemetry Heads-Up Display (HUD):
     - Priority Alert Level (HIGH / MEDIUM / LOW / SOUND)
     - Damage Class Breakdown (D40 Pothole, D20 Alligator, D10 Transverse, D00 Longitudinal)
     - Real-Time Inference FPS & Latency
     - Actionable Maintenance Dispatch Order
  5. Renders and saves annotated demonstration video `demo/output.mp4`.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add root directory to python path to import priority module
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
from scripts.priority import RoadDamagePrioritizer, PRIORITY_LEVELS


def run_video_demo(model_path, input_source, output_video_path, max_frames=120, fps=15):
    """
    Processes video or image directory into annotated demonstration video.
    """
    print(f"\n=======================================================")
    print(f"Initializing End-to-End Municipal Road Inspection Demo")
    print(f"Model: {model_path}")
    print(f"Target Output: {output_video_path}")
    print(f"=======================================================")
    
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    prioritizer = RoadDamagePrioritizer(conf_threshold=0.25)
    
    # Check if input is a directory of images or a video file
    input_path = Path(input_source)
    frame_paths = []
    cap = None
    
    if input_path.is_dir():
        # Collect test images for continuous driving dashcam simulation
        all_imgs = sorted(list(input_path.glob("*.jpg")))
        # Select an illustrative sequence with a mix of defects
        frame_paths = all_imgs[:max_frames]
        print(f"[INFO] Ingesting {len(frame_paths)} sequential road dashcam frames.")
    elif input_path.is_file():
        cap = cv2.VideoCapture(str(input_path))
        print(f"[INFO] Ingesting input video stream: {input_source}")
    else:
        raise FileNotFoundError(f"Input source not found: {input_source}")

    # Determine resolution from first frame
    if frame_paths:
        first_frame = cv2.imread(str(frame_paths[0]))
        h, w = first_frame.shape[:2]
    else:
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read from video source")
        h, w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize VideoWriter with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    total_inference_time = 0.0
    frame_count = 0

    print("[INFO] Commencing real-time stream processing & HUD telemetry rendering...")

    try:
        while True:
            if frame_paths:
                if frame_count >= len(frame_paths):
                    break
                frame = cv2.imread(str(frame_paths[frame_count]))
            else:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

            t0 = time.perf_counter()
            # YOLO forward pass
            res = model.predict(frame, conf=0.25, verbose=False)[0]
            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000.0
            total_inference_time += inference_ms
            current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            # Compute priority triage
            triage = prioritizer.evaluate_detections(boxes, classes, confs, w, h)
            annotated = prioritizer.render_overlay(frame, triage)

            # Overlay real-time performance telemetry badge at bottom-right
            perf_text = f"RTX 5070 GPU | Latency: {inference_ms:.1f}ms | FPS: {current_fps:.1f} | Frame: {frame_count+1}/{max_frames if frame_paths else 'EOS'}"
            (pw, ph), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (w - pw - 20, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(annotated, perf_text, (w - pw - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            out_writer.write(annotated)
            frame_count += 1

            if frame_count % 20 == 0:
                print(f"  Processed {frame_count} frames | Avg Latency: {total_inference_time/frame_count:.2f} ms ({1000.0/(total_inference_time/frame_count):.1f} FPS)")

    finally:
        out_writer.release()
        if cap is not None:
            cap.release()

    avg_latency = total_inference_time / max(1, frame_count)
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print(f"\n[SUCCESS] Demo video rendered and exported to {output_video_path}")
    print(f"Summary: {frame_count} frames encoded at {fps} FPS ({frame_count/fps:.1f}s duration)")
    print(f"Average Inference Latency: {avg_latency:.2f} ms ({avg_fps:.1f} FPS)\n")


def main():
    os.chdir(root_dir)
    model_path = "models/compare_yolo11s_best.pt"
    input_source = "data/yolo/images/test"
    output_video = "demo/output.mp4"
    
    run_video_demo(
        model_path=model_path,
        input_source=input_source,
        output_video_path=output_video,
        max_frames=120, # ~8 seconds at 15 FPS
        fps=15
    )


if __name__ == '__main__':
    main()
