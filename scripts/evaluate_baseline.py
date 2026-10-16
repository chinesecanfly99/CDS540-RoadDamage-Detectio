"""Evaluate an existing road-damage checkpoint; never depend on an old runs folder."""
import argparse
import json
from pathlib import Path
from evaluation_runtime import ROOT, evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/baseline_best.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'configs/data.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--candidate-conf', type=float, default=0.001)
    parser.add_argument('--nms-iou', type=float, default=0.7)
    parser.add_argument('--purpose', choices=['smoke', 'evaluation'], default='evaluation')
    parser.add_argument('--save-predictions', action='store_true', help='Cache post-NMS detections for validation-only threshold selection and fixed-threshold diagnostics')
    parser.add_argument('--rect', action='store_true', help='Historical rectangular validation padding; default is square-padding deployment protocol')
    args = parser.parse_args()
    try:
        result = evaluate_checkpoint(args.model, args.data, args.output, args.split, args.device, args.batch,
                                     args.imgsz, args.candidate_conf, args.nms_iou, args.purpose, args.save_predictions, args.rect)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({'purpose': result['purpose'], 'output': str(args.output), 'overall': result['overall']}))


if __name__ == '__main__':
    main()
