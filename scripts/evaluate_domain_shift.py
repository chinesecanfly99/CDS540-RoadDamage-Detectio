"""Evaluate source and country-test data with the same checkpoint and explicit protocol."""
import argparse
import json
from pathlib import Path
from evaluation_runtime import ROOT, evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, nargs='+', default=[ROOT / 'models/baseline_best.pt', ROOT / 'models/compare_yolo11s_best.pt'])
    parser.add_argument('--source-data', type=Path, default=ROOT / 'configs/data.yaml')
    parser.add_argument('--target-data', type=Path, default=ROOT / 'configs/data_domain_test.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--candidate-conf', type=float, default=0.001)
    parser.add_argument('--nms-iou', type=float, default=0.7)
    parser.add_argument('--purpose', choices=['smoke', 'evaluation'], default='evaluation')
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--rect', action='store_true', help='Historical rectangular validation padding; default is square-padding deployment protocol')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    if len({p.stem for p in args.models}) != len(args.models):
        parser.error('Model filenames must have unique stems')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for model in args.models:
        pair = {}
        for domain, data in [('source', args.source_data), ('country_test', args.target_data)]:
            record = evaluate_checkpoint(model, data, args.output / model.stem / domain, 'test', args.device,
                                         args.batch, args.imgsz, args.candidate_conf, args.nms_iou, args.purpose, args.save_predictions, args.rect)
            pair[domain] = record
        if pair['source']['model_sha256'] != pair['country_test']['model_sha256']:
            raise ValueError('Checkpoint changed between evaluations')
        src, target = pair['source']['overall'], pair['country_test']['overall']
        pair['overall_delta'] = {key: {'absolute': target[key] - src[key],
                                      'relative_percent': 100 * (target[key] - src[key]) / src[key] if src[key] else None}
                                 for key in ('map50', 'map50_95', 'precision', 'recall')
                                 if src[key] is not None and target[key] is not None}
        records.append(pair)
    (args.output / 'domain_comparison.json').write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'purpose': args.purpose, 'models_evaluated': len(records), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
