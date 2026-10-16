"""Compare saved image predictions with original VOC labels; no threshold tuning."""
import argparse
import json
from pathlib import Path
import zipfile
from rdd_quality import read_target_annotation, match_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--annotations-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--match-iou', type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    reports = []
    with zipfile.ZipFile(args.annotations_zip) as archive:
        by_name = {}
        for name in archive.namelist():
            if name.endswith('.xml'):
                by_name.setdefault(Path(name).name, []).append(name)
        for path in args.predictions:
            prediction = json.loads(path.read_text(encoding='utf-8'))
            xml_name = Path(prediction['source']).stem + '.xml'
            matches = by_name.get(xml_name, [])
            if len(matches) != 1:
                raise ValueError(f'Expected one annotation for {xml_name}, found {len(matches)}')
            annotation = read_target_annotation(archive.read(matches[0]))
            shape = prediction['image_size']
            if (shape['width'], shape['height']) != (annotation['width'], annotation['height']):
                raise ValueError('Prediction and annotation use different image coordinates')
            quality = match_predictions(prediction['detections'], annotation['targets'], args.match_iou)
            reports.append({'prediction_file': str(path.resolve()), 'model': prediction['model'],
                            'source': prediction['source'], 'annotation_entry': matches[0],
                            'candidate_conf': prediction.get('conf'), 'nms_iou': prediction.get('nms_iou'),
                            'annotation': annotation, 'diagnostic': quality})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'diagnosis.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    for report in reports:
        d = report['diagnostic']
        print(json.dumps({'model': Path(report['model']).name, 'TP': d['TP'], 'FP': d['FP'], 'FN': d['FN']}))


if __name__ == '__main__':
    main()
