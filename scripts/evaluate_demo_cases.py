"""Fixed-threshold case regression, not full-dataset AP or an unbiased benchmark."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.cases import CASE_DIR, read_cases
from demo.inference import MODELS, analyze_image
from scripts.rdd_quality import match_predictions
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', '0', 'mps'])
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--output', type=Path, default=ROOT / '.local/case-regression')
    args = parser.parse_args()
    cases = read_cases()
    if not cases:
        parser.error('请先执行 scripts/prepare_demo_samples.py 生成样例库。')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, reports = [], []
    for index, item in enumerate(cases):
        with Image.open(CASE_DIR / item['image']) as image:
            image = np.array(image.convert('RGB'))
        for model in MODELS:
            _, report = analyze_image(image, model, args.confidence, args.device)
            predictions = [{'class_name': value['class_code'], 'confidence': value['confidence'], 'xyxy': value['box']}
                           for value in report['review']['items']]
            matching = match_predictions(predictions, item['targets'], .5)
            rows.append({'case': item['id'], 'country': item['country'], 'split': item['split'],
                         'model': model, 'device': report['device'], 'detections': len(predictions),
                         'TP': matching['TP'], 'FP': matching['FP'], 'FN': matching['FN'],
                         'priority': report['review']['priority_level']})
            reports.append({'case': item['id'], 'source_sha256': item['sha256'],
                            'prediction': report, 'diagnostic_matching': matching})
        print(f"{index + 1}/{len(cases)} {item['id']} passed", flush=True)
    with (args.output / 'cases.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'cases': len(cases), 'predictions': len(rows),
               'note': 'Stratified demonstration cases; IoU=.5 diagnostic matching, not mAP or representative accuracy.',
               'models': {model: {key: sum(row[key] for row in rows if row['model'] == model)
                                  for key in ('detections', 'TP', 'FP', 'FN')} for model in MODELS}}
    (args.output / 'reports.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
