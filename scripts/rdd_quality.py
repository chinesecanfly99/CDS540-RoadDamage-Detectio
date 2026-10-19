"""Strict target annotations and fixed-threshold diagnostic matching (not AP)."""
import math
import xml.etree.ElementTree as ET

CLASS_NAMES = {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}


def read_target_annotation(xml_bytes):
    root = ET.fromstring(xml_bytes)
    width, height = int(root.findtext('size/width')), int(root.findtext('size/height'))
    if width <= 0 or height <= 0:
        raise ValueError('Annotation dimensions must be positive')
    targets, skipped = [], {}
    raw_count = 0
    for obj in root.findall('object'):
        raw_count += 1
        raw = (obj.findtext('name') or '').strip()
        if not raw:
            raise ValueError('Annotation object is missing its class name')
        name = 'D00' if raw == 'D0w0' else raw
        if name not in CLASS_NAMES.values():
            skipped[raw] = skipped.get(raw, 0) + 1
            continue
        coords = [float(obj.findtext('bndbox/' + key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not all(math.isfinite(x) for x in coords):
            raise ValueError('Annotation contains non-finite coordinates')
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'Invalid annotation box: {coords}')
        targets.append({'class_name': name, 'xyxy': coords})
    return {'width': width, 'height': height, 'targets': targets, 'raw_count': raw_count, 'skipped': skipped}


def box_iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def match_predictions(predictions, targets, match_iou=0.5):
    if not 0 < match_iou <= 1:
        raise ValueError('match_iou must be in (0, 1]')
    for item in [*predictions, *targets]:
        if item['class_name'] not in CLASS_NAMES.values():
            raise ValueError(f"Unexpected target class: {item['class_name']}")
        coords = item['xyxy']
        if len(coords) != 4 or not all(math.isfinite(float(x)) for x in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
            raise ValueError('Invalid matching box')
    for item in predictions:
        if not math.isfinite(item['confidence']) or not 0 <= item['confidence'] <= 1:
            raise ValueError('Invalid detection confidence')
    used, details = set(), []
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i]['confidence']):
        item = predictions[index]
        candidates = [(box_iou(item['xyxy'], gt['xyxy']), j) for j, gt in enumerate(targets)
                      if j not in used and item['class_name'] == gt['class_name']]
        overlap, target_index = max(candidates, default=(0., -1))
        matched = overlap >= match_iou
        if matched:
            used.add(target_index)
        details.append({'prediction_index': index, 'class_name': item['class_name'],
                        'confidence': item['confidence'], 'matched': matched,
                        'target_index': target_index if matched else None,
                        'best_unmatched_iou': overlap})
    rows = []
    for name in CLASS_NAMES.values():
        tp = sum(x['matched'] for x in details if x['class_name'] == name)
        count = sum(x['class_name'] == name for x in predictions)
        support = sum(x['class_name'] == name for x in targets)
        fp, fn = count - tp, support - tp
        rows.append({'class_name': name, 'support': support, 'TP': tp, 'FP': fp, 'FN': fn,
                     'precision': tp / count if count else None,
                     'recall': tp / support if support else None,
                     'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None})
    return {'TP': len(used), 'FP': len(predictions) - len(used), 'FN': len(targets) - len(used),
            'match_iou': match_iou, 'matching': 'confidence-ordered, same-class, one-to-one; diagnostic only, not mAP',
            'by_class': rows, 'predictions': details,
            'unmatched_target_indices': [j for j in range(len(targets)) if j not in used]}


def validation_class_rows(metrics):
    """Metric arrays are indexed by ap_class_index, not necessarily all four IDs."""
    index_by_class = {int(cid): i for i, cid in enumerate(metrics.box.ap_class_index)}
    counts = getattr(metrics, 'nt_per_class', None)
    rows = []
    for cid, name in CLASS_NAMES.items():
        index = index_by_class.get(cid)
        row = {'class_id': cid, 'class_name': name,
               'support': int(counts[cid]) if counts is not None and cid < len(counts) else None}
        for key, values in [('precision', metrics.box.p), ('recall', metrics.box.r),
                            ('ap50', metrics.box.ap50), ('ap50_95', metrics.box.ap)]:
            row[key] = float(values[index]) if index is not None else None
        rows.append(row)
    return rows
