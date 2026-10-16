"""Portable evaluation of existing checkpoints, without copying/retraining weights."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from importlib import metadata

import yaml
from rdd_quality import CLASS_NAMES, validation_class_rows

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def resolve_dataset(data_file, split):
    data_file = Path(data_file).expanduser().resolve()
    data = yaml.safe_load(data_file.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('Dataset YAML must be a mapping')
    names = data.get('names', {})
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if names != CLASS_NAMES:
        raise ValueError(f'Unexpected dataset class mapping: {names}')
    # Repository YAML roots are project-relative, not relative to global Ultralytics settings.
    root = Path(data.get('path') or data_file.parent).expanduser()
    root = root.resolve() if root.is_absolute() else (ROOT / root).resolve()
    resolved = dict(data)
    resolved['path'] = str(root)
    resolved['names'] = names
    for key in ('train', 'val', 'test'):
        value = data.get(key)
        if not value:
            if key in ('train', 'val', split):
                raise ValueError(f'Dataset YAML is missing {key}')
            continue
        values = value if isinstance(value, list) else [value]
        paths = [Path(x).expanduser() for x in values]
        paths = [p.resolve() if p.is_absolute() else (root / p).resolve() for p in paths]
        resolved[key] = [str(p) for p in paths] if isinstance(value, list) else str(paths[0])
        if key in ('val', split):
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(f'Dataset {key} path does not exist: {path}')
    # Do not trigger implicit dataset downloads through a supplied YAML.
    resolved.pop('download', None)
    return data_file, resolved


def evaluate_checkpoint(model_path, data_file, output, split='test', device='cpu', batch=1,
                        imgsz=640, candidate_conf=0.001, nms_iou=0.7, purpose='evaluation'):
    if split not in ('val', 'test'):
        raise ValueError('Use val or test for this evaluation entry')
    if purpose not in ('smoke', 'evaluation'):
        raise ValueError('Invalid evaluation purpose')
    if not 0 <= candidate_conf <= 1 or not 0 < nms_iou <= 1 or batch < 1 or imgsz < 1:
        raise ValueError('Invalid evaluation parameters')
    model_path, output = Path(model_path).expanduser().resolve(), Path(output).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {model_path}')
    if output.exists():
        raise FileExistsError(f'Choose a new evaluation output directory: {output}')
    data_file, resolved = resolve_dataset(data_file, split)
    local_config = ROOT / '.local/ultralytics'
    local_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(local_config))
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.names != CLASS_NAMES:
        raise ValueError(f'Unexpected checkpoint class mapping: {model.names}')
    output.mkdir(parents=True, exist_ok=False)
    resolved_path = output / 'resolved_data.yaml'
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding='utf-8')
    metrics = model.val(data=str(resolved_path), split=split, device=device, batch=batch, imgsz=imgsz,
                        workers=0, conf=candidate_conf, iou=nms_iou, rect=False,
                        project=str(output), name='validation', exist_ok=False, plots=False, verbose=False)
    rows = validation_class_rows(metrics)
    overall = {'precision': _number(metrics.box.mp), 'recall': _number(metrics.box.mr),
               'map50': _number(metrics.box.map50), 'map50_95': _number(metrics.box.map),
               'support': sum(r['support'] or 0 for r in rows)}
    record = {'purpose': purpose, 'model': str(model_path), 'model_sha256': file_hash(model_path),
              'data_yaml': str(data_file), 'data_yaml_sha256': file_hash(data_file),
              'resolved_data_yaml_sha256': file_hash(resolved_path),
              'split': split, 'parameters': {'device': str(device), 'batch': batch, 'imgsz': imgsz,
                                           'candidate_conf': candidate_conf, 'nms_iou': nms_iou, 'rect': False},
              'python': sys.version, 'platform': platform.platform(),
              'versions': {p: metadata.version(p) for p in ('torch', 'torchvision', 'ultralytics')},
              'overall': overall, 'per_class': rows,
              'pr_definition': 'Ultralytics max-F1 summary; not fixed deployment-threshold TP/FP/FN',
              'independence': 'Not established by this command; verify split membership and model provenance',
              'artifact_directory': str(output / 'validation')}
    (output / 'metrics.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'per_class.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['class_id', 'class_name', 'support', 'precision', 'recall', 'ap50', 'ap50_95'])
        writer.writeheader()
        writer.writerows(rows)
    return record
