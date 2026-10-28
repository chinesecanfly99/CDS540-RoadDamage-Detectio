"""Select diverse cases from existing test manifests without changing any split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rdd_quality import read_target_annotation

COUNTRIES = ('Czech', 'India', 'United_States')
CLASSES = ('D00', 'D10', 'D20', 'D40')


def select_cases(candidates, count):
    """Deterministic class coverage, then fill the country quota; no prediction selection."""
    chosen = {}
    ranked = sorted(candidates, key=lambda x: hashlib.sha256(('demo-v1:' + x['id']).encode()).hexdigest())
    for category in (*CLASSES, 'empty'):
        group = [item for item in ranked if (not item['classes'] if category == 'empty'
                                             else category in item['classes'])]
        for item in group[:max(1, count // 5)]:
            chosen[item['id']] = item
    for item in ranked:
        if len(chosen) >= count:
            break
        chosen[item['id']] = item
    return list(chosen.values())[:count]


def prepare(archive_root, per_country=20, output=None):
    output = output or ROOT / '.local/demo_samples'
    if not 5 <= per_country <= 100:
        raise ValueError('--per-country 应在 5 到 100 之间。')
    allowed = {}
    for split in ('test', 'domain_test'):
        with (ROOT / f'data/manifests/{split}.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                stem = Path(row['image_path'].replace('\\', '/')).stem
                allowed[(row['country'], stem)] = split
    output.mkdir(parents=True, exist_ok=True)
    selected, seen = [], set()
    for country in COUNTRIES:
        with zipfile.ZipFile(archive_root / f'{country}.zip') as archive:
            names = archive.namelist()
            images = {Path(name).stem: name for name in names if name.lower().endswith('.jpg')}
            candidates = []
            for name in names:
                stem = Path(name).stem
                if not name.lower().endswith('.xml') or (country, stem) not in allowed or stem not in images:
                    continue
                annotation = read_target_annotation(archive.read(name))
                candidates.append({'id': stem, 'country': country, 'split': allowed[(country, stem)],
                                   'classes': sorted({target['class_name'] for target in annotation['targets']}),
                                   'targets': annotation['targets'], 'image_member': images[stem],
                                   'xml_member': name, 'width': annotation['width'], 'height': annotation['height']})
            # Remove exact image duplicates before selecting so quotas stay reproducible.
            unique = []
            for item in candidates:
                raw = archive.read(item['image_member'])
                digest = hashlib.sha256(raw).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    item['sha256'] = digest
                    unique.append(item)
            for item in select_cases(unique, per_country):
                item['image'] = item['id'] + '.jpg'
                raw = archive.read(item['image_member'])
                (output / item['image']).write_bytes(raw)
                selected.append(item)
    manifest = {'version': 'demo-cases-v1', 'selection': 'deterministic, class-stratified; not representative evaluation',
                'data_version': 'original-split-v1; source manifests unchanged',
                'cases': selected}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'total': len(selected), 'by_country': {c: sum(x['country'] == c for x in selected) for c in COUNTRIES},
                      'class_coverage': {c: sum(c in x['classes'] for x in selected) for c in CLASSES},
                      'output': str(output)}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True, help='Folder containing the three country ZIPs')
    parser.add_argument('--per-country', type=int, default=20)
    args = parser.parse_args()
    prepare(args.archive_root, args.per_country)
