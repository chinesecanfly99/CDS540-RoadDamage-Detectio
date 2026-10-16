"""Audit existing split membership, target labels and byte-identical leakage in raw ZIPs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile
from collections import Counter, defaultdict
from rdd_quality import read_target_annotation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-root', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT / 'data/manifests')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    membership, overlaps = {}, []
    stats = {split: Counter() for split in ('train', 'val', 'test', 'domain_test')}
    for split in stats:
        with (args.manifests / (split + '.csv')).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                image_id = Path(row['image_path'].replace('\\', '/')).name
                if image_id in membership:
                    overlaps.append({'image': image_id, 'splits': [membership[image_id]['assigned_split'], split]})
                membership[image_id] = {**row, 'assigned_split': split}
    hashes, seen, unknown, changed_presence = defaultdict(list), set(), [], []
    skipped, country_stats = Counter(), {}
    fingerprint = hashlib.sha256()
    for country in ('Czech', 'India', 'United_States'):
        current = Counter()
        with zipfile.ZipFile(args.archives_root / (country + '.zip')) as archive:
            images = sorted(n for n in archive.namelist() if '/train/images/' in n and n.lower().endswith('.jpg'))
            for entry in images:
                image_id = Path(entry).name
                row = membership.get(image_id)
                if row is None:
                    unknown.append(image_id)
                    continue
                if row['country'] != country:
                    raise ValueError(f'Country mismatch for {image_id}')
                xml_entry = entry.replace('/images/', '/annotations/xmls/').rsplit('.', 1)[0] + '.xml'
                xml_bytes = archive.read(xml_entry)
                annotation = read_target_annotation(xml_bytes)
                image_hash = hashlib.sha256(archive.read(entry)).hexdigest()
                fingerprint.update((image_id + image_hash + hashlib.sha256(xml_bytes).hexdigest()).encode())
                split = row['assigned_split']
                hashes[image_hash].append({'image_id': image_id, 'split': split})
                seen.add(image_id)
                target_positive = bool(annotation['targets'])
                legacy_positive = row['has_object'].strip().lower() == 'true'
                if legacy_positive != target_positive:
                    changed_presence.append({'image_id': image_id, 'split': split})
                for bucket in (current, stats[split]):
                    bucket['images'] += 1
                    bucket['raw_boxes'] += annotation['raw_count']
                    bucket['target_boxes'] += len(annotation['targets'])
                    bucket['target_positive_images' if target_positive else 'no_target_annotation_images'] += 1
                    for target in annotation['targets']:
                        bucket[target['class_name']] += 1
                skipped.update(annotation['skipped'])
        country_stats[country] = dict(current)
        print(json.dumps({'country': country, 'images': current['images']}, ensure_ascii=True), flush=True)
    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    cross_split = [items for items in duplicate_groups if len({r['split'] for r in items}) > 1]
    report = {'manifests': str(args.manifests.resolve()), 'archive_dataset_fingerprint': fingerprint.hexdigest(),
              'images_scanned': len(seen), 'manifest_images': len(membership), 'id_overlaps': overlaps,
              'manifest_images_missing_from_archives': sorted(set(membership) - seen),
              'raw_training_images_missing_from_manifests': unknown,
              'country_stats': country_stats, 'split_stats': {k: dict(v) for k, v in stats.items()},
              'skipped_raw_classes': dict(skipped), 'presence_definition_changed_images': changed_presence,
              'exact_duplicate_groups': duplicate_groups, 'cross_split_exact_duplicate_groups': cross_split,
              'near_duplicate_or_sequence_leakage': 'NOT_AUDITED; path/byte disjointness is insufficient',
              'image_decode_quality': 'NOT_AUDITED; images were decompressed and hashed, not visually reviewed'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'images': len(seen), 'id_overlaps': len(overlaps), 'exact_duplicate_groups': len(duplicate_groups),
               'cross_split_exact_duplicates': len(cross_split), 'target_presence_mismatches': len(changed_presence)}
    print(json.dumps(summary), flush=True)
    if overlaps or cross_split or unknown or set(membership) - seen:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
