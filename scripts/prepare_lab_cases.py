"""Bundle a small, source-verifiable diagnostic subset from the group's ZIPs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_demo_samples import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'demo/assets/lab_cases')
    args = parser.parse_args()
    # Reuse the deterministic source/test-manifest and annotation validation rules.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        prepare(args.archive_root, per_country=20, output=Path(temporary))
        catalog = json.loads((Path(temporary) / 'manifest.json').read_text(encoding='utf-8'))['cases']
        selected = []
        for country in ('Czech', 'India', 'United_States'):
            group = [item for item in catalog if item['country'] == country]
            chosen = {}
            preferred = next((item for item in group if item['id'] == 'Czech_001639'), None)
            if preferred:
                chosen[preferred['id']] = preferred
            for code in ('D00', 'D10', 'D20', 'D40', None):
                item = next((item for item in group if item['id'] not in chosen and
                             (code in item['classes'] if code else not item['targets'])), None)
                if item:
                    chosen[item['id']] = item
            for item in group:
                if len(chosen) >= 6:
                    break
                chosen[item['id']] = item
            selected.extend(list(chosen.values())[:6])
        args.output.mkdir(parents=True, exist_ok=True)
        for item in selected:
            raw = (Path(temporary) / item['image']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item['sha256']
            (args.output / item['image']).write_bytes(raw)
            with zipfile.ZipFile(args.archive_root / f"{item['country']}.zip") as archive:
                item['annotation_sha256'] = hashlib.sha256(archive.read(item['xml_member'])).hexdigest()
        (args.output / 'manifest.json').write_text(json.dumps({
            'version': 'diagnostic-cases-v1', 'source': 'group-provided RDD2022 country ZIPs',
            'selection': '6 per country, target-class coverage where available; diagnostic, not representative evaluation',
            'source_splits_unchanged': True, 'cases': selected}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'bundled': len(selected), 'countries': {c: sum(i['country'] == c for i in selected)
                          for c in ('Czech', 'India', 'United_States')}}, indent=2))


if __name__ == '__main__':
    main()
