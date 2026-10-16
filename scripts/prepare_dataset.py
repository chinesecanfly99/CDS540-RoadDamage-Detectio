"""Prepare the unchanged manifest splits directly from original country ZIPs."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import yaml
from rdd_quality import CLASS_NAMES, read_target_annotation

ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}
SPLITS = ('train', 'val', 'test', 'domain_test')


def make_labels(annotation):
    width, height = annotation['width'], annotation['height']
    lines, maximum = [], 0.
    for obj in annotation['targets']:
        x1, y1, x2, y2 = obj['xyxy']
        values = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
        encoded = [float(f'{v:.6f}') for v in values]
        xc,yc,w,h = encoded
        restored = [(xc-w/2)*width, (yc-h/2)*height, (xc+w/2)*width, (yc+h/2)*height]
        error = max(abs(a-b) for a,b in zip(restored, obj['xyxy']))
        if error >= 1.:
            raise ValueError(f'Coordinate round-trip failed: {error}')
        maximum = max(maximum, error)
        lines.append(str(CLASS_IDS[obj['class_name']]) + ' ' + ' '.join(f'{v:.6f}' for v in values))
    return '\n'.join(lines) + ('\n' if lines else ''), maximum


def perceptual_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)
    frequencies = cv2.dct(small)[:8,:8].flatten()
    median = np.median(frequencies[1:])
    bits = frequencies > median
    bits[0] = False
    dh = cv2.resize(gray, (9,8), interpolation=cv2.INTER_AREA)
    return np.packbits(bits), np.packbits((dh[:,1:] > dh[:,:-1]).flatten()), cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA)


def prepare(archives_root, manifests, output):
    archives_root, manifests, output = [Path(p).resolve() for p in (archives_root, manifests, output)]
    if output.exists():
        raise FileExistsError(f'Choose a new dataset directory: {output}')
    rows = []
    for split in SPLITS:
        with (manifests / f'{split}.csv').open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                image_id = Path(row['image_path'].replace('\\','/')).name
                if Path(image_id).suffix.lower() != '.jpg':
                    raise ValueError(f'Unexpected image: {image_id}')
                rows.append({**row, 'assigned_split':split, 'image_id':image_id})
    if len(set(r['image_id'] for r in rows)) != len(rows):
        raise ValueError('Image IDs overlap across manifests')
    output.mkdir(parents=True,exist_ok=False)
    for split in SPLITS:
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    (output/'verification').mkdir()
    archives = {}
    indexes = {}
    stats = {s:Counter() for s in SPLITS}
    phashes, dhashes, grays, ids, memberships, countries = [],[],[],[],[],[]
    manifest_fingerprints = {s:hashlib.sha256((manifests/f'{s}.csv').read_bytes()).hexdigest() for s in SPLITS}
    content = hashlib.sha256()
    sample_counts = Counter()
    max_error, pixel_hashes, skipped = 0.,{},Counter()
    inventory_path = output/'inventory.jsonl'
    try:
        for country in ('Czech','India','United_States'):
            archive = archives[country] = zipfile.ZipFile(archives_root/f'{country}.zip')
            index = {}
            for name in archive.namelist():
                if '/train/images/' in name and name.lower().endswith('.jpg'):
                    filename = Path(name).name
                    if filename in index:
                        raise ValueError(f'Ambiguous raw image: {filename}')
                    index[filename] = name
            indexes[country] = index
        with inventory_path.open('w',encoding='utf-8') as inventory:
            for i,row in enumerate(rows):
                country,split,image_id = row['country'],row['assigned_split'],row['image_id']
                archive = archives[country]
                entry = indexes[country][image_id]
                xml_entry = entry.replace('/images/','/annotations/xmls/').rsplit('.',1)[0]+'.xml'
                data,xml = archive.read(entry),archive.read(xml_entry)
                annotation = read_target_annotation(xml)
                image = cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot decode image: {image_id}')
                h,w = image.shape[:2]
                if (w,h) != (annotation['width'],annotation['height']):
                    raise ValueError(f'Actual image size differs from XML: {image_id}')
                if (w,h) != (int(row['image_width']),int(row['image_height'])):
                    raise ValueError(f'Actual image size differs from manifest: {image_id}')
                label,error = make_labels(annotation)
                max_error = max(max_error,error)
                raw_hash = hashlib.sha256(data).hexdigest()
                label_hash = hashlib.sha256(label.encode()).hexdigest()
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                content.update((split+image_id+raw_hash+label_hash).encode())
                pixel_hashes.setdefault(pixel_hash,[]).append({'image_id':image_id,'split':split})
                (output/'images'/split/image_id).write_bytes(data)
                (output/'labels'/split/(Path(image_id).stem+'.txt')).write_text(label,encoding='utf-8')
                item = {'image_id':image_id,'split':split,'country':country,'width':w,'height':h,
                        'image_sha256':raw_hash,'label_sha256':label_hash,'pixel_sha256':pixel_hash,
                        'target_count':len(annotation['targets']),'has_target':bool(annotation['targets']),
                        'legacy_has_object':row['has_object'].lower()=='true','archive_entry':entry}
                inventory.write(json.dumps(item,ensure_ascii=False)+'\n')
                stat = stats[split]
                stat['images'] += 1
                stat['target_boxes'] += len(annotation['targets'])
                stat['positive' if item['has_target'] else 'no_target_annotation'] += 1
                skipped.update(annotation['skipped'])
                p,d,g = perceptual_features(image)
                phashes.append(p); dhashes.append(d); grays.append(g)
                ids.append(image_id); memberships.append(split); countries.append(country)
                group=(split,country,item['has_target'])
                if sample_counts[group] < 2:
                    card=image.copy()
                    for target in annotation['targets']:
                        x1,y1,x2,y2=map(int,target['xyxy'])
                        cv2.rectangle(card,(x1,y1),(x2,y2),(50,220,80),2)
                        cv2.putText(card,target['class_name'],(x1,max(14,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,220,80),1)
                    ok,encoded=cv2.imencode('.jpg',card)
                    if not ok:
                        raise RuntimeError('Cannot encode verification card')
                    encoded.tofile(output/'verification'/image_id)
                    sample_counts[group] += 1
                if (i+1)%500==0:
                    print(json.dumps({'prepared':i+1,'total':len(rows)}),flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    np.savez_compressed(output/'perceptual_features.npz', ids=np.array(ids), splits=np.array(memberships),
                        countries=np.array(countries), phash=np.array(phashes), dhash=np.array(dhashes), gray64=np.array(grays))
    configs={}
    for name,test in [('source','test'),('country','domain_test')]:
        config={'path':str(output),'train':'images/train','val':'images/val','test':f'images/{test}','names':CLASS_NAMES}
        path=output/f'{name}.yaml'
        path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
        configs[name]=str(path)
    report={'state':'COMPLETE','unchanged_split_membership':True,'images':len(rows),
            'manifest_sha256':manifest_fingerprints,'dataset_fingerprint':content.hexdigest(),
            'split_stats':{k:dict(v) for k,v in stats.items()},'skipped_classes':dict(skipped),
            'all_images_decoded_and_dimensions_verified':True,'all_target_boxes_roundtrip_verified':True,
            'maximum_roundtrip_error_pixels':max_error,'sample_cards':sum(sample_counts.values()),
            'decoded_pixel_duplicate_groups':[v for v in pixel_hashes.values() if len(v)>1],
            'configs':configs,'limits':'Successful decode and coordinate checks do not prove annotation completeness or split independence.'}
    (output/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-root',type=Path,required=True)
    parser.add_argument('--manifests',type=Path,default=ROOT/'data/manifests')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.archives_root,args.manifests,args.output)


if __name__=='__main__':
    main()
