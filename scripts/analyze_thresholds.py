"""Select thresholds using validation only, then evaluate frozen thresholds from caches."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import yaml
from rdd_quality import CLASS_NAMES, match_predictions

GRID=(.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_label(path,prediction=False):
    items=[]
    if not path.exists():
        if prediction:
            return items
        raise FileNotFoundError(f'Missing ground-truth label: {path}')
    import math
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values=list(map(float,line.split()))
        if len(values)!=(6 if prediction else 5) or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid label line: {path}')
        cls,xc,yc,w,h=values[:5]
        if cls!=int(cls) or int(cls) not in CLASS_NAMES or w<=0 or h<=0:
            raise ValueError(f'Invalid label class/box: {path}')
        box=[xc-w/2,yc-h/2,xc+w/2,yc+h/2]
        if min(box)<-1e-5 or max(box)>1.00001:
            raise ValueError(f'Label outside normalized image: {path}')
        record={'class_name':CLASS_NAMES[int(cls)],'xyxy':[min(1.,max(0.,v)) for v in box]}
        if prediction:
            if not 0<=values[5]<=1:
                raise ValueError(f'Invalid confidence: {path}')
            record['confidence']=values[5]
        items.append(record)
    return items


def load_cache(metrics_path):
    metrics_path=Path(metrics_path).resolve()
    meta=json.loads(metrics_path.read_text(encoding='utf-8'))
    if meta.get('purpose')!='evaluation' or not meta.get('prediction_cache',{}).get('enabled'):
        raise ValueError('Use a completed full evaluation with prediction caching')
    config_path=metrics_path.parent/'resolved_data.yaml'
    if hash_file(config_path)!=meta['resolved_data_yaml_sha256']:
        raise ValueError('Resolved dataset YAML changed')
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    image_dir=Path(config[meta['split']])
    if not image_dir.is_dir() or image_dir.parent.name!='images':
        raise ValueError('Expected a generated dataset image split directory')
    dataset=image_dir.parent.parent
    preparation=json.loads((dataset/'preparation.json').read_text(encoding='utf-8'))
    if preparation['dataset_fingerprint']!=meta.get('dataset_fingerprint'):
        raise ValueError('Dataset fingerprint does not match completed evaluation')
    label_dir=dataset/'labels'/image_dir.name
    prediction_dir=Path(meta['prediction_cache']['directory'])
    rows=[]
    for image in sorted(image_dir.glob('*.jpg')):
        target_path=label_dir/(image.stem+'.txt')
        rows.append({'image_id':image.name,'targets':parse_label(target_path),
                     'predictions':parse_label(prediction_dir/(image.stem+'.txt'),True)})
    if len(rows)!=meta['images_expected']:
        raise ValueError('Evaluated image count changed')
    counts=Counter(x['class_name'] for row in rows for x in row['targets'])
    if any(counts[r['class_name']]!=r['support'] for r in meta['per_class']):
        raise ValueError('Ground truth support differs from completed evaluation')
    meta['cache_metrics_path']=str(metrics_path)
    meta['cache_metrics_sha256']=hash_file(metrics_path)
    return meta,rows


def summarize(rows,threshold,exclude=()):
    counts={name:Counter({'TP':0,'FP':0,'FN':0,'support':0}) for name in CLASS_NAMES.values()}
    background,background_fp,images=0,0,0
    cases=[]
    for row in rows:
        if row['image_id'] in exclude:
            continue
        images+=1
        predictions=[x for x in row['predictions'] if x['confidence']>=threshold]
        match=match_predictions(predictions,row['targets'],.5)
        if not row['targets']:
            background+=1
            background_fp+=bool(predictions)
        for cls in match['by_class']:
            for key in ('TP','FP','FN','support'):
                counts[cls['class_name']][key]+=cls[key]
        if match['FP'] or match['FN']:
            cases.append({'image_id':row['image_id'],'TP':match['TP'],'FP':match['FP'],'FN':match['FN']})
    per_class=[]
    for name,c in counts.items():
        tp,fp,fn=c['TP'],c['FP'],c['FN']
        per_class.append({'class_name':name,**dict(c),'precision':tp/(tp+fp) if tp+fp else None,
                          'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    total={k:sum(r.get(k,0) for r in per_class) for k in ('TP','FP','FN','support')}
    tp,fp,fn=total['TP'],total['FP'],total['FN']
    supported=[r['f1'] for r in per_class if r['support']>0]
    return {'threshold':threshold,'match_iou':.5,'images':images,**total,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'macro_f1':sum(supported)/len(supported) if supported else None,'per_class':per_class,
            'background_images':background,'background_images_with_predictions':background_fp,
            'background_false_alarm_image_rate':background_fp/background if background else None,
            'failure_images':sorted(cases,key=lambda r:-(r['FP']+r['FN']))}


def select_from_sweep(sweep):
    if not sweep or any(row['macro_f1'] is None for row in sweep):
        raise ValueError('Cannot select threshold without validation targets')
    return max(sweep,key=lambda row:(row['macro_f1'],row['threshold']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('select','report'):
        command=sub.add_parser(name)
        command.add_argument('--evaluations',type=Path,nargs='+',required=True)
        command.add_argument('--output',type=Path,required=True)
        command.add_argument('--exclude-images',nargs='*',default=[],help='Previously audited image IDs; selection excludes only validation members, report adds sensitivity counts')
        if name=='report':
            command.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    if args.command=='select':
        selected=[]
        for path in args.evaluations:
            meta,rows=load_cache(path)
            if meta['split']!='val':
                raise ValueError('Threshold selection accepts validation evaluations only')
            if meta['parameters']['candidate_conf']>min(GRID):
                raise ValueError('Prediction cache cannot support the threshold grid')
            sweep=[summarize(rows,threshold,args.exclude_images) for threshold in GRID]
            for result in sweep:
                result.pop('failure_images')
            chosen=select_from_sweep(sweep)
            selected.append({'model_sha256':meta['model_sha256'],'model':Path(meta['model']).name,
                             'dataset_fingerprint':meta['dataset_fingerprint'],'parameters':meta['parameters'],
                             'validation_metrics':meta['cache_metrics_path'],'validation_metrics_sha256':meta['cache_metrics_sha256'],
                             'threshold':chosen['threshold'],'chosen_validation_result':chosen,'sweep':sweep})
        result={'objective':'maximum macro F1 across supported target classes; ties prefer larger threshold',
                'selection_split':'val','validation_exclusions':args.exclude_images,'threshold_grid':GRID,'match_iou':.5,'models':selected,
                'limits':'Threshold tuning does not improve AP or prove physical severity. Cached normalized TXT coordinates use finite decimal precision. Keep test and observed US holdout out of selection.'}
        filename='threshold_selection.json'
    else:
        selection=json.loads(args.selection.read_text(encoding='utf-8'))
        if selection['selection_split']!='val':
            raise ValueError('Invalid frozen threshold source')
        models={x['model_sha256']:x for x in selection['models']}
        records=[]
        for path in args.evaluations:
            meta,rows=load_cache(path)
            if meta['split']!='test':
                raise ValueError('Frozen test report requires test split')
            profile=models[meta['model_sha256']]
            if profile['dataset_fingerprint']!=meta['dataset_fingerprint'] or profile['parameters']!=meta['parameters']:
                raise ValueError('Test and validation protocols differ')
            tuned=summarize(rows,profile['threshold'])
            default=summarize(rows,.25)
            record={'model':Path(meta['model']).name,'model_sha256':meta['model_sha256'],
                    'metrics_path':str(path.resolve()),'metrics_sha256':hash_file(path),
                    'dataset_fingerprint':meta['dataset_fingerprint'], 'ap_result':meta['overall'],
                    'at_default_025':default,'at_val_selected_threshold':tuned}
            # Report sensitivity to the previously established exact duplicate, without silently changing the set.
            if any(row['image_id']=='United_States_003917.jpg' for row in rows):
                record['exact_duplicate_sensitivity_fixed_threshold']=summarize(rows,profile['threshold'],{'United_States_003917.jpg'})
            applied=[r['image_id'] for r in rows if r['image_id'] in args.exclude_images]
            if applied:
                record['audited_near_duplicate_exclusions']=applied
                record['near_duplicate_sensitivity_fixed_threshold']=summarize(rows,profile['threshold'],applied)
            records.append(record)
        result={'frozen_selection':str(args.selection.resolve()),'selection_sha256':hash_file(args.selection),
                'matching':'confidence-ordered, same-class, one-to-one IoU >= 0.5; differs from max-F1 AP summary',
                'records':records}
        filename='fixed_threshold_report.json'
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'command':args.command,'output':str(args.output/filename)}))


if __name__=='__main__':
    main()
