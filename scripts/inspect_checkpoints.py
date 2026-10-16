"""Recover saved training arguments and embedded history without training or changing weights."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,nargs='+',default=[ROOT/'models/baseline_best.pt',ROOT/'models/compare_yolo11s_best.pt'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    config=ROOT/'.local/ultralytics'
    config.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR',str(config))
    from ultralytics import YOLO
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for model_path in args.models:
        ckpt=YOLO(str(model_path.resolve())).ckpt
        history=ckpt.get('train_results') or {}
        if len({len(values) for values in history.values()})>1:
            raise ValueError('Embedded training history has inconsistent column lengths')
        if history:
            with (args.output/f'{model_path.stem}_history.csv').open('w',newline='',encoding='utf-8-sig') as handle:
                writer=csv.writer(handle)
                writer.writerow(history.keys())
                writer.writerows(zip(*history.values()))
        train_args={k:v for k,v in (ckpt.get('train_args') or {}).items() if k not in ('save_dir','data','project')}
        record={'model':model_path.name,'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
                'date':ckpt.get('date'),'version':ckpt.get('version'),'epoch_field':ckpt.get('epoch'),
                'saved_history_rows':len(next(iter(history.values()))) if history else 0,
                'saved_history_epochs':history.get('epoch',[]),'stopper_metadata':ckpt.get('stopper'),
                'saved_validation_metrics':ckpt.get('train_metrics'),'training_parameters':train_args,
                'limits':'Embedded history belongs to the saved checkpoint. It may end at a best epoch or be replaced during checkpoint stripping; it is not proof of the complete training duration or exact dataset membership.'}
        records.append(record)
    (args.output/'checkpoint_history.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps([{'model':x['model'],'saved_history_rows':x['saved_history_rows'],'epoch_field':x['epoch_field']} for x in records]))


if __name__=='__main__':
    main()
