"""Screen all pairs using two perceptual hashes and verify candidate similarity.

This is a conservative screening tool, not proof of sequence independence.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

POPCOUNT=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)


def hamming_distances(a,b):
    return POPCOUNT[np.bitwise_xor(a,b)].sum(axis=-1,dtype=np.uint16)


def structural_similarity(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    m1=cv2.GaussianBlur(a,(11,11),1.5)
    m2=cv2.GaussianBlur(b,(11,11),1.5)
    v1=cv2.GaussianBlur(a*a,(11,11),1.5)-m1*m1
    v2=cv2.GaussianBlur(b*b,(11,11),1.5)-m2*m2
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-m1*m2
    score=((2*m1*m2+6.5025)*(2*cov+58.5225))/((m1*m1+m2*m2+6.5025)*(v1+v2+58.5225))
    return float(score[5:-5,5:-5].mean())


def load_image(dataset,split,filename):
    path=dataset/'images'/split/filename
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phash-distance',type=int,default=8)
    parser.add_argument('--dhash-distance',type=int,default=10)
    parser.add_argument('--ssim-min',type=float,default=.85)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.output.exists():
        parser.error('Choose a new output directory')
    if not 0 <= args.phash_distance <= 64 or not 0 <= args.dhash_distance <= 64 or not 0 < args.ssim_min <= 1:
        parser.error('Invalid similarity thresholds')
    prep=json.loads((args.dataset/'preparation.json').read_text(encoding='utf-8'))
    features=np.load(args.dataset/'perceptual_features.npz',allow_pickle=False)
    phash,dhash=features['phash'],features['dhash']
    ids,splits,countries=features['ids'],features['splits'],features['countries']
    gray=features['gray64']
    args.output.mkdir(parents=True,exist_ok=False)
    pairs=[]
    hash_candidates=0
    n=len(ids)
    for start in range(0,n,128):
        distances=hamming_distances(phash[start:start+128,None,:],phash[None,:,:])
        for local,other in zip(*np.where(distances<=args.phash_distance)):
            index=start+int(local)
            other=int(other)
            if other<=index:
                continue
            dh=int(hamming_distances(dhash[index],dhash[other]))
            if dh>args.dhash_distance:
                continue
            hash_candidates+=1
            coarse=structural_similarity(gray[index],gray[other])
            if coarse<.75:
                continue
            a=load_image(args.dataset,str(splits[index]),str(ids[index]))
            b=load_image(args.dataset,str(splits[other]),str(ids[other]))
            if a.shape != b.shape:
                continue
            a=cv2.resize(a,(256,256),interpolation=cv2.INTER_AREA)
            b=cv2.resize(b,(256,256),interpolation=cv2.INTER_AREA)
            score=structural_similarity(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY))
            if score<args.ssim_min:
                continue
            mae=float(np.abs(a.astype(float)-b.astype(float)).mean()/255.)
            pairs.append({'image_a':str(ids[index]),'split_a':str(splits[index]),'country_a':str(countries[index]),
                          'image_b':str(ids[other]),'split_b':str(splits[other]),'country_b':str(countries[other]),
                          'phash_distance':int(distances[local,other]),'dhash_distance':dh,'ssim_256':score,
                          'color_mae_256':mae,'cross_split':bool(splits[index]!=splits[other]),
                          'strong_similarity_flag':bool(score>=.93 and mae<=.08)})
        if start%1280==0:
            print(json.dumps({'rows_compared':min(n,start+128),'total':n,'candidate_pairs':len(pairs)}),flush=True)
    pairs.sort(key=lambda x:(not x['cross_split'],-x['ssim_256']))
    report={'dataset_fingerprint':prep['dataset_fingerprint'],'images':n,'image_pairs_screened':n*(n-1)//2,
            'method':{'phash_bits':63,'phash_distance_max':args.phash_distance,'dhash_distance_max':args.dhash_distance,
                      'ssim_resolution':256,'ssim_min':args.ssim_min,'coarse_ssim64_min':.75,
                      'strong_flag_ssim_min':.93,'strong_flag_color_mae_max':.08},
            'hash_candidates':hash_candidates,'candidate_pairs':len(pairs),
            'cross_split_candidate_pairs':sum(x['cross_split'] for x in pairs),
            'strong_cross_split_pairs':sum(x['cross_split'] and x['strong_similarity_flag'] for x in pairs),
            'pairs':pairs,'status':'SCREENING_COMPLETE_MANUAL_REVIEW_REQUIRED',
            'limits':'Perceptual similarity is a candidate signal. Different viewpoints, crops, exposure changes and sequences may be missed. Filename adjacency does not prove a sequence. Do not infer a leakage-free split, automatically deduplicate or change membership from this report.'}
    (args.output/'near_duplicates.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if pairs:
        with (args.output/'candidate_pairs.csv').open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(pairs[0]))
            writer.writeheader(); writer.writerows(pairs)
    for page in range(min(3,(len(pairs)+5)//6)):
        card=Image.new('RGB',(960,6*208),'#f4f6f8')
        draw=ImageDraw.Draw(card)
        for row,pair in enumerate(pairs[page*6:page*6+6]):
            for col,key in enumerate(('a','b')):
                image=load_image(args.dataset,pair[f'split_{key}'],pair[f'image_{key}'])
                image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                image.thumbnail((460,170))
                card.paste(image,(col*480,row*208+32))
                draw.text((col*480+5,row*208+3),f"{pair[f'image_{key}']} [{pair[f'split_{key}']}]",fill='#152c40')
            draw.text((485,row*208+18),f"SSIM={pair['ssim_256']:.4f} pHash={pair['phash_distance']} cross={pair['cross_split']}",fill='#152c40')
        card.save(args.output/f'candidate_review_{page+1}.jpg',quality=92)
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
