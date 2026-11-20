"""Exploratory typed contrasts using ONLY the existing frozen SigLIP2.

Fixed verbal hypotheses for head/body/scene/object roles provide an explicit
alternative to untyped feature distance. No learned readout or source pose label.
Vocabulary/task mapping is supplied by this restricted protocol, NOT a general
free-text parser. Fresh dataset_v3 is evaluated only after thresholds are locked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModel,AutoProcessor

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
from analyze import threshold,summarize,LEGAL,ERRORS
sys.path.insert(0,str(ROOT/'experiments/phase_permission'))
from common import CleanScorer,seed_setup,write

HEAD=['turned to the right','turned to the left','turned toward the camera','turned away from the camera']
MORPH=['small rounded ears and a broad muzzle','larger rounded ears and a more slender muzzle']


def templates():
    records=[]
    def add(group,label,nuisance,text):records.append(dict(group=group,label=label,nuisance=nuisance,text=text))
    for color in ['brown','black','']:
        for morph in MORPH:
            for j,pose in enumerate(HEAD):
                add('head',j,[color,morph],f'Two clay bears on a gray table. The {color} bear on the left has its head {pose}, with {morph}.')
    for color in ['brown','black']:
        for j,morph in enumerate(MORPH):
            add('morph',j,[color],f'Two clay bears on a gray table. The {color} bear on the left has {morph}. The bear on the right is brown.')
    for j,color in enumerate(['brown','black']):
        for morph in MORPH:
            add('left_color',j,[morph],f'Two clay bears on a gray table. The bear on the left is {color} with {morph}. The bear on the right is brown.')
    for left in ['brown','black']:
        for j,right in enumerate(['brown','black']):
            add('right_color',j,[left],f'Two clay bears on a gray table. The bear on the left is {left}. The bear on the right is {right}.')
    for color in ['brown','black']:
        for j,pose in enumerate(['shown from the side','facing the camera','facing away from the camera']):
            add('body',j,[color],f'Two clay bears on a gray table. The {color} bear on the left has its body {pose}.')
        for j,bg in enumerate(['blue','pink','green']):
            add('background',j,[color],f'Two clay bears on a gray table in front of a {bg} wall. The bear on the left is {color}.')
    return records


def readout(feature,cache):
    scores=(cache['alpha']*(feature.double()@cache['text'].double().T)).flatten()
    out={}
    for group in ['head','morph','left_color','right_color','body','background']:
        labels=sorted(set(r['label'] for r in cache['records'] if r['group']==group))
        v=torch.stack([scores[[i for i,r in enumerate(cache['records']) if r['group']==group and r['label']==label]].mean() for label in labels])
        v=v-v.mean();out[group]=dict(logits=v.tolist(),probabilities=torch.softmax(v/2.,dim=0).tolist())
    return out


def energy(a,b,task):
    # Semantic factor selection, NOT a spatial edit gate.
    keep=['head','body','background','right_color']
    if task=='color':keep+=['morph']
    if task=='pose':keep=['body','background','right_color','morph','left_color']
    return float(sum(np.mean((np.array(a[g]['probabilities'])-np.array(b[g]['probabilities']))**2) for g in keep))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['cache','calibrate','fresh'],required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    out=HERE/'outputs/typed_v1';out.mkdir(exist_ok=True);cachefile=out/'text_bank.pt'
    if args.stage=='cache':
        if cachefile.exists():raise RuntimeError('Text bank already exists')
        rows=templates();device='cpu'
        model=AutoModel.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',local_files_only=True).eval().requires_grad_(False)
        proc=AutoProcessor.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',local_files_only=True)
        embeddings=[]
        for k in range(0,len(rows),16):
            tokens=proc(text=[r['text'] for r in rows[k:k+16]],padding='max_length',max_length=64,truncation=True,return_tensors='pt')
            with torch.no_grad():e=model.get_text_features(**tokens);e=e/e.norm(dim=-1,keepdim=True)
            embeddings.append(e.cpu())
        torch.save(dict(records=rows,text=torch.cat(embeddings),alpha=float(model.logit_scale.exp()),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),cachefile)
        write(out/'protocol.json',dict(scope='Restricted supplied semantic vocabulary, no general instruction decomposition; existing frozen SigLIP2 only',
            temperature=2.,fresh_holdout='all eight newly seeded sources of dataset_v3, irrespective of inherited split labels',
            task_maps={'shape':['head','body','background','right_color'],'color':['head','body','background','right_color','morph'],
                'pose':['body','background','right_color','morph','left_color']},templates=rows))
        print('TEXT_BANK_COMPLETE',len(rows),flush=True);return
    cache=torch.load(cachefile,weights_only=True)
    rows=[]
    if args.stage=='calibrate':
        for scene in range(4,8):
            folder=HERE/'outputs/features_v2'/f'scene{scene:02d}'
            source=torch.load(folder/'source.pt',weights_only=True)
            ref=readout(source['siglip']['global_feature'],cache)
            for file in folder.glob('*.pt'):
                item=torch.load(file,weights_only=True);obs=readout(item['siglip']['global_feature'],cache)
                rows.append(dict(scene=scene,variant=file.stem,energies={'typed_shape':energy(obs,ref,'shape')},observation=obs))
        limit=threshold(rows,'typed_shape')
        write(out/'threshold.json',dict(threshold=limit,source='dataset_v2 calibration sources 4-7; fixed before fresh v3 evaluation'))
        write(out/'calibration.json',dict(rows=rows,summary=summarize(rows,'typed_shape',limit)))
        print('TYPED_CALIBRATED',limit,flush=True)
    else:
        # Fresh pixels are encoded and decoded to a genuine current clean y.
        seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
        manifest=json.loads((HERE/'outputs/dataset_v3/manifest.json').read_text())
        from PIL import Image
        for entry in manifest['records']:
            source_obs=None;local=[]
            for name,item in entry['variants'].items():
                image=torch.from_numpy(np.array(Image.open(HERE/item['file']).convert('RGB')).copy()).permute(2,0,1)[None].float().to(sc.device)/255
                with torch.no_grad():
                    raw=sc.vae.encode(image*2-1).latent_dist.mode()
                    y=(raw-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor
                    decoded=sc.image(y);f=sc.model.get_image_features(pixel_values=sc.preprocess(decoded));f=f/f.norm(dim=-1,keepdim=True)
                obs=readout(f.cpu(),cache)
                if name=='source':source_obs=obs
                local.append(dict(scene=entry['scene'],variant=name,energies={'typed_shape':energy(obs,source_obs,'shape')},observation=obs))
            rows.extend(local);print('TYPED_FRESH',entry['scene'],flush=True)
        limit=json.loads((out/'threshold.json').read_text())['threshold']
        byscene={}
        for r in rows:byscene.setdefault(r['scene'],{})[r['variant']]=r
        def contrast(r,g):
            v=r['observation'][g]['logits'];return v[1]-v[0]
        completion=dict(morph_valid_above_color_only=sum(contrast(s['valid'],'morph')>contrast(s['color_only'],'morph') for s in byscene.values()),
            black_valid_above_source=sum(contrast(s['valid'],'left_color')>contrast(s['source'],'left_color') for s in byscene.values()),
            head_away_increases=sum(s['wrong_head']['observation']['head']['probabilities'][3]>s['valid']['observation']['head']['probabilities'][3] for s in byscene.values()),sources=len(byscene))
        write(out/'fresh.json',dict(rows=rows,summary=summarize(rows,'typed_shape',limit),completion=completion))
        print('TYPED_FRESH_COMPLETE',summarize(rows,'typed_shape',limit),completion,flush=True)


if __name__=='__main__':main()
