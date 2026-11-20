"""Auxiliary SigLIP endpoint records; not an oracle editing-success metric."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel,AutoProcessor
from run_real import ROOT,OUT,PAIRS

ap=argparse.ArgumentParser();ap.add_argument('--refined',action='store_true');args=ap.parse_args()
if args.refined:OUT=OUT/'refined';PAIRS=['bear__00','cat_crown__00']
torch.set_num_threads(4);device=torch.device('cuda:1')
model=AutoModel.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
processor=AutoProcessor.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',local_files_only=True)
rows=[]
for pair in PAIRS:
 inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);rec=inp['record']
 filenames=dict(FlowEdit=OUT/f'{pair}_FlowEdit_512.png',SINGLE=OUT/f'{pair}_SINGLE_512.png',DUAL=OUT/f'{pair}_DUAL_512.png',SourceVAE=OUT/f'{pair}_SourceVAE_512.png',SourceRGB=ROOT/rec['source_image'])
 ref=np.asarray(Image.open(filenames['SourceVAE']).convert('RGB'),dtype=np.float64)/255
 base=np.asarray(Image.open(filenames['FlowEdit']).convert('RGB'),dtype=np.float64)/255
 for method,path in filenames.items():
  im=Image.open(path).convert('RGB');px=np.asarray(im.resize((512,512)),dtype=np.float64)/255
  inputs=processor(text=[rec['target_prompt'],rec['source_prompt']],images=im,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
  with torch.no_grad():logits=model(**inputs).logits_per_image[0].cpu().tolist()
  rows.append(dict(pair=pair,method=method,target_logit=logits[0],source_logit=logits[1],whole_image_mae_vs_source_vae=float(np.abs(px-ref).mean()),whole_image_mae_vs_flowedit=float(np.abs(px-base).mean()),path=str(path.relative_to(ROOT))))
(OUT/'endpoint_aux.json').write_text(json.dumps(dict(scope='PIL evaluation path; no reward optimization; global MAE is not non-edit-region consistency',rows=rows),indent=2)+'\n');print(json.dumps(rows,indent=2))
