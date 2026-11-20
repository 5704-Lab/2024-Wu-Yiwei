"""Frozen-source overlapping-view evidence; never an image editing mask.

Views are selected on source-only presence/attribute evidence, then fixed for
all outputs. This tests an observed head-only shortcut; it is not a general
object detector, segmentation module, or independent semantic ground truth.
"""
import json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel,AutoProcessor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,seed_setup,write,panel

class ViewEvidence:
 def __init__(self,model,processor,pair):
  self.model=model;self.processor=processor;self.device=next(model.parameters()).device
  if pair=='bear__00':texts=['A close-up photo of bear fur.','Green grass and plants.','Water in a stream.','Rocks and soil.','Black bear fur.','Brown bear fur.']
  else:texts=['A close-up photo of owl feathers.','Green grass.','Rocks and soil.','Blurred vegetation.','White owl feathers.','Brown owl feathers.']
  self.texts=texts;tokens=processor(text=texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(self.device)
  with torch.no_grad():e=model.get_text_features(**tokens);self.e=F.normalize(e,dim=-1)
  self.boxes=[(x,y,x+224,y+224) for y in [0,144,288] for x in [0,144,288]];self.selected=None

 def pixels(self,im,selected=False):
  boxes=self.boxes if not selected else [self.boxes[i] for i in self.selected]
  return torch.cat([F.interpolate(im[:,:,y0:y1,x0:x1],(256,256),mode='bilinear',align_corners=False,antialias=True) for x0,y0,x1,y1 in boxes])

 def logits(self,im,selected=False):
  pixels=self.pixels(im,selected);f=self.model.get_image_features(pixel_values=(pixels-.5)/.5)
  return self.model.logit_scale.exp()*(F.normalize(f,dim=-1)@self.e.T)+self.model.logit_bias

 def choose(self,source):
  with torch.no_grad():logs=self.logits(source);presence=logs[:,0]-logs[:,1:4].max(1).values;attr=logs[:,4]-logs[:,5]
  eligible=[i for i in range(9) if float(attr[i])<-.1]
  self.selected=sorted(eligible,key=lambda i:float(presence[i]),reverse=True)[:3]
  return dict(selected=self.selected,boxes=self.boxes,source_presence=presence.cpu().tolist(),source_attribute=attr.cpu().tolist(),texts=self.texts)

 def facts(self,im):
  s=self.logits(im,True);return s[:,4]-s[:,5]

def image_tensor(path,device):
 a=np.array(Image.open(path).convert('RGB')).astype(np.float32)/255
 return torch.tensor(a,device=device).permute(2,0,1)[None]

def main():
 seed_setup();device=torch.device('cuda:0');directory=ROOT/'pretrained_models/siglip2-base-patch16-256'
 model=AutoModel.from_pretrained(directory,torch_dtype=torch.float32,local_files_only=True,attn_implementation='eager').eval().requires_grad_(False).to(device)
 processor=AutoProcessor.from_pretrained(directory,local_files_only=True);allrows=[];selectors={}
 for pair in ['bear__00','brown_owl__00']:
  source_path=ROOT/'experiments/joint_permission/outputs/analysis'/f'{pair}_source.png'
  source=image_tensor(source_path,device);v=ViewEvidence(model,processor,pair);selection=v.choose(source);selectors[pair]=selection
  cases={'SOURCE':source_path,'FlowEdit':ROOT/'experiments/phase_permission/outputs/full'/pair/'BASE/final.png',
   'old_exposure':ROOT/'experiments/joint_permission/outputs/achromatic_strong'/pair/'final.png',
   'material_pilot':HERE/'outputs/material_pilot'/pair/'final.png',
   'material_chroma':HERE/'outputs/material_full_chroma'/pair/'final.png'}
  for name,path in cases.items():
   if not path.exists():continue
   im=image_tensor(path,device)
   with torch.no_grad():f=v.facts(im)
   allrows.append(dict(pair=pair,case=name,facts=f.cpu().tolist(),minimum=float(f.min())))
   crops=[(f'view {i}',im[:,:,b[1]:b[3],b[0]:b[2]]) for i,b in [(i,v.boxes[i]) for i in v.selected]]
   panel(crops,HERE/'outputs/views'/f'{pair}_{name}.png',title=pair+' '+name,size=224)
 write(HERE/'outputs/views/results.json',dict(selectors=selectors,rows=allrows,
  caveat='source-only selection, same frozen scorer, nine fixed overlapping windows; limited diagnostic, not independent semantic validation'))
 print('VIEW_PROBE_COMPLETE',allrows,flush=True)

if __name__=='__main__':main()
