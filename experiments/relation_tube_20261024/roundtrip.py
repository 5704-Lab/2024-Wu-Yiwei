"""Measure source-reference roundtrip error to inform a finite source tube."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,seed_setup,write
from run import Relation

def main():
 seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');rows=[]
 short=torch.load(ROOT/'experiments/paired_relation_validation/outputs/short_condition_encodings.pt',weights_only=True)['short']
 for pair in ['bear__00','brown_owl__00']:
  d=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=d['x'].to(sc.device)
  e,_=sc.encode([d['record']['target_prompt'],d['record']['source_prompt'],short[pair][1],short[pair][0]])
  def encode(im):
   z=sc.vae.encode(im*2-1).latent_dist.mode();return (z-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor
  with torch.no_grad():
   source=sc.image(x);anchor=encode(source)
   rel=Relation(SimpleNamespace(sc=sc),dict(source=source,x=x,e=e),'SHARED_FACT')
   for case,param in [('identity',0.),('dark_gain',-.5),('light_gain',.5),('blue_gain',.4)]:
    theta=torch.zeros(rel.dim,device=sc.device)
    if case=='blue_gain':theta[rel.app.count*2]=param
    else:theta[::rel.app.count][:3]=param
    ref=rel.app(theta*rel.bound).clamp(0,1);y=x+encode(ref)-anchor;actual=sc.image(y)
    error=float((rel.observe(actual)-rel.observe(ref)).norm())
    rows.append(dict(pair=pair,case=case,joint_roundtrip_error=error,
     latent_source_zero_error=float((y-x).abs().max()) if case=='identity' else None))
 write(HERE/'outputs/calibration/roundtrip.json',dict(rows=rows,
  caveat='Optical source-reference consistency probes, not semantically legal edits or a universal bound.'))
 print(rows,flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
