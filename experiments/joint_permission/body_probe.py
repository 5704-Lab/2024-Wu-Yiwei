"""Post hoc independent wording probes; same encoder, not independent evidence."""
import sys
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import CleanScorer,ROOT,write,seed_setup

def main():
 seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
 names=['An owl with white feathers.','An owl with brown feathers.',
 'An owl with a white feathered body.','An owl with a brown feathered body.',
 'An owl with white wings.','An owl with brown wings.']
 e,_=sc.encode(names);rows=[]
 for case in ['SOURCE','FlowEdit','achromatic_strong','achromatic_neutral']:
  if case=='SOURCE':p=ROOT/'experiments/stage1_full/inputs/brown_owl__00.pt'
  elif case=='FlowEdit':p=ROOT/'experiments/flowedit_frequency_trajectory/outputs/brown_owl__00/trajectory.pt'
  else:p=HERE/'outputs'/case/'brown_owl__00/trajectory.pt'
  if not p.exists():continue
  d=torch.load(p,weights_only=True);y=d['x' if case=='SOURCE' else 'final'].to(sc.device)
  with torch.no_grad():
   im=sc.image(y);s=sc.logits_image(torch.cat([im,im.flip(-1)]),e).mean(0).reshape(-1,2)
  rows.append(dict(case=case,contrasts=(s[:,0]-s[:,1]).cpu().tolist()))
 write(HERE/'outputs/analysis/body_probe.json',dict(texts=names,rows=rows,
   caveat='Post hoc diagnostic with same frozen scorer; wording diversity is not independent semantic evidence.'))

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
