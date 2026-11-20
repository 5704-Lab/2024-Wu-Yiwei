"""CPU finite differences of the global colour law along fixed-chroma rays."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));from material import Material
sys.path.insert(0,str(HERE.parent/'phase_permission'));from common import write,seed_setup

def main():
 seed_setup();rows=[]
 for pair in ['bear__00','brown_owl__00']:
  src=HERE.parent/'joint_permission/outputs/analysis'/f'{pair}_source.png'
  im=torch.from_numpy(np.array(Image.open(src)).astype(np.float32)/255).permute(2,0,1)[None]
  payload=torch.load(HERE/'outputs/material_pilot'/pair/'state.pt',weights_only=True)
  for name,chroma in [('luminance_chroma',False),('chroma_only',True)]:
   mat=Material(im,pair,clusters=8 if chroma else 12,chroma_only=chroma)
   if not chroma:mat.centers=payload['centers'].cpu();theta=payload['theta'].float()
   else:theta=torch.linspace(.15,.85,mat.dim)
   with torch.no_grad():
    eps=.001;plus=mat.apply(im*(1+eps),theta);minus=mat.apply(im*(1-eps),theta)
    rgbw=torch.tensor([.299,.587,.114])[None,:,None,None]
    lum=(im*rgbw).sum(1);derivative=((plus-minus)*rgbw).sum(1)/(2*eps*lum.clamp_min(1e-6))
    valid=lum>.02
    rows.append(dict(pair=pair,law=name,negative_luminance_slope_fraction=float((derivative[valid]<-1e-3).float().mean()),
     derivative_min=float(derivative[valid].min()),derivative_median=float(derivative[valid].median()),
     note='old endpoint parameters vs arbitrary bounded parameters for new structural-law verification; not a matched edit-quality comparison'))
 write(HERE/'outputs/tone_probe.json',dict(rows=rows));print(rows)

if __name__=='__main__':main()
