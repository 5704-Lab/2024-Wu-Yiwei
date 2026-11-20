import sys,math
from pathlib import Path
import torch,numpy as np
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import source_rgb,seed_setup,write,warp,PAIRS
from geometry import Geometry

seed_setup();rows=[]
for pair in PAIRS+['flowers__00','corgi__00','penguins__00','dalmatian__00']:
 src,_=source_rgb(pair,dtype=torch.float64)
 src=torch.nn.functional.interpolate(src,(128,128),mode='bilinear',align_corners=False,antialias=True)
 obs=Geometry(src)
 xx=(torch.arange(128,dtype=src.dtype)+.5)/128
 cases=dict(identity=src,uniform_gain=src*.65,channel_gain=src*torch.tensor([.55,.85,1.05])[None,:,None,None],
            gamma=src.clamp_min(1e-5)**torch.tensor([.7,1.2,1.])[None,:,None,None],
            spatial_gain=src*torch.exp(.25*torch.cos(4*math.pi*xx))[None,None,None,:],
            shift_050=warp(src,tx=.5),shift_100=warp(src,tx=1.))
 for name,image in cases.items():
  with torch.no_grad():z=obs(image)
  rows.append(dict(pair=pair,case=name,mean=float(obs.loss(z,0)),tail=float(obs.loss(z,40))))
 write(HERE/'outputs/geometry_calibration.json',dict(rows=rows,scope='8 natural RGBs x 7 known controls; all source positions retained'))
for case in cases:
 print(case,np.median([r['mean'] for r in rows if r['case']==case]),np.median([r['tail'] for r in rows if r['case']==case]),flush=True)
