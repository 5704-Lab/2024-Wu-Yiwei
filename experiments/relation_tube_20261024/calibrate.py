"""Source-local, known-intervention response calibration; not semantic truth."""
import sys,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,seed_setup,write,warp,panel
from run import Relation,CFG

def main():
 seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
 short=torch.load(ROOT/'experiments/paired_relation_validation/outputs/short_condition_encodings.pt',weights_only=True)['short']
 out=HERE/'outputs/calibration';rows=[];infos=[]
 for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
  start=time.monotonic();d=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
  x=d['x'].to(sc.device)
  with torch.no_grad():src=sc.image(x)
  e,_=sc.encode([d['record']['target_prompt'],d['record']['source_prompt'],short[pair][1],short[pair][0]])
  rel=Relation(SimpleNamespace(sc=sc),dict(source=src,x=x,e=e),'SHARED_FACT')
  theta=torch.zeros(rel.dim,device=sc.device);V=rel.jacobian(theta).double()
  with torch.no_grad():
   orig=rel.observe(src)[0].double();plen=rel.phase(src).numel();clen=plen
   probes=[(warp(src,tx=1),warp(src,tx=-1)),(warp(src,ty=1),warp(src,ty=-1)),(warp(src,angle=.002),warp(src,angle=-.002))]
   M=torch.stack([(rel.observe(a)[0]-rel.observe(b)[0]).double()*.5 for a,b in probes],1)
  normal=M[:plen].norm(dim=0).clamp_min(1e-12);M=M/normal
  xx=torch.linspace(0,1,src.shape[-1],device=sc.device)[None,None,None,:]
  controls=dict(identity=src,uniform_gain=src*.7,gamma=src.clamp_min(1e-5)**.8,
   channel_gain=(src*torch.tensor([.7,.9,1.1],device=sc.device)[None,:,None,None]).clamp(0,1),
   spatial_gain=(src*torch.exp(.2*torch.cos(2*np.pi*xx))).clamp(0,1),
   shift_half_at128=warp(src,tx=2),shift_one_at128=warp(src,tx=4),rotation=warp(src,angle=.01),
   blue_gain=(src*torch.tensor([.5,.7,1.6],device=sc.device)[None,:,None,None]).clamp(0,1),
   green_gain=(src*torch.tensor([.5,1.6,.6],device=sc.device)[None,:,None,None]).clamp(0,1))
  for name,end in [('PHASE',plen),('SHARED',plen+clen),('INDEPENDENT',plen+clen),('SHARED_FACT',len(orig))]:
   v=V[:end];m=M[:end]
   if name=='INDEPENDENT':
    v=torch.block_diag(V[:plen],V[plen:plen+clen])
   u,s,_=torch.linalg.svd(v,full_matrices=False)
   rank=int((s>s.max()*1e-5).sum());u=u[:,:rank]
   residual=m-u@(u.T@m);info=residual.T@residual
   infos.append(dict(pair=pair,mode=name,rank=rank,conditional_gram=info.cpu().tolist(),
    minimum_eigenvalue=float(torch.linalg.eigvalsh(info).min()),
    projected_geometry_fraction=(residual.norm(dim=0)/m.norm(dim=0)).cpu().tolist()))
   gram=(v.T@v).cpu().numpy()
   for case,im in controls.items():
    with torch.no_grad():delta=(rel.observe(im)[0].double()-orig)[:end]
    rhs=(v.T@delta).cpu().numpy()
    dim=v.shape[1]
    opt=minimize(lambda a:.5*a@gram@a-rhs@a,np.zeros(dim),jac=lambda a:gram@a-rhs,
      bounds=[(-1.,1.)]*dim,method='L-BFGS-B',options=dict(maxiter=100,ftol=1e-12,gtol=1e-8))
    a=torch.tensor(opt.x,device=sc.device,dtype=torch.float64);r=delta-v@a
    rows.append(dict(pair=pair,mode=name,case=case,raw_norm=float(delta.norm()),
     residual_norm=float(r.norm()),explained_energy=1-float(r.square().sum()/delta.square().sum()) if float(delta.norm())>1e-12 else None,
     solver_success=bool(opt.success),parameter_bound_fraction=float((a.abs()>.999).double().mean())))
  panel([(k,v) for k,v in controls.items() if k in ['identity','gamma','shift_one_at128','blue_gain','green_gain']],
    out/f'{pair}_controls.png',title=pair)
  write(out/'results.json',dict(rows=rows,geometry_information=infos,config=CFG,
   scope='source tangent, bounded coefficient fit; four source images, 10 constructed controls, 4 observation variants; photometric legality is optical, not automatic semantic permission'))
  print(pair,'calibrated',time.monotonic()-start,flush=True)
 print('CALIBRATION_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
