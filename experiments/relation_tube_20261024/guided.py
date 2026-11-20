"""Source-guided photometric reaction, with no region/attention editing mask.

RGB guided filter: He, Sun, Tang, ECCV 2010. The source is the fixed guidance;
native FlowEdit supplies a bounded log-colour reaction field. This hypothesis
is restricted to appearance edits; it is not a new object-binding solution.
"""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import SD3Transformer2DModel
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,Phase,seed_setup,write,save_image,panel
sys.path.insert(0,str(HERE.parent/'joint_permission'));from geometry import Geometry
sys.path.insert(0,str(ROOT/'vendor/flowedit'));from FlowEdit_utils import calc_v_sd3

class Guided:
 def __init__(self,source,radius,mode='guided',epsilon=.001):
  self.source=np.asarray(source,np.float32);self.r=radius;self.mode=mode
  self.mean=self.box(self.source)
  cov=np.stack([self.box(self.source[:,:,i,None]*self.source)-self.mean[:,:,i,None]*self.mean for i in range(3)],2)
  self.inverse=np.linalg.inv(cov+np.eye(3,dtype=np.float32)[None,None]*epsilon)
 def box(self,a):return cv2.boxFilter(a,-1,(2*self.r+1,2*self.r+1),normalize=True,borderType=cv2.BORDER_REFLECT_101)
 def __call__(self,p):
  meanp=self.box(p)
  if self.mode=='box':return meanp
  cov=np.stack([self.box(self.source[:,:,i,None]*p)-self.mean[:,:,i,None]*meanp for i in range(3)],2)
  a=np.einsum('hwik,hwko->hwio',self.inverse,cov)
  b=meanp-np.einsum('hwi,hwio->hwo',self.mean,a)
  ma=self.box(a.reshape(*a.shape[:2],9)).reshape(a.shape);mb=self.box(b)
  return np.einsum('hwi,hwio->hwo',self.source,ma)+mb
 def reaction(self,target):
  floor=.03;ratio=np.log((np.clip(target,0,1)+floor)/(self.source+floor)).clip(-2,2)
  field=self(ratio).clip(-2,2)
  return np.clip((self.source+floor)*np.exp(field)-floor,0,1)

def load(path):return np.array(Image.open(path).convert('RGB'),dtype=np.float32)/255
def tensor(a,device='cpu'):return torch.tensor(np.ascontiguousarray(a),device=device).permute(2,0,1)[None]

def offline():
 out=HERE/'outputs/guided_offline';rows=[]
 for pair in ['bear__00','brown_owl__00']:
  source=load(ROOT/'experiments/joint_permission/outputs/analysis'/f'{pair}_source.png')
  target=load(ROOT/'experiments/phase_permission/outputs/full'/pair/'BASE/final.png')
  for mode in ['box','guided']:
   cols=[('Source',tensor(source)),('Original FlowEdit',tensor(target))]
   for radius in [16,32,64,128]:
    filt=Guided(source,radius,mode);result=filt.reaction(target)
    source_error=float(np.max(np.abs(filt.reaction(source)-source)))
    assert source_error<2e-6
    path=out/pair/f'{mode}_{radius}.png';save_image(tensor(result),path);cols.append((f'r={radius}',tensor(result)))
    rows.append(dict(pair=pair,mode=mode,radius=radius,zero_reaction_error=source_error))
   panel(cols,out/f'{pair}_{mode}.png',title='Offline diagnostic only: '+pair,size=256)
 write(out/'results.json',dict(rows=rows,scope='endpoint image diagnostic, not a FlowEdit trajectory or inference result'))
 print('GUIDED_OFFLINE_COMPLETE',flush=True)

def full(args):
 seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');out=HERE/'outputs'/('guided_paired' if args.paired else 'guided_full')
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(sc.device)
 pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=sc.device)
 for pair in args.pairs:
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=data['x'].to(sc.device)
  es,et=[tuple(t.to(sc.device) for t in emb) for emb in data['embeddings']]
  seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
  with torch.no_grad():source=sc.image(x)
  source_np=source[0].permute(1,2,0).cpu().numpy();geom=Geometry(source);phase=Phase(source)
  def encode(im):
   z=sc.vae.encode(im*2-1).latent_dist.mode();return (z-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor
  with torch.no_grad():anchor=encode(source)
  for mode in args.modes:
   dest=out/pair/f'{mode}_{args.radius}'
   if (dest/'complete.json').exists():continue
   filt=Guided(source_np,args.radius,mode);gen=torch.Generator('cpu').manual_seed(0)
   noise=[torch.randn(x.shape,generator=gen).to(sc.device) for _ in range(33)];y=x.clone();states=[y.cpu()];rows=[];start=time.monotonic()
   for k in range(17,50):
    t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
    with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
     vs,vt=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k]);raw=y-(t-tn)*(vt-vs)
    with torch.no_grad():raw_np=sc.image(raw)[0].permute(1,2,0).cpu().numpy()
    projected=tensor(filt.reaction(raw_np),sc.device)
    with torch.no_grad():
     y=raw+encode(projected)-encode(tensor(raw_np,sc.device)) if args.paired else x+encode(projected)-anchor
    assert torch.isfinite(y).all();states.append(y.cpu());rows.append(dict(step=k,raw_increment=float((raw-states[-2].to(sc.device)).norm())))
    if k in [17,25,33,41,49]:
     with torch.no_grad():save_image(sc.image(y),dest/f'step{k}.png')
   with torch.no_grad():
    im=sc.image(y);metrics=dict(geometry=float(geom.loss(geom(im),40)),rgb_rmse=float((im-source).square().mean().sqrt()),
     background_top64_rmse=float((im[:,:,:64]-source[:,:,:64]).square().mean().sqrt()),**phase.diagnostics(im))
   save_image(im,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
   write(dest/'complete.json',dict(pair=pair,mode=mode,radius=args.radius,rows=rows,endpoint=metrics,seconds=time.monotonic()-start,paired=args.paired))
   print(pair,mode,args.radius,metrics,flush=True)
 print('GUIDED_FULL_COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00'])
 ap.add_argument('--modes',nargs='+',default=['box','guided']);ap.add_argument('--radius',type=int,default=64)
 ap.add_argument('--paired',dest='paired',action='store_true');ap.add_argument('--anchored',dest='paired',action='store_false');ap.set_defaults(paired=True);args=ap.parse_args()
 with sdpa_kernel(SDPBackend.MATH):
  if args.full:full(args)
  else:offline()
