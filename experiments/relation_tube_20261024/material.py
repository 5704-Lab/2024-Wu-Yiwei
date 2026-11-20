"""Global source-colour LUT mechanism, with no spatial editing region.

Equal source colours use the same mapping at every location. This removes a
coordinate-specific head/background shortcut, but cannot bind identical-colour
objects. Source prototypes are deterministic per-image statistics, not training.
"""
import sys,time,json,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,seed_setup,write,save_image,panel,Phase
sys.path.insert(0,str(HERE.parent/'joint_permission'))
from geometry import Geometry

class Material:
 def __init__(self,source,pair,clusters=12,chroma_only=False):
  self.source=source;self.pair=pair;self.n=clusters;self.chroma_only=chroma_only
  self.gray=(source*torch.tensor([.299,.587,.114],device=source.device)[None,:,None,None]).sum(1,keepdim=True)
  features=self.features(source)
  pixels=features[0].permute(1,2,0).reshape(-1,features.shape[1]).cpu().numpy()
  sample=pixels[np.linspace(0,len(pixels)-1,4096,dtype=int)]
  centers=[sample[np.argmin(np.sum((sample-sample.mean(0))**2,1))]]
  for _ in range(clusters-1):
   distances=np.min(np.sum((sample[:,None]-np.array(centers)[None])**2,2),1)
   centers.append(sample[np.argmax(distances)])
  centers=np.array(centers)
  for _ in range(20):
   labels=np.argmin(np.sum((sample[:,None]-centers[None])**2,2),1)
   centers=np.array([sample[labels==k].mean(0) if np.any(labels==k) else centers[k] for k in range(clusters)])
  self.centers=torch.tensor(centers,device=source.device)
  distances=(features[:,None]-self.centers[None,:,:,None,None]).square().sum(2)
  self.weights=torch.softmax(-distances/.12**2,1)
  self.dim=clusters*2

 def features(self,image):
  gray=(image*torch.tensor([.299,.587,.114],device=image.device)[None,:,None,None]).sum(1,keepdim=True)
  total=image.sum(1,keepdim=True).clamp_min(1e-6) if self.chroma_only else image.sum(1,keepdim=True)+.05
  chroma=[2*(image[:,0:1]-image[:,1:2])/total,2*(image[:,2:3]-image[:,1:2])/total]
  return torch.cat(chroma if self.chroma_only else [gray,*chroma],1)

 def apply(self,image,theta):
  features=self.features(image);distance=(features[:,None]-self.centers[None,:,:,None,None]).square().sum(2)
  weights=torch.softmax(-distance/.12**2,1)
  b=(weights*theta[:self.n][None,:,None,None]).sum(1,keepdim=True)
  s=(weights*theta[self.n:][None,:,None,None]).sum(1,keepdim=True)
  gray=(image*torch.tensor([.299,.587,.114],device=image.device)[None,:,None,None]).sum(1,keepdim=True)
  neutral=(1-s)*image+s*gray
  return neutral*torch.exp(-2*b) if self.pair=='bear__00' else (1-b)*neutral+b

 def image(self,theta):
  b=(self.weights*theta[:self.n][None,:,None,None]).sum(1,keepdim=True)
  s=(self.weights*theta[self.n:][None,:,None,None]).sum(1,keepdim=True)
  neutral=(1-s)*self.source+s*self.gray
  if self.pair=='bear__00':return neutral*torch.exp(-2*b)
  return (1-b)*neutral+b

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00'])
 ap.add_argument('--full',action='store_true');ap.add_argument('--chroma-only',action='store_true');args=ap.parse_args()
 seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
 name=('material_full' if args.full else 'material_pilot')+('_chroma' if args.chroma_only else '')
 out=HERE/'outputs'/name
 if args.full:
  from diffusers import SD3Transformer2DModel
  from types import SimpleNamespace
  sys.path.insert(0,str(ROOT/'vendor/flowedit'))
  from FlowEdit_utils import calc_v_sd3
  model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(sc.device)
  pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
  times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=sc.device)
 for pair in args.pairs:
  dest=out/pair
  if (dest/'complete.json').exists():continue
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=data['x'].to(sc.device)
  with torch.no_grad():source=sc.image(x)
  material=Material(source,pair,clusters=8 if args.chroma_only else 12,chroma_only=args.chroma_only);geom=Geometry(source);phase=Phase(source)
  upper=np.ones(material.dim)
  if args.chroma_only and pair=='brown_owl__00':upper[:material.n]=.9
  names=['A bear with black fur.','A bear with brown fur.'] if pair=='bear__00' else ['An owl with white feathers.','An owl with brown feathers.']
  e,_=sc.encode(names)
  def encode(im):
   z=sc.vae.encode(im*2-1).latent_dist.mode();return (z-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor
  with torch.no_grad():anchor=encode(source)
  def state(p):return x+encode(material.image(p))-anchor
  def fact(y):
   im=sc.image(y);logs=sc.logits_image(torch.cat([im,im.flip(-1)]),e)
   return (logs[:,0]-logs[:,1]).mean()
  theta=np.zeros(material.dim);rows=[];start=time.monotonic();states=[x.cpu()];previous_y=x.clone()
  if args.full:
   es,et=[tuple(t.to(sc.device) for t in emb) for emb in data['embeddings']]
   seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
   gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(sc.device) for _ in range(33)]
  with torch.no_grad():current=float(fact(x))
  source_margin=current
  for k in range(33 if args.full else 40):
   if args.full:
    j=k+17;t=times[j]/1000;tn=times[j+1]/1000 if j<49 else torch.zeros_like(t)
    qs=(1-t)*x+t*noise[k];qa=qs+previous_y-x
    with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
     vs,vt=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[j])
     delta=-(t-tn)*(vt-vs);proposal=previous_y+delta;scale=max(float(delta.square().sum()),1e-5)
   desired=max(0.,min(.16 if args.full else .15,3.5-current));lo=np.maximum(0,theta-.12);hi=np.minimum(upper,theta+.12)
   cache={}
   def evaluate(a):
    key=np.asarray(a,np.float64).tobytes()
    if key in cache:return cache[key]
    p=torch.tensor(a,device=sc.device,dtype=x.dtype,requires_grad=True);im=material.image(p)
    loss=.5*(im-source).square().mean()/source.square().mean().clamp_min(1e-5)+.001*p.square().mean()
    latent=state(p)
    if args.full:loss=loss+.002*(latent-proposal).square().sum()/scale
    f=fact(latent)
    gl,=torch.autograd.grad(loss,p,retain_graph=True);gf,=torch.autograd.grad(f,p)
    cache[key]=(float(loss),gl.cpu().double().numpy(),float(f),gf.cpu().double().numpy())
    return cache[key]
   accepted=False;trials=[];old=theta.copy()
   for fraction in [1.,.5,.25,0.]:
    requested=current+fraction*desired;guess=old.copy()
    for _ in range(4):
     lv,gl,fv,gf=evaluate(guess);low=lo-guess;high=hi-guess;metric=max(np.linalg.norm(gl)/.05,1.)
     required=requested-fv
     def direction(mu):return np.clip((mu*gf-gl)/metric,low,high)
     step=direction(0.)
     if gf@step<required:
      if gf@np.where(gf>=0,high,low)<required:break
      left,right=0.,1.
      for _ in range(80):
       if gf@direction(right)>=required:break
       right*=2
      for _ in range(60):
       mid=(left+right)/2
       if gf@direction(mid)<required:left=mid
       else:right=mid
      step=direction(right)
     guess=np.clip(guess+step,lo,hi)
     if evaluate(guess)[2]>=requested-.002:break
    actual=evaluate(guess)[2];passed=actual>=requested-.002
    trials.append(dict(fraction=fraction,requested=requested,actual=actual,passed=bool(passed)))
    if passed:theta=guess;current=actual;accepted=True;break
   rows.append(dict(iteration=k,step=k+17 if args.full else None,accepted=accepted,margin=current,theta=theta.tolist(),trials=trials,calls=len(cache)))
   write(dest/'progress.json',dict(rows=rows,source_margin=source_margin))
   with torch.no_grad():y=state(torch.tensor(theta,device=sc.device,dtype=x.dtype));im=sc.image(y)
   if args.full:previous_y=y.detach();states.append(y.cpu())
   if k%5==0 or current>=3.498:save_image(im,dest/f'iteration{k}.png')
   print(pair,k,current,accepted,len(cache),time.monotonic()-start,flush=True)
   if not args.full and (current>=3.498 or (len(rows)>=3 and all(not r['accepted'] for r in rows[-3:]))):break
  with torch.no_grad():
   m=dict(source_margin=source_margin,final_margin=current,geometry=float(geom.loss(geom(im),40)),
    rgb_rmse=float((im-source).square().mean().sqrt()),background_top64_rmse=float((im[:,:,:64]-source[:,:,:64]).square().mean().sqrt()),
    **phase.diagnostics(im))
  save_image(im,dest/'final.png');torch.save(dict(final=y.cpu(),theta=torch.tensor(theta),centers=material.centers.cpu()),dest/'state.pt')
  if args.full:torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
  panel([('Source',source),('Colour LUT pilot',im)],dest/'comparison.png',title=pair)
  write(dest/'complete.json',dict(rows=rows,endpoint=m,seconds=time.monotonic()-start,chroma_only=args.chroma_only,
   parameter_upper=upper.tolist(),clusters=material.n,target_margin=3.5,
   scope='source-colour LUT with native FlowEdit proposal prior' if args.full else 'mechanism reachability / proxy pilot with no native FlowEdit velocity; not a full FlowEdit trajectory'))
 print('MATERIAL_PILOT_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
