"""Explicit semantic progress on the finite source-anchored appearance family.

The task is a declared single-object fur/feather colour fact. It is not inferred
from arbitrary instructions. Source contrast sign is checked before proceeding.
"""
import sys,json,time,argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,seed_setup,write,save_image
from control import Run
from FlowEdit_utils import calc_v_sd3

CFG=dict(order=4,brightness_log_bound=2.,brightness_step=.06,desaturation_step=.08,
         beta=.08,goal_margin=.5,iterations=4,semantic_tolerance=.002)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00'])
 ap.add_argument('--tag',default='achromatic');ap.add_argument('--goal-margin',type=float,default=.5)
 ap.add_argument('--beta',type=float,default=.08)
 ap.add_argument('--family',choices=['exposure','neutralmix'],default='exposure');args=ap.parse_args()
 CFG['goal_margin']=args.goal_margin;CFG['beta']=args.beta;CFG['family']=args.family
 seed_setup();run=Run();out=HERE/'outputs'/args.tag;write(out/'protocol.json',dict(config=CFG,pairs=args.pairs,
   changed_from_penalty='source-anchored brightness/desaturation family; minimum cumulative RGB change with FlowEdit proposal prior and explicit attribute progress'))
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=run.device)
 for pair in args.pairs:
  dest=out/pair;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists():continue
  ctx=run.setup(pair);x=ctx['x'];source=ctx['source'];m=CFG['order']**2;n=source.shape[-1]
  names=['A bear with black fur.','A bear with brown fur.'] if pair=='bear__00' else ['An owl with white feathers.','An owl with brown feathers.']
  embeddings,_=run.sc.encode(names)
  grid=(torch.arange(n,device=x.device,dtype=x.dtype)+.5)/n
  basis=torch.stack([torch.cos(np.pi*i*grid)[:,None]*torch.cos(np.pi*j*grid)[None,:] for i in range(CFG['order']) for j in range(CFG['order'])])
  bounds=np.ones(2*m)
  steps=np.r_[np.full(m,.06),np.full(m,.08)]
  brightness_sign=-1. if pair=='bear__00' else 1.
  source_gray=(source*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
  tb=torch.tensor(bounds,device=x.device,dtype=x.dtype)
  def encode(im):
   z=run.sc.vae.encode(im*2-1).latent_dist.mode();return (z-run.sc.vae.config.shift_factor)*run.sc.vae.config.scaling_factor
  with torch.no_grad():anchor=encode(source)
  def photograph(a):
   brightness=torch.nn.functional.interpolate(a[:m].reshape(1,1,CFG['order'],CFG['order']),source.shape[-2:],mode='bilinear',align_corners=False)
   desaturation=torch.nn.functional.interpolate(a[m:].reshape(1,1,CFG['order'],CFG['order']),source.shape[-2:],mode='bilinear',align_corners=False)
   neutral=(1-desaturation)*source+desaturation*source_gray
   if CFG['family']=='neutralmix':
    return (1-brightness)*neutral+brightness*(0. if brightness_sign<0 else 1.)
   return (neutral*torch.exp(brightness_sign*CFG['brightness_log_bound']*brightness)).clamp(0,1)
  def state(a):
   return x+encode(photograph(a))-anchor
  def fact(z):
   im=run.sc.image(z);scores=run.sc.logits_image(torch.cat([im,im.flip(-1)]),embeddings)
   return (scores[:,0]-scores[:,1]).mean()
  y=x.clone();theta=np.zeros(2*m);states=[y.cpu()];rows=[];start=time.monotonic()
  with torch.no_grad():source_margin=float(fact(y));zero_error=float((state(torch.zeros(2*m,device=x.device))-x).abs().max())
  write(dest/'source_check.json',dict(texts=names,source_margin=source_margin,source_zero_error=zero_error))
  if source_margin>=0:
   write(dest/'blocked.json',dict(reason='declared target attribute already scored above source on unedited source; evidence fails source check'));continue
  gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(x.device) for _ in range(33)]
  for k in range(17,50):
   t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn)
   qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
   with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
    vs,va=calc_v_sd3(run.pipe,torch.cat([qs,qs,qa,qa]),ctx['seq'],ctx['pool'],3.5,13.5,times[k])
    delta=-h*(va-vs);proposal=y+delta
    current=float(fact(y))
   desired=max(0.,min(CFG['beta'],CFG['goal_margin']-current))
   scale=max(float(delta.square().sum()),1e-5);old=theta.copy()
   lo=np.maximum(0.,old-steps/bounds);hi=np.minimum(1.,old+steps/bounds)
   cache={}
   def evaluate(a):
    key=np.asarray(a,dtype=np.float64).tobytes()
    if key in cache:return cache[key]
    p=torch.tensor(a,device=x.device,dtype=x.dtype,requires_grad=True);z=state(p)
    loss=.5*(photograph(p)-source).square().mean()/source.square().mean().clamp_min(1e-5)+.002*(z-proposal).square().sum()/scale+.001*p.square().mean()
    f=fact(z)
    gl,=torch.autograd.grad(loss,p,retain_graph=True);gf,=torch.autograd.grad(f,p)
    result=(float(loss),gl.cpu().double().numpy(),float(f),gf.cpu().double().numpy())
    cache[key]=result;return result
   attempts=[];new=y;accepted=False;new_margin=current;newtheta=old
   for fraction in [1.,.5,.25,0.]:
    target=current+desired*fraction
    guess=old.copy();nit=0
    for nit in range(CFG['iterations']):
     lv,gl,fv,gf=evaluate(guess)
     low=lo-guess;high=hi-guess;metric=max(np.linalg.norm(gl)/.05,1.)
     required=target-fv
     def direction(mu):return np.clip((mu*gf-gl)/metric,low,high)
     step=direction(0.)
     if gf@step<required:
      if gf@np.where(gf>=0,high,low)<required:break
      left_mu,right_mu=0.,1.
      for _ in range(80):
       if gf@direction(right_mu)>=required:break
       right_mu*=2
      for _ in range(60):
       mid=(left_mu+right_mu)/2
       if gf@direction(mid)<required:left_mu=mid
       else:right_mu=mid
      step=direction(right_mu)
     guess=np.clip(guess+step,lo,hi)
     if evaluate(guess)[2]>=target-CFG['semantic_tolerance']:break
    opt=SimpleNamespace(x=guess,success=evaluate(guess)[2]>=target-CFG['semantic_tolerance'],nit=nit+1)
    with torch.no_grad():candidate=state(torch.tensor(opt.x,device=x.device,dtype=x.dtype));fm=float(fact(candidate))
    passed=bool(fm>=target-CFG['semantic_tolerance'] and np.isfinite(fm))
    attempts.append(dict(fraction=fraction,requested=target,actual=fm,solver_success=bool(opt.success),iterations=int(opt.nit),accepted=passed))
    if passed:new=candidate.detach();newtheta=opt.x;new_margin=fm;accepted=True;break
   y=new;theta=newtheta;states.append(y.cpu())
   rows.append(dict(step=k,accepted=accepted,margin_before=current,margin_after=new_margin,desired_gain=desired,
       actual_gain=new_margin-current,attempts=attempts,theta=(theta*bounds).tolist(),base_increment_norm=float(delta.norm()),
       increment_norm=float((y-states[-2].to(y.device)).norm()),calls=len(cache)))
   write(dest/'progress.json',dict(rows=rows))
   if k in [17,25,33,41,49]:
    with torch.no_grad():save_image(run.sc.image(y),dest/f'step{k}.png')
   print(pair,k,accepted,current,new_margin,len(cache),time.monotonic()-start,flush=True)
  with torch.no_grad():im=run.sc.image(y);logs=run.sc.logits_image(im,ctx['e'])[0].cpu().tolist();geometry=float(ctx['geometry'].loss(ctx['geometry'](im),40.))
  save_image(im,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
  write(dest/'complete.json',dict(rows=rows,config=CFG,seconds=time.monotonic()-start,source_margin=source_margin,
      final_margin=new_margin,full_target=logs[0],full_source=logs[1],short_target=logs[2],short_source=logs[3],geometry=geometry,
      rf=run.rf(ctx,y).tolist()))
 print('ACHROMATIC_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
