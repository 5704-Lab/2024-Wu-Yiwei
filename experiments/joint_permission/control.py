"""Source-local geometry, latent matrix-free GN and finite text/state factors.

SIG uses two SigLIP factors and a linearized multi-factor repair. RF uses SD3
conditional residual factors as a finite verifier, without differentiating DiT.
Neither branch implements a general free-language parser.
"""
import argparse,sys,json,time,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn.attention import sdpa_kernel,SDPBackend
from diffusers import SD3Transformer2DModel
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,write,save_image,seed_setup,panel
from geometry import Geometry
sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from FlowEdit_utils import calc_v_sd3

CFG=dict(kappa=40.,radius=.6,gradient_step=.8,cg_steps=4,task_cg_steps=3,retention=.25,
         backtracks=4,rf_times=[.5,.75],seed=0,source_cfg=3.5,target_cfg=13.5)

def dot(a,b):return (a.double()*b.double()).sum()
def cg(op,rhs,n):
 x=torch.zeros_like(rhs);r=rhs.clone();p=r.clone();rr=dot(r,r);start=float(rr.sqrt())
 for k in range(n):
  if float(rr.sqrt())<max(start*1e-3,1e-10):break
  ap=op(p);den=dot(p,ap)
  if float(den)<=0:raise RuntimeError('non-positive GN operator')
  alpha=(rr/den).to(p.dtype);x=x+alpha*p;r=r-alpha*ap;new=dot(r,r)
  p=r+(new/rr).to(p.dtype)*p;rr=new
 return x,dict(relative_residual=float(rr.sqrt())/max(start,1e-20))

class Run:
 def __init__(self):
  self.device=torch.device('cuda:0');self.sc=CleanScorer(self.device,torch.float32,'native')
  self.model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,
              local_files_only=True).eval().requires_grad_(False).to(self.device)
  self.pipe=SimpleNamespace(transformer=self.model,do_classifier_free_guidance=True)
  self.short=torch.load(ROOT/'experiments/paired_relation_validation/outputs/short_condition_encodings.pt',weights_only=True)

 def setup(self,pair):
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
  x=data['x'].to(self.device)
  e,_=self.sc.encode([data['record']['target_prompt'],data['record']['source_prompt'],
                     self.short['short'][pair][1],self.short['short'][pair][0]])
  with torch.no_grad():source=self.sc.image(x)
  geometry=Geometry(source)
  es,et=[tuple(t.to(self.device) for t in emb) for emb in data['embeddings']]
  ss,st=[tuple(t.to(self.device) for t in emb) for emb in self.short['embeddings'][pair]]
  rf=[]
  for source_emb,target_emb in [(es,et),(ss,st)]:
   rf.append((torch.cat([source_emb[0],target_emb[0]]),torch.cat([source_emb[2],target_emb[2]])))
  gen=torch.Generator('cpu').manual_seed(733)
  return dict(x=x,source=source,geometry=geometry,e=e,rf=rf,data=data,
              auxnoise=torch.randn(x.shape,generator=gen).to(self.device),
              seq=torch.cat([es[1],es[0],et[1],et[0]]),pool=torch.cat([es[3],es[2],et[3],et[2]]))

 def sig(self,ctx,image):
  # Appearance contrast averaged over reflection; full target checked separately.
  logs=self.sc.logits_image(torch.cat([image,image.flip(-1)]),ctx['e'])
  return torch.stack([logs[0,0],(logs[:,2]-logs[:,3]).mean()])

 def rf(self,ctx,y):
  vals=[]
  with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
   for seq,pool in ctx['rf']:
    margins=[]
    for t in CFG['rf_times']:
     eps=ctx['auxnoise'];q=((1-t)*y+t*eps).expand(2,-1,-1,-1)
     pred=self.model(hidden_states=q,timestep=torch.full((2,),t*1000,device=y.device),
                     encoder_hidden_states=seq,pooled_projections=pool,return_dict=False)[0]
     err=(pred.double()-(eps-y).double()).square().mean((1,2,3))
     margins.append(float(err[0]-err[1]))
    vals.append(float(np.mean(margins)))
  return np.array(vals)

 def values(self,ctx,y,mode):
  if mode=='RF':return self.rf(ctx,y)
  with torch.no_grad():return self.sig(ctx,self.sc.image(y)).cpu().double().numpy()

 def correct(self,ctx,y,delta,mode,previous=None,audit=False):
  geom=ctx['geometry'];base=(y+delta).detach();started=time.monotonic()
  before=self.values(ctx,y,mode) if previous is None else previous
  with torch.no_grad():
   base_image=self.sc.image(base);base_obs=geom(base_image)
   base_loss=float(geom.loss(base_obs,CFG['kappa']))
  fbase=self.values(ctx,base,mode)
  floor=before+CFG['retention']*np.maximum(fbase-before,0)+np.minimum(fbase-before,0)
  tolerance=np.maximum(np.abs(fbase-before)*.05,1e-6 if mode=='RF' else .005)
  with sdpa_kernel(SDPBackend.MATH):
   point=base.detach().requires_grad_(True);image=self.sc.image(point);obs=geom(image)
   weights=geom.gn_weights(obs,CFG['kappa'])[:,:,None]/obs.shape[-1]
   residual=(obs-geom.source).detach()
   def jt(v):return torch.autograd.grad(obs,point,grad_outputs=v,retain_graph=True)[0].detach()
   b=jt(weights*residual)
   gradients=[]
   if mode=='SIG':
    facts=self.sig(ctx,image)
    for j in range(2):gradients.append(torch.autograd.grad(facts[j],point,retain_graph=True)[0].detach())
   lam=CFG['gradient_step']*float(delta.norm())/max(float(b.norm()),1e-12)
   def jv(v):
    with sdpa_kernel(SDPBackend.MATH):_,j=torch.func.jvp(lambda z:geom(self.sc.image(z)),(base,),(v,))
    return j.detach()
   def op(v):return v+lam*jt(weights*jv(v))
   c,ci=cg(op,-lam*b,CFG['cg_steps'])
   solver=dict(primary=ci)
   if mode=='SIG':
    us=[cg(op,g,CFG['task_cg_steps'])[0] for g in gradients]
    mat=np.array([[float(dot(g,u)) for u in us] for g in gradients]);mat=(mat+mat.T)/2
    mat+=np.eye(2)*max(np.trace(mat)*1e-5,1e-10)
    gaps=floor-fbase-np.array([float(dot(g,c)) for g in gradients])
    dual=minimize(lambda nu:.5*nu@mat@nu-gaps@nu,np.zeros(2),jac=lambda nu:mat@nu-gaps,
                  bounds=[(0,None)]*2,method='L-BFGS-B')
    for nu,u in zip(dual.x,us):c+=float(nu)*u
    solver['dual']=dual.x.tolist()
   radius=CFG['radius']*float(delta.norm())
   c=c*min(1.,radius/max(float(c.norm()),1e-20))
   checks={}
   if audit:
    v=delta/delta.norm();j=jv(v);left=float(dot(j,weights*residual));right=float(dot(v,b))
    checks['adjoint_error']=abs(left-right)/max(abs(left),1e-10)
    assert checks['adjoint_error']<1e-3,checks
  del point,image,obs
  accepted=False;new=base;fnew=fbase;trials=[]
  for retry in range(CFG['backtracks']):
   alpha=.5**retry;proposal=(base+alpha*c).detach()
   with torch.no_grad():newloss=float(geom.loss(geom(self.sc.image(proposal)),CFG['kappa']))
   merit=.5*float((proposal-base).square().sum())+lam*(newloss-base_loss)
   if newloss>=base_loss or merit>1e-6:
    trials.append(dict(alpha=alpha,loss=newloss,merit=merit,task_checked=False));continue
   values=self.values(ctx,proposal,mode)
   passed=bool(np.all(values>=floor-tolerance))
   trials.append(dict(alpha=alpha,loss=newloss,merit=merit,task_checked=True,values=values.tolist(),passed=passed))
   if passed:new=proposal;fnew=values;accepted=True;break
  info=dict(accepted=accepted,mode=mode,geometry_before=base_loss,geometry_after=trials[-1]['loss'] if accepted else base_loss,
            task_before=before.tolist(),task_base=fbase.tolist(),task_after=fnew.tolist(),task_floor=floor.tolist(),
            task_tolerance=tolerance.tolist(),correction_norm=float((new-base).norm()),delta_norm=float(delta.norm()),
            increment_ratio=float((new-y).norm()/delta.norm()),radius=radius,lambda_actual=lam,solver=solver,
            checks=checks,trials=trials,seconds=time.monotonic()-started)
  del jt,jv,op,b,residual,weights,c
  return new.detach(),info,fnew

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['local','full'],required=True)
 ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00']);ap.add_argument('--modes',nargs='+',default=['SIG','RF'])
 args=ap.parse_args();seed_setup();run=Run();out=HERE/'outputs/control'
 write(out/f'protocol_{args.stage}.json',dict(config=CFG,args=vars(args),scope='limited single-object colour tasks, not general instruction compilation'))
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=run.device)
 for pair in args.pairs:
  ctx=run.setup(pair);x=ctx['x'];source=ctx['source']
  saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
  for mode in args.modes:
   dest=out/args.stage/pair/mode;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'complete.json').exists():continue
   rows=[];states=[];previous=None;start=time.monotonic()
   if args.stage=='local':
    for step in [33,47]:
     y=saved['states'][step].to(run.device);delta=saved['states'][step+1].to(run.device)-y
     new,info,_=run.correct(ctx,y,delta,mode,audit=True);rows.append(dict(step=step,**info))
     with torch.no_grad():im=run.sc.image(new);base_im=run.sc.image(y+delta)
     save_image(im,dest/f'step{step}.png');torch.save(new.cpu(),dest/f'step{step}.pt')
     panel([('Source',source),('Original FlowEdit local',base_im),(f'GEO_CG_{mode}',im)],dest/f'step{step}_comparison.png',title=pair)
     print(pair,mode,step,info['accepted'],info['geometry_before'],info['geometry_after'],info['seconds'],flush=True)
   else:
    gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(run.device) for _ in range(33)]
    y=x.clone();states=[y.cpu()]
    for k in range(17,50):
     t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn)
     qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
     with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
      vs,va=calc_v_sd3(run.pipe,torch.cat([qs,qs,qa,qa]),ctx['seq'],ctx['pool'],3.5,13.5,times[k])
     y,info,previous=run.correct(ctx,y,-h*(va-vs),mode,previous,audit=k==17)
     rows.append(dict(step=k,**info));states.append(y.cpu())
     write(dest/'progress.json',dict(rows=rows))
     if k in [17,25,33,41,49]:
      with torch.no_grad():save_image(run.sc.image(y),dest/f'step{k}.png')
     print(pair,mode,k,info['accepted'],info['geometry_after'],time.monotonic()-start,flush=True)
    torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
    with torch.no_grad():im=run.sc.image(y);logs=run.sc.logits_image(im,ctx['e'])[0].cpu().tolist()
    save_image(im,dest/'final.png')
    write(dest/'endpoint.json',dict(full_target=logs[0],full_source=logs[1],short_target=logs[2],short_source=logs[3],
         geometry=float(ctx['geometry'].loss(ctx['geometry'](im),CFG['kappa'])),rf_factors=run.rf(ctx,y).tolist()))
   write(dest/'complete.json',dict(pair=pair,mode=mode,rows=rows,seconds=time.monotonic()-start,config=CFG))
   gc.collect();torch.cuda.empty_cache()
 print('CONTROL_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
