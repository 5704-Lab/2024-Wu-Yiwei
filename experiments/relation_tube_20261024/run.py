"""Real-state ablations of shared phase/amplitude/fact source relations.

Full latent correction, truncated matrix-free GN, bounded cumulative source
appearance parameters and actual finite merit/factor checks. This is not an
exact hard-tube QP, nor a validated general language-binding system.
"""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,Phase,Appearance,seed_setup,write,save_image,panel
sys.path.insert(0,str(HERE.parent/'joint_permission'))
from control import Run,cg,dot
from FlowEdit_utils import calc_v_sd3

CFG=dict(order=2,complex_weight=.5,factor_weight=.1,probe=.01,ridge=.01,
 gradient_step=1.2,radius=1.,cg_steps=4,retention=.25,task_tolerance=.005,
 backtracks=5,source_cfg=3.5,target_cfg=13.5,seed=0)
MODES=['BASE','PHASE','SHARED','INDEPENDENT','SHARED_FACT','VELOCITY','TUBE']

class Relation:
 def __init__(self,run,ctx,mode):
  self.run=run;self.ctx=ctx;self.mode=mode;self.phase=Phase(ctx['source']);self.app=Appearance(ctx['source'],CFG['order'])
  with torch.no_grad():
   z=self.phase.coefficients(ctx['source']);self.scale=torch.sqrt(z.abs().square()+self.phase.epsilon.square())
  self.dim=self.app.n*(2 if mode=='INDEPENDENT' else 1)
  self.bound=self.app.bound

 def facts(self,im):
  b=len(im);logs=self.run.sc.logits_image(torch.cat([im,im.flip(-1)]),self.ctx['e'])
  return torch.stack([logs[:b,0],((logs[:b,2]-logs[:b,3])+(logs[b:,2]-logs[b:,3]))*.5],1)

 def parts(self,im):
  z=self.phase.coefficients(im);p=self.phase.embed(z)
  c=torch.view_as_real(z/self.scale*self.phase.sqrt_weight).flatten(1)*CFG['complex_weight']
  return p,c

 def observe(self,im):
  p,c=self.parts(im)
  if self.mode=='PHASE':return p
  out=[p,c]
  if self.mode in ['SHARED_FACT','TUBE']:out.append(self.facts(im)*CFG['factor_weight'])
  return torch.cat(out,1)

 def reference(self,a):
  a=a.reshape(-1,self.dim)
  if self.mode=='INDEPENDENT':
   n=self.app.n
   first=self.app(a[:,:n]*self.bound).clamp(0,1);second=self.app(a[:,n:]*self.bound).clamp(0,1)
   p,_=self.parts(first);_,c=self.parts(second)
   return torch.cat([p,c],1)
  return self.observe(self.app(a*self.bound).clamp(0,1))

 def jacobian(self,theta):
  columns=[];eps=CFG['probe']
  with torch.no_grad():
   for k in range(0,self.dim,4):
    ids=list(range(k,min(k+4,self.dim)));batch=[]
    for j in ids:
     e=torch.zeros_like(theta);e[j]=eps;batch.extend([theta+e,theta-e])
    obs=self.reference(torch.stack(batch))
    columns.extend([(obs[2*j]-obs[2*j+1])/(2*eps) for j in range(len(ids))])
  return torch.stack(columns,1)

 def fit(self,observation):
  def fun(a):
   p=torch.tensor(a,device=self.ctx['x'].device,dtype=torch.float32,requires_grad=True)
   residual=self.reference(p)[0]-observation
   loss=.5*residual.square().sum()+1e-5*p.square().sum()
   g,=torch.autograd.grad(loss,p)
   return float(loss),g.cpu().double().numpy()
  opt=minimize(fun,np.zeros(self.dim),jac=True,method='L-BFGS-B',bounds=[(-1.,1.)]*self.dim,
    options=dict(maxiter=12,ftol=1e-8,gtol=1e-6))
  return torch.tensor(opt.x,device=self.ctx['x'].device,dtype=torch.float32),dict(success=bool(opt.success),iterations=int(opt.nit))

 def correct(self,y,delta,theta,audit=False):
  started=time.monotonic();base=(y+delta).detach()
  with torch.no_grad():
   before=self.facts(self.run.sc.image(y))[0];im=self.run.sc.image(base)
   ob=self.observe(im)[0];fb=self.facts(im)[0];ref=self.reference(theta)[0]
   offset=self.observe(self.run.sc.image(y))[0]-ref if self.mode=='VELOCITY' else torch.zeros_like(ref)
   err=ob-ref-offset
  floor=before+CFG['retention']*(fb-before).clamp_min(0)+(fb-before).clamp_max(0)
  v=self.jacobian(theta);gram=v.T.double()@v.double()
  ridge=max(float(torch.trace(gram))/self.dim*CFG['ridge'],1e-8)
  factor=torch.linalg.cholesky(gram+ridge*torch.eye(self.dim,device=y.device,dtype=torch.float64))
  def coeff(r):return torch.cholesky_solve((v.T.double()@r.double()).reshape(self.dim,-1),factor).to(r.dtype).reshape(self.dim,*r.shape[1:])
  def profile(r):return r-v@coeff(r)
  point=base.detach().requires_grad_(True);image=self.run.sc.image(point);obs=self.observe(image)[0]
  def jt(r):return torch.autograd.grad(obs,point,grad_outputs=r,retain_graph=True)[0].detach()
  gobs=jt(profile(err))
  f=self.facts(image)[0]
  gradients=[torch.autograd.grad(f[j],point,retain_graph=True)[0].detach() for j in range(2)]
  lam=CFG['gradient_step']*float(delta.norm())/max(float(gobs.norm()),1e-12)
  def jv(d):
   _,r=torch.func.jvp(lambda z:self.observe(self.run.sc.image(z))[0],(base,),(d,))
   return r.detach()
  def op(d):return d+lam*jt(profile(jv(d)))
  correction,solver=cg(op,-lam*gobs,CFG['cg_steps'])
  # A cheap multi-factor Euclidean repair; actual merit is checked afterwards.
  mat=np.array([[float(dot(a,b)) for b in gradients] for a in gradients]);mat=(mat+mat.T)/2
  mat+=np.eye(2)*max(float(np.trace(mat))*1e-7,1e-12)
  gaps=(floor-fb).cpu().double().numpy()-np.array([float(dot(g,correction)) for g in gradients])
  dual=minimize(lambda nu:.5*nu@mat@nu-gaps@nu,np.zeros(2),jac=lambda nu:mat@nu-gaps,
    bounds=[(0,None)]*2,method='L-BFGS-B')
  for nu,g in zip(dual.x,gradients):correction+=float(nu)*g
  radius=CFG['radius']*float(delta.norm());scale=min(1.,radius/max(float(correction.norm()),1e-20))
  correction=correction*scale
  target=err+jv(correction)
  small_h=(gram+ridge*torch.eye(self.dim,device=y.device,dtype=torch.float64)).cpu().numpy()
  small_b=(v.T.double()@target.double()).cpu().numpy()
  th=theta.cpu().double().numpy();initial=np.clip(coeff(target).cpu().double().numpy(),-1-th,1-th)
  bounded=minimize(lambda a:.5*a@small_h@a-small_b@a,initial,jac=lambda a:small_h@a-small_b,
   bounds=list(zip(-1-th,1-th)),method='L-BFGS-B',options=dict(maxiter=80,ftol=1e-12,gtol=1e-8))
  a=torch.tensor(bounded.x,device=y.device,dtype=y.dtype)
  checks={}
  if audit:
   direction=delta/delta.norm();j=jv(direction);left=float(dot(j,profile(err)));right=float(dot(direction,gobs))
   checks['adjoint_relative_error']=abs(left-right)/max(abs(left),abs(right),1e-9)
   assert checks['adjoint_relative_error']<2e-3,checks
  del point,image,obs
  accepted=False;new=base;newtheta=theta;trials=[];newfacts=fb
  oldloss=float(err.square().sum())
  for k in range(CFG['backtracks']):
   alpha=.5**k;candidate=(base+alpha*correction).detach();th=theta+alpha*a
   with torch.no_grad():
    image=self.run.sc.image(candidate);r=self.observe(image)[0]-self.reference(th)[0]-offset
    facts=self.facts(image)[0];loss=float(r.square().sum())
   merit=.5*float((candidate-base).square().sum())+.5*lam*(loss-oldloss+ridge*float((alpha*a).square().sum()))
   inside=self.mode!='TUBE' or loss<=CFG.get('tube_radius',.1)**2+1e-8
   passed=bool(torch.all(facts>=floor-CFG['task_tolerance']) and merit<=1e-6 and loss<oldloss and inside)
   trials.append(dict(alpha=alpha,loss=loss,merit=merit,facts=facts.cpu().tolist(),passed=passed))
   if passed:new=candidate;newtheta=th.detach();accepted=True;newfacts=facts;break
  status='corrected' if accepted else 'raw'
  actual_loss=trials[-1]['loss'] if accepted else oldloss
  if self.mode=='TUBE' and not accepted and oldloss>CFG.get('tube_radius',.1)**2+1e-8:
   new=y;newtheta=theta;newfacts=before;status='tube_hold'
   with torch.no_grad():actual_loss=float((self.observe(self.run.sc.image(y))[0]-self.reference(theta)[0]).square().sum())
  if self.mode=='TUBE':assert actual_loss<=CFG.get('tube_radius',.1)**2+1e-8
  info=dict(accepted=accepted,status=status,loss_before=oldloss,loss_after=actual_loss,
   task_before=before.cpu().tolist(),task_base=fb.cpu().tolist(),task_after=newfacts.cpu().tolist(),
   task_floor=floor.cpu().tolist(),radius=radius,correction_norm=float((new-base).norm()),
   nominal_increment_norm=float(delta.norm()),parameter_step_norm=float((newtheta-theta).norm()),theta=newtheta.cpu().tolist(),
   lambda_actual=lam,ridge=ridge,box_scale=scale,parameter_solver_success=bool(bounded.success),solver=solver,dual=dual.x.tolist(),checks=checks,
   trials=trials,task_floor_satisfied=bool(torch.all(newfacts>=floor-CFG['task_tolerance'])),seconds=time.monotonic()-started)
  del jt,jv,op,gradients,correction,gobs
  return new.detach(),newtheta,info

def metrics(run,ctx,y):
 with torch.no_grad():
  im=run.sc.image(y);logs=run.sc.logits_image(torch.cat([im,im.flip(-1)]),ctx['e'])
  phase=Phase(ctx['source'])
  return dict(full_target=float(logs[0,0]),short_contrast=float((logs[:,2]-logs[:,3]).mean()),
   geometry=float(ctx['geometry'].loss(ctx['geometry'](im),40.)),
   rgb_rmse=float((im-ctx['source']).square().mean().sqrt()),
   background_top64_rmse=float((im[:,:,:64]-ctx['source'][:,:,:64]).square().mean().sqrt()),
   **phase.diagnostics(im)),im

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['local','full'],required=True)
 ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00']);ap.add_argument('--modes',nargs='+',default=MODES)
 ap.add_argument('--steps',nargs='+',type=int,default=[17,33,47]);ap.add_argument('--radius',type=float,default=1.)
 ap.add_argument('--tag',default='v1');ap.add_argument('--tube-radius',type=float,default=.1)
 ap.add_argument('--cg-steps',type=int,default=4)
 args=ap.parse_args();CFG['radius']=args.radius;CFG['tube_radius']=args.tube_radius;CFG['cg_steps']=args.cg_steps
 seed_setup();run=Run();out=HERE/'outputs'/args.tag
 protocol=dict(config=CFG,args=vars(args),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  scope='pure-colour development tasks; existing global factors, not validated object binding; approximate GN, not exact hard-tube solver')
 protocol_path=out/f'protocol_{args.stage}.json'
 if protocol_path.exists():
  digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12]
  protocol_path=out/f'protocol_{args.stage}_attempt_{digest}.json'
 if not protocol_path.exists():write(protocol_path,protocol)
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=run.device)
 for pair in args.pairs:
  ctx=run.setup(pair);x=ctx['x'];saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
  for mode in args.modes:
   dest=out/args.stage/pair/mode
   if (dest/'complete.json').exists():continue
   rel=Relation(run,ctx,mode);rows=[];states=[x.cpu()];y=x.clone();theta=torch.zeros(rel.dim,device=x.device)
   gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(x.device) for _ in range(33)]
   start=time.monotonic()
   for k in (args.steps if args.stage=='local' else range(17,50)):
    if args.stage=='local':
     y=saved['states'][k].to(x.device);delta=saved['states'][k+1].to(x.device)-y
     with torch.no_grad():ob=rel.observe(run.sc.image(y))[0]
     theta,fit=rel.fit(ob)
    else:
     t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t)
     qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
     with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
      vs,vt=calc_v_sd3(run.pipe,torch.cat([qs,qs,qa,qa]),ctx['seq'],ctx['pool'],3.5,13.5,times[k])
     delta=-(t-tn)*(vt-vs)
    if mode=='BASE':
     new=(y+delta).detach();info=dict(accepted=False,status='base',baseline_step_error=float((new-saved['states'][k+1].to(x.device)).abs().max()))
     assert info['baseline_step_error']<1e-6,info
    else:new,theta,info=rel.correct(y,delta,theta,audit=(args.stage=='local' or k==17))
    assert torch.isfinite(new).all()
    row=dict(step=k,**info);rows.append(row);write(dest/'progress.json',dict(rows=rows))
    if args.stage=='local':
     m,im=metrics(run,ctx,new);row['metrics']=m;save_image(im,dest/f'step{k}.png')
     with torch.no_grad():base=run.sc.image(y+delta)
     panel([('Source',ctx['source']),('Original FlowEdit',base),(mode,im)],dest/f'step{k}_comparison.png',title=pair)
     torch.save(dict(state=new.cpu(),row=row),dest/f'step{k}.pt')
    else:
     y=new;states.append(y.cpu())
     if k in [17,25,33,41,49]:
      with torch.no_grad():save_image(run.sc.image(y),dest/f'step{k}.png')
    print(pair,mode,k,info['accepted'],info.get('loss_before'),info.get('loss_after'),info.get('seconds'),flush=True)
   result=dict(rows=rows,config=CFG,seconds=time.monotonic()-start)
   if args.stage=='full':
    m,im=metrics(run,ctx,y);result['endpoint']=m
    save_image(im,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu(),theta=theta.cpu()),dest/'trajectory.pt')
   write(dest/'complete.json',result);gc.collect();torch.cuda.empty_cache()
 print('RELATION_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
