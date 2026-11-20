"""Pre-registered adaptive total-increment proximal control, frozen inference."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import minimize
from PIL import Image,ImageDraw
from torch.nn.attention import sdpa_kernel,SDPBackend
from run_precheck import CleanScorer,ROOT

OUT=Path(__file__).parent/'adaptive_prox';OUT.mkdir(exist_ok=True)
PAIRS=['bear__00','brown_owl__00','bus__00','cat_crown__00']
CONFIG=dict(omega_min=.25,omega_max=4.,power=2.,gradient_rms_floor=1e-8,
 gradient_norm_floor=1e-8,gamma_ratio=.1,V_floor=0.,V_cap=10000.,
 local_lambdas=[.25,1.,4.],full_lambda=1.,seed=0,shuffle_seed=123,
 basis='fixed latent coordinate',precision='float32',full_methods=['G','ISO','ADAPT'],
 scope='development mechanism diagnostic, not independent task quality or speed validation')

def weights(g):
 rho=torch.sqrt(g.square().mean()+CONFIG['gradient_rms_floor']**2)
 return CONFIG['omega_min']+(CONFIG['omega_max']-CONFIG['omega_min'])/(1+(g.abs()/rho)**2)

def solve(d,g,W,lam,h):
 gamma=CONFIG['gamma_ratio']*d.norm();gu=g/g.norm().clamp_min(CONFIG['gradient_norm_floor'])
 b=d+gamma*gu;A=1+lam*W;R=min(float(h)*CONFIG['V_cap'],float(d.norm()+gamma))
 mu=0.;u=b/A
 if float(u.norm())>R:
  lo=0.;hi=1.
  while float((b/(A+hi)).norm())>R:hi*=2
  for _ in range(50):
   mid=(lo+hi)/2
   if float((b/(A+mid)).norm())>R:lo=mid
   else:hi=mid
  mu=hi;u=b/(A+mu)
 return u,dict(mu=mu,radius=R,gamma=float(gamma),budget_ratio=float(u.norm())/max(R,1e-30))

def analytic():
 rng=np.random.default_rng(42);checks=[]
 for idx in range(8):
  n=7;d=rng.normal(size=n);g=rng.normal(size=n);W=rng.uniform(.25,4,size=n);lam=[.25,1,4][idx%3]
  # Some tests deliberately activate the ball independently of deployment radius.
  gamma=.1*np.linalg.norm(d);b=d+gamma*g/np.linalg.norm(g);A=1+lam*W
  R=(.3 if idx%2 else 1.1)*np.linalg.norm(d)
  lo=0.;hi=1.
  if np.linalg.norm(b/A)<=R:mu=0.
  else:
   while np.linalg.norm(b/(A+hi))>R:hi*=2
   for _ in range(80):
    mid=(lo+hi)/2
    if np.linalg.norm(b/(A+mid))>R:lo=mid
    else:hi=mid
   mu=hi
  u=b/(A+mu)
  f=lambda v:.5*np.sum((v-d)**2)+.5*lam*np.sum(W*v*v)-gamma*np.dot(g/np.linalg.norm(g),v)
  ref=minimize(f,np.zeros(n),jac=lambda v:A*v-b,constraints=[dict(type='ineq',fun=lambda v:R*R-v@v,jac=lambda v:-2*v)],method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
  checks.append(dict(test=idx,solver_success=bool(ref.success),relative_solution_error=float(np.linalg.norm(u-ref.x)/(1+np.linalg.norm(u))),stationarity=float(np.linalg.norm((A+mu)*u-b)),feasible=bool(np.linalg.norm(u)<=R+1e-10)))
 # Independently verify deployment solve, zero cases and effective velocity signs.
 d=torch.tensor([1.,-2.],dtype=torch.float64);g=torch.zeros_like(d)
 u,info=solve(d,g,torch.full_like(d,4),1.,.1)
 zero,_=solve(g,g,torch.full_like(d,4),1.,.1)
 x=torch.tensor([.3,.9],dtype=torch.float64);qs=torch.tensor([.6,.4],dtype=torch.float64);w=u-d;h=.1
 closure=float(((qs+x+u-x)-(qs+u)).abs().max())
 effective=float(((-d/h-w/h)-(-u/h)).abs().max())
 passed=all(c['solver_success'] and c['relative_solution_error']<1e-5 and c['feasible'] for c in checks) and bool(torch.isfinite(u).all()) and float(zero.norm())==0 and closure<1e-12 and effective<1e-12
 result=dict(passed=passed,checks=checks,zero_gradient_shrink=u.tolist(),zero_zero_norm=float(zero.norm()),query_error=closure,effective_velocity_error=effective)
 (OUT/'analytic.json').write_text(json.dumps(result,indent=2)+'\n')
 if not passed:raise RuntimeError('analytic gate failed')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['local','full'],required=True);args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 provenance=dict(config=CONFIG,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stage=args.stage)
 (OUT/f'protocol_{args.stage}.json').write_text(json.dumps(provenance,indent=2)+'\n')
 analytic();device=torch.device('cuda:0');sc=CleanScorer(device,torch.float32,'native')
 def grad(y,et):
  with sdpa_kernel(SDPBackend.MATH):
   q=y.detach().requires_grad_(True);ell=sc.logits(q,et)[0,0];g=torch.autograd.grad(ell,q)[0]
  if not torch.isfinite(g).all():raise RuntimeError('nonfinite response')
  return float(ell),g.detach()
 def logits(y,e):
  with torch.no_grad():return sc.logits(y,e)[0].cpu().tolist()
 def img(y):
  with torch.no_grad():pixels=sc.image(y)[0].permute(1,2,0).cpu().numpy()
  return Image.fromarray((pixels*255).round().astype('uint8'))
 def canvas(pair,rec,columns,name):
  ims=[('Source',Image.open(ROOT/rec['source_image']).convert('RGB'))]+columns
  out=Image.new('RGB',(256*len(ims),300),'white');dr=ImageDraw.Draw(out)
  for i,(label,im) in enumerate(ims):out.paste(im.resize((256,256)),(i*256,35));dr.text((i*256+3,6),label,fill='black')
  dr.text((4,280),pair+' | '+name,fill='black');out.save(OUT/f'{pair}_{name}.png')
 if args.stage=='local':
  rows=[];tensors={}
  for pair in PAIRS:
   states=torch.load(Path(__file__).parent/'baseline_states'/f'{pair}.pt',weights_only=True)
   e,_=sc.encode([states['record']['target_prompt'],states['record']['source_prompt']])
   for s in states['snapshots']:
    step=s['step'];y=s['z'].to(device);d=(s['zE'].to(device)-y);ell,g=grad(y,e[:1]);W=weights(g)
    pg=torch.Generator('cpu').manual_seed(CONFIG['shuffle_seed']);perm=torch.randperm(W.numel(),generator=pg).to(device);shuffle=W.flatten()[perm].reshape_as(W)
    base=logits(y+d,e);columns=[('FlowEdit local',img(y+d)),('FlowEdit final',img(states['baseline_final'].to(device)))]
    for lam in CONFIG['local_lambdas']:
     for method,wgt,actual_lam in [('G',W,0.),('ISO',torch.full_like(W,float(W.mean())),lam),('ADAPT',W,lam),('SHUFFLE',shuffle,lam)]:
      u,info=solve(d,g,wgt,actual_lam,s['h']);new=y+u;actual=new-y;wactual=new-(y+d);vals=logits(new,e)
      obj=lambda v:.5*(v-d).square().sum()+.5*actual_lam*(wgt*v.square()).sum()-info['gamma']*(g/g.norm().clamp_min(1e-8)*v).sum()
      rows.append(dict(pair=pair,step=step,method=method,lam=actual_lam,nominal_lambda=lam,logit_before=ell,target_logit=vals[0],source_logit=vals[1],baseline_target_logit=base[0],delta_target_vs_baseline=vals[0]-base[0],predicted_target_gain=float((g*actual).sum()),actual_target_gain=vals[0]-ell,total_increment_ratio=float(actual.norm()/d.norm()),correction_ratio=float(wactual.norm()/d.norm()),proxy_objective_delta=float(obj(actual)-obj(d)),weight_mean=float(W.mean()),gradient_norm=float(g.norm()),**info))
      tensors[f'{pair}_{step}_{lam}_{method}']=new.cpu()
      if step==33 and lam==CONFIG['full_lambda']:columns.append((method+' local',img(new)))
    if step==33:canvas(pair,states['record'],columns,'local')
    (OUT/'local.json').write_text(json.dumps(dict(protocol=provenance,rows=rows),indent=2)+'\n');print(json.dumps(dict(pair=pair,step=step,complete=True)),flush=True)
  torch.save(tensors,OUT/'local_states.pt')
 else:
  sys.path.insert(0,str(ROOT/'vendor/flowedit'))
  from FlowEdit_utils import calc_v_sd3
  from diffusers import SD3Transformer2DModel
  model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
  pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
  meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text());times=torch.tensor(meta['timesteps'],device=device)
  results=[]
  for pair in PAIRS:
   payload=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);states=torch.load(Path(__file__).parent/'baseline_states'/f'{pair}.pt',weights_only=True)
   x=payload['x'].to(device);es,et=[tuple(t.to(device) for t in e) for e in payload['embeddings']]
   seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
   gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(device) for _ in range(33)]
   e,_=sc.encode([payload['record']['target_prompt'],payload['record']['source_prompt']]);columns=[];finals={}
   methods=(['BASE'] if pair=='bear__00' else [])+CONFIG['full_methods']
   for method in methods:
    start=time.monotonic();y=x.clone();rows=[]
    for k in range(17,50):
     t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn);qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
     with torch.no_grad():vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k]);d=-h*(va-vs)
     if method=='BASE':u=d;info={};ell=0.;g=None
     else:
      ell,g=grad(y,e[:1]);W=weights(g);W=torch.full_like(W,float(W.mean())) if method=='ISO' else W
      u,info=solve(d,g,W,0. if method=='G' else CONFIG['full_lambda'],h)
     new=y+u;wa=new-(y+d);effective=-((new-y)/h)
     rows.append(dict(step=k,t=float(t),h=h,optimization_calls=0 if method=='BASE' else 1,actual_correction_norm=float(wa.norm()),correction_ratio=float(wa.norm()/d.norm().clamp_min(1e-30)),increment_ratio=float((new-y).norm()/d.norm().clamp_min(1e-30)),gradient_norm=None if g is None else float(g.norm()),logit_before=None if g is None else ell,effective_velocity_norm=float(effective.norm()),finite=bool(torch.isfinite(new).all()),**info))
     if not rows[-1]['finite']:raise RuntimeError('nonfinite accepted state')
     y=new.detach()
     if k%10==7:print(json.dumps(dict(pair=pair,method=method,step=k)),flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-start;vals=logits(y,e)
    final_error=float((y.cpu().double()-states['baseline_final'].double()).norm()/states['baseline_final'].double().norm())
    summary=dict(pair=pair,method=method,target_logit=vals[0],source_logit=vals[1],seconds=seconds,steps=33,optimization_calls=sum(r['optimization_calls'] for r in rows),nonzero_corrections=sum(r['actual_correction_norm']>0 for r in rows),relative_to_baseline=final_error)
    results.append(summary);finals[method]=y.cpu();torch.save(dict(summary=summary,rows=rows,final=y.cpu(),protocol=provenance),OUT/f'{pair}_{method}_trajectory.pt')
    (OUT/'full.json').write_text(json.dumps(dict(protocol=provenance,results=results),indent=2)+'\n');print(json.dumps(summary),flush=True)
   columns=[('Original FlowEdit final',img(states['baseline_final'].to(device)))]+[(m+' every-step',img(finals[m].to(device))) for m in CONFIG['full_methods']]
   canvas(pair,payload['record'],columns,'full')
 print('EXPERIMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
