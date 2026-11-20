"""Source-coordinate appearance family as a finite latent-state constraint.

Pure-colour task hypothesis only. No spatial editing masks. Every coordinate is
processed by the same bounded smooth RGB gain family. Decoder reconstruction
can still change edges, so image geometry must be checked rather than assumed.
"""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,write,save_image,seed_setup
from control import Run
from FlowEdit_utils import calc_v_sd3

CONFIG=dict(order=4,constant_bound=.8,spatial_sum_bound=.8,constant_step=.08,spatial_step=.012,
            projection_iterations=12,prior=.005,rf_tolerance=2e-6,backtracks=4,
            hypothesis='pure appearance; source-coordinate smooth log RGB gain, no geometric warp')

def main():
 seed_setup();run=Run();out=HERE/'outputs/manifold'
 write(out/'protocol.json',dict(config=CONFIG,pairs=['bear__00','brown_owl__00'],
       status='exploratory stronger constraint after weak geometry control failed; not universal text parsing'))
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=run.device)
 for pair in ['bear__00','brown_owl__00']:
  ctx=run.setup(pair);x=ctx['x'];source=ctx['source'];n=source.shape[-1];order=CONFIG['order'];m=order**2
  grid=(torch.arange(n,device=x.device,dtype=x.dtype)+.5)/n
  basis=torch.stack([torch.cos(np.pi*i*grid)[:,None]*torch.cos(np.pi*j*grid)[None,:] for i in range(order) for j in range(order)])
  bounds=np.tile([CONFIG['constant_bound']]+[CONFIG['spatial_sum_bound']/(m-1)]*(m-1),3)
  steps=np.tile([CONFIG['constant_step']]+[CONFIG['spatial_step']]*(m-1),3)
  def encode(image):
   z=run.sc.vae.encode(image*2-1).latent_dist.mode()
   return (z-run.sc.vae.config.shift_factor)*run.sc.vae.config.scaling_factor
  with torch.no_grad():encoded_source=encode(source)
  def state(theta):
   field=torch.einsum('cm,mhw->chw',theta.reshape(3,m),basis)[None]
   image=(source*field.exp()).clamp(0,1)
   return x+encode(image)-encoded_source
  with torch.no_grad():zero_error=float((state(torch.zeros(3*m,device=x.device))-x).abs().max())
  assert zero_error<2e-6
  for mode in ['PROJECT','PROJECT_RF']:
   dest=out/pair/mode;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'complete.json').exists():continue
   y=x.clone();theta=np.zeros(3*m);rows=[];states=[y.cpu()];start=time.monotonic()
   previous=run.rf(ctx,y)
   gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(x.device) for _ in range(33)]
   for k in range(17,50):
    t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn)
    qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
    with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
     vs,va=calc_v_sd3(run.pipe,torch.cat([qs,qs,qa,qa]),ctx['seq'],ctx['pool'],3.5,13.5,times[k])
     delta=-h*(va-vs);proposal=y+delta
    prior=theta.copy();scale=max(float(delta.square().sum()),1e-6)
    def objective(a):
     p=torch.tensor(a,device=x.device,dtype=x.dtype,requires_grad=True)
     z=state(p)
     loss=.5*(z-proposal).square().sum()/scale+CONFIG['prior']*(p-torch.tensor(prior,device=x.device,dtype=x.dtype)).square().sum()
     g,=torch.autograd.grad(loss,p)
     return float(loss),g.cpu().double().numpy()
    lo=np.maximum(-bounds,prior-steps);hi=np.minimum(bounds,prior+steps)
    opt=minimize(objective,theta,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(maxiter=CONFIG['projection_iterations'],ftol=1e-8,gtol=1e-5))
    accepted=False;trials=[];new=y;newtheta=prior
    for retry in range(1 if mode=='PROJECT' else CONFIG['backtracks']):
     alpha=.5**retry;candidate_theta=prior+alpha*(opt.x-prior)
     with torch.no_grad():candidate=state(torch.tensor(candidate_theta,device=x.device,dtype=x.dtype))
     fact=run.rf(ctx,candidate)
     passed=mode=='PROJECT' or bool(np.all(fact>=previous-CONFIG['rf_tolerance']))
     trials.append(dict(alpha=alpha,rf=fact.tolist(),passed=passed))
     if passed:new=candidate.detach();newtheta=candidate_theta;previous=fact;accepted=True;break
    theta=newtheta;y=new.detach();states.append(y.cpu())
    rows.append(dict(step=k,accepted=accepted,projection_converged=bool(opt.success),projection_iterations=int(opt.nit),
        projection_residual=float((y-proposal).norm()),base_increment_norm=float(delta.norm()),
        increment_norm=float((y-states[-2].to(y.device)).norm()),theta=theta.tolist(),rf=previous.tolist(),trials=trials))
    write(dest/'progress.json',dict(rows=rows,source_zero_error=zero_error))
    if k in [17,25,33,41,49]:
     with torch.no_grad():save_image(run.sc.image(y),dest/f'step{k}.png')
    print(pair,mode,k,accepted,previous.tolist(),time.monotonic()-start,flush=True)
   with torch.no_grad():
    im=run.sc.image(y);logs=run.sc.logits_image(im,ctx['e'])[0].cpu().tolist()
    geometry=float(ctx['geometry'].loss(ctx['geometry'](im),40.))
   save_image(im,dest/'final.png')
   torch.save(dict(states=torch.stack(states),final=y.cpu(),theta=torch.tensor(theta)),dest/'trajectory.pt')
   write(dest/'complete.json',dict(rows=rows,config=CONFIG,source_zero_error=zero_error,seconds=time.monotonic()-start,
          full_target=logs[0],full_source=logs[1],short_target=logs[2],short_source=logs[3],geometry=geometry,
          final_rf=previous.tolist(),zero_step_count=sum(r['increment_norm']<1e-8 for r in rows)))
 print('MANIFOLD_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
