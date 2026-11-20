"""Cancel estimated transport while retaining generated appearance innovation.

Fixed-source or incremental DIS registration, raw/locally-normalized controls.
No edit-region mask, external reference or learned optical-flow model. Dense
flow is fallible under occlusion, 3D pose, texture changes and low contrast.
"""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from PIL import Image
from diffusers import SD3Transformer2DModel
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,Phase,seed_setup,write,save_image,panel
sys.path.insert(0,str(HERE.parent/'joint_permission'));from geometry import Geometry
sys.path.insert(0,str(HERE.parent/'phase_followup'));from dense_probe import prepare
sys.path.insert(0,str(ROOT/'vendor/flowedit'));from FlowEdit_utils import calc_v_sd3

def align(source,target,normalized=True,maxflow=16):
 a=cv2.resize(source,(256,256),interpolation=cv2.INTER_AREA);b=cv2.resize(target,(256,256),interpolation=cv2.INTER_AREA)
 u=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(prepare(a,normalized),prepare(b,normalized),None)
 flow=cv2.resize(u,(512,512),interpolation=cv2.INTER_LINEAR)*2
 flow=cv2.GaussianBlur(flow,(0,0),1)
 magnitude=np.linalg.norm(flow,axis=-1)
 flow=flow*np.minimum(1,maxflow/np.maximum(magnitude,1e-8))[:,:,None]
 yy,xx=np.mgrid[:512,:512].astype(np.float32)
 corrected=cv2.remap(target,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
 ux=np.gradient(flow[:,:,0],axis=1);uy=np.gradient(flow[:,:,0],axis=0)
 vx=np.gradient(flow[:,:,1],axis=1);vy=np.gradient(flow[:,:,1],axis=0)
 determinant=(1+ux)*(1+vy)-uy*vx
 return corrected,dict(flow_median=float(np.median(magnitude)),flow_p95=float(np.quantile(magnitude,.95)),
  clipped_fraction=float(np.mean(magnitude>maxflow)),negative_jacobian_fraction=float(np.mean(determinant<=0)))

def tensor(im,device='cpu'):return torch.tensor(np.ascontiguousarray(im),device=device).permute(2,0,1)[None]

def offline():
 rows=[];out=HERE/'outputs/transport_offline'
 for pair in ['bear__00','brown_owl__00']:
  source=np.array(Image.open(ROOT/'experiments/joint_permission/outputs/analysis'/f'{pair}_source.png'),np.float32)/255
  target=np.array(Image.open(ROOT/'experiments/phase_permission/outputs/full'/pair/'BASE/final.png'),np.float32)/255
  columns=[('Source',tensor(source)),('Original FlowEdit',tensor(target))]
  for norm in [False,True]:
   corrected,info=align(source,target,norm);columns.append((f'normalized={norm}',tensor(corrected)))
   rows.append(dict(pair=pair,normalized=norm,**info))
  panel(columns,out/f'{pair}.png',title='Offline diagnostic, not trajectory',size=320)
 write(out/'results.json',dict(rows=rows))

def main(args):
 seed_setup();cv2.setNumThreads(4);sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
 out=HERE/'outputs'/args.tag
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(sc.device)
 pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=sc.device)
 for pair in args.pairs:
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=data['x'].to(sc.device)
  es,et=[tuple(t.to(sc.device) for t in emb) for emb in data['embeddings']]
  seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
  with torch.no_grad():source=sc.image(x)
  src=source[0].permute(1,2,0).cpu().numpy();geom=Geometry(source);phase=Phase(source)
  def encode(im):
   z=sc.vae.encode(im*2-1).latent_dist.mode();return (z-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor
  with torch.no_grad():anchor=encode(source)
  cached=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
  for mode in args.modes:
   dest=out/pair/mode
   if (dest/'complete.json').exists():continue
   y=x.clone();states=[y.cpu()];previous=src.copy();rows=[];start=time.monotonic()
   gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(sc.device) for _ in range(33)]
   for k in range(17,50):
    t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
    with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
     vs,vt=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k]);raw=y-(t-tn)*(vt-vs)
    with torch.no_grad():rawim=sc.image(raw)[0].permute(1,2,0).cpu().numpy()
    reference=previous if mode=='STEP_NORM' else src
    if mode=='NO_WARP':corrected=rawim;info=dict(flow_median=0.,flow_p95=0.,clipped_fraction=0.,negative_jacobian_fraction=0.)
    else:corrected,info=align(reference,rawim,mode!='SOURCE_RAW')
    with torch.no_grad():
     if args.lift=='paired':
      c=encode(tensor(corrected,sc.device))-encode(tensor(rawim,sc.device))
      before=float(geom.loss(geom(tensor(rawim,sc.device)),40));accepted=False;attempts=[];y=raw
      if mode.endswith('_SAFE'):
       radius=2*float((raw-states[-1].to(sc.device)).norm());c=c*min(1.,radius/max(float(c.norm()),1e-20))
       for retry in range(4):
        alpha=.5**retry;candidate=raw+alpha*c;image=sc.image(candidate)
        value=float(geom.loss(geom(image),40));passed=value<=before+1e-8
        attempts.append(dict(alpha=alpha,geometry=value,passed=passed))
        if passed:y=candidate;accepted=True;break
      else:y=raw+c;accepted=True
      info.update(correction_norm=float((y-raw).norm()),accepted=accepted,geometry_before=before,attempts=attempts)
     else:y=x+encode(tensor(corrected,sc.device))-anchor
     previous=sc.image(y)[0].permute(1,2,0).cpu().numpy()
     if mode=='NO_WARP' and args.lift=='paired':
      info['baseline_error']=float((y-cached['states'][k+1].to(sc.device)).abs().max());assert info['baseline_error']<1e-6,info
    assert torch.isfinite(y).all();states.append(y.cpu());rows.append(dict(step=k,**info))
    if k in [17,25,33,41,49]:save_image(tensor(previous),dest/f'step{k}.png')
   im=tensor(previous,sc.device)
   with torch.no_grad():metrics=dict(geometry=float(geom.loss(geom(im),40)),rgb_rmse=float((im-source).square().mean().sqrt()),
    background_top64_rmse=float((im[:,:,:64]-source[:,:,:64]).square().mean().sqrt()),**phase.diagnostics(im))
   save_image(im,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
   write(dest/'complete.json',dict(pair=pair,mode=mode,rows=rows,endpoint=metrics,seconds=time.monotonic()-start,
    hypothesis='appearance-only; estimated source-to-proposal transport removed, not ground-truth pose control',max_flow_pixels=16,lift=args.lift))
   print(pair,mode,metrics,time.monotonic()-start,flush=True)
 print('TRANSPORT_COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--offline',action='store_true');ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00'])
 ap.add_argument('--modes',nargs='+',default=['SOURCE_NORM_SAFE'])
 ap.add_argument('--lift',choices=['anchored','paired'],default='paired');ap.add_argument('--tag');args=ap.parse_args()
 if args.tag is None:args.tag='transport_paired' if args.lift=='paired' else 'transport_full'
 with sdpa_kernel(SDPBackend.MATH):
  if args.offline:offline()
  else:main(args)
