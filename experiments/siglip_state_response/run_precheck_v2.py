"""Real clean-state SigLIP chain; independent from old velocity VJP."""
import argparse,gc,hashlib,json,math,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from diffusers import AutoencoderKL
from transformers import AutoModel,AutoProcessor
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

class CleanScorer:
 def __init__(self,device,dtype,resize_mode,split=False):
  self.device=device;self.dtype=dtype;self.resize_mode=resize_mode;self.weights={}
  self.vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=dtype,local_files_only=True).eval().requires_grad_(False).to(device)
  self.model_device=torch.device('cuda:1') if split else device
  self.model=AutoModel.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',torch_dtype=dtype,local_files_only=True,attn_implementation='eager').eval().requires_grad_(False).to(self.model_device)
  self.processor=AutoProcessor.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',local_files_only=True)
  self.alpha=self.model.logit_scale.exp().detach();self.bias=self.model.logit_bias.detach()
 def encode(self,texts):
  tokens=self.processor(text=texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(self.model_device)
  with torch.no_grad():e=self.model.get_text_features(**tokens);e=e/e.norm(dim=-1,keepdim=True)
  lengths=[len(self.processor.tokenizer(t,add_special_tokens=True,truncation=False)['input_ids']) for t in texts]
  return e.detach(),lengths
 def image(self,y):
  raw=self.vae.decode(y/self.vae.config.scaling_factor+self.vae.config.shift_factor,return_dict=False)[0]
  return ((raw+1)/2).clamp(0,1)
 def linear_weight(self,n):
  if n not in self.weights:
   basis=torch.eye(n,device=self.device,dtype=self.dtype).reshape(n,1,1,n)
   self.weights[n]=F.interpolate(basis,size=(1,256),mode='bilinear',align_corners=False,antialias=True)[:,0,0,:].T.detach()
  return self.weights[n]
 def preprocess(self,image):
  if self.resize_mode=='native':
   pixels=F.interpolate(image,size=(256,256),mode='bilinear',align_corners=False,antialias=True)
  else:
   wh=self.linear_weight(image.shape[-2]);ww=self.linear_weight(image.shape[-1])
   pixels=torch.matmul(torch.matmul(wh,image),ww.T)
  return (pixels-.5)/.5
 def logits_image(self,image,e):
  pixels=self.preprocess(image)
  f=self.model.get_image_features(pixel_values=pixels.to(self.model_device));f=f/f.norm(dim=-1,keepdim=True)
  return (self.alpha*(f@e.T)+self.bias).to(self.device)
 def logits(self,y,e):return self.logits_image(self.image(y),e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--dtype',choices=['float32','float64'],default='float32');ap.add_argument('--resize',choices=['native','linear'],default='native');ap.add_argument('--pair',default='bear__00');ap.add_argument('--step',type=int,default=33);ap.add_argument('--hvp',action='store_true');ap.add_argument('--split',action='store_true');args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(args.device);dtype=getattr(torch,args.dtype);start=time.monotonic()
 inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{args.pair}.pt',weights_only=True)
 saved=torch.load(ROOT/'experiments/stage1_full/outputs'/f'{args.pair}__seed0'/'states.pt',weights_only=True);snap=next(s for s in saved['snapshots'] if s['step']==args.step)
 y=snap['z'].to(device=device,dtype=dtype);yb=snap['zE'].to(device=device,dtype=dtype)
 sc=CleanScorer(device,dtype,args.resize,args.split)
 rec=inp['record'];e,lengths=sc.encode([rec['target_prompt'],rec['source_prompt']]);et=e[:1]
 result=dict(pair=args.pair,step=args.step,dtype=args.dtype,resize=args.resize,two_gpu_pipeline=args.split,clean_state=True,weights_frozen=True,full_prompt_token_lengths=lengths,text_truncated=[n>64 for n in lengths],alpha=float(sc.alpha),bias=float(sc.bias),baseline='retained original FlowEdit snapshot',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 def scoregrad(point,graph=False):
  q=point.detach().requires_grad_(True);ell=sc.logits(q,et)[0,0];g=torch.autograd.grad(ell,q,create_graph=graph)[0]
  return q,ell,g
 try:
  with sdpa_kernel(SDPBackend.MATH):
   # Forward compatibility audit: same decoded image, no generated editing assets.
   with torch.no_grad():
    im=sc.image(yb);px=sc.preprocess(im);tokens=sc.processor(text=[rec['target_prompt']],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(sc.model_device)
    official=sc.model(pixel_values=px.to(sc.model_device),**tokens).logits_per_image[0,0].to(device);manual=sc.logits_image(im,et)[0,0]
    pil=Image.fromarray((im[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype('uint8'))
    pilpx=sc.processor(images=pil,return_tensors='pt')['pixel_values'].to(device=device,dtype=dtype)
    pfl=sc.model.get_image_features(pixel_values=pilpx.to(sc.model_device));pfl=pfl/pfl.norm(dim=-1,keepdim=True)
    pillogit=(sc.alpha*(pfl@et.T)+sc.bias)[0,0].to(device)
    native=F.interpolate(im,size=(256,256),mode='bilinear',align_corners=False,antialias=True)
    wh=sc.linear_weight(im.shape[-2]);ww=sc.linear_weight(im.shape[-1]);linear=torch.matmul(torch.matmul(wh,im),ww.T)
    result.update(manual_official_logit_error=float(abs(manual-official)),tensor_pil_logit_delta=float(manual-pillogit),tensor_pil_pixel_max_error=float((px-pilpx).abs().max()),linear_native_pixel_max_error=float((linear-native).abs().max()))
   _,ellr,gr=scoregrad(y);ur=(gr/gr.norm()).detach();del gr
   q,ell,g=scoregrad(yb,args.hvp);u=g/g.norm()
   result.update(logit_previous=float(ellr.detach()),logit_candidate=float(ell.detach()),gradient_norm=float(g.detach().norm()),direction_cosine=float((ur*u).sum().detach()))
   gd=g.detach()/g.detach().norm();ell0=float(ell.detach());gn=float(g.detach().norm())
   b=None
   if args.hvp:
    try:
     align=(ur*u).sum();b=torch.autograd.grad(align,q,retain_graph=True)[0]
     zeta=((ur-u.detach()*(ur*u.detach()).sum())/g.detach().norm()).detach()
     bm=torch.autograd.grad((g*zeta).sum(),q)[0]
     result.update(hvp_supported=True,hvp_chain_error=float((b-bm).norm()/(1+b.norm())),direction_gradient_norm=float(b.norm()))
    except Exception:
     result.update(hvp_supported=False,hvp_error=traceback.format_exc());b=None
   result['all_finite']=bool(torch.isfinite(g).all())
   del q,ell,g,u;gc.collect();torch.cuda.empty_cache()
   dnorm=float((yb-y).norm());trials=[]
   for rel in [.05,.025,.005]:
    w=rel*dnorm*gd
    with torch.no_grad():new=float(sc.logits(yb+w,et)[0,0])
    predicted=gn*float(w.norm())
    trials.append(dict(relative_to_original_update=rel,logit_gain=new-ell0,predicted_gain=predicted,first_order_ratio=(new-ell0)/predicted))
   result['semantic_trials']=trials
   # Independent derivative finite steps. These do not perturb the text.
   diffs=[]
   for rel in [1e-2,1e-3,1e-4,1e-5,1e-6]:
    eps=rel*dnorm
    with torch.no_grad():plus=float(sc.logits(yb+eps*gd,et)[0,0]);minus=float(sc.logits(yb-eps*gd,et)[0,0])
    fd=(plus-minus)/(2*eps);diffs.append(dict(relative_to_original_update=rel,automatic_derivative=gn,central_derivative=fd,relative_error=abs(fd-gn)/max(gn,1e-30)))
   result['state_derivative_checks']=diffs
   if b is not None:
    bn=float(b.norm());c=float((gd*b).sum()/bn) if bn else None;result['objective_gradient_cosine']=c
    if c is not None and 1+c>=.05:
     pd=gd+b/bn;pd=pd/pd.norm();aL=float((gd*pd).sum())*gn;aF=float((b*pd).sum())
     R=.05*dnorm
     maxell=float(sc.alpha+sc.bias);f0=result['direction_cosine']
     R=min(R,.25*max(0.,maxell-ell0)/aL,.25*max(0.,1-f0)/aF)
     joint=[]
     for factor in [1.,.5]:
      w=factor*R*pd
      _,elnew,gnew=scoregrad(yb+w);fnew=float((ur*gnew/gnew.norm()).sum());dl=float(elnew)-ell0;df=fnew-f0
      joint.append(dict(factor=factor,norm=float(w.norm()),logit_gain=dl,alignment_gain=df,predicted_logit_gain=aL*factor*R,predicted_alignment_gain=aF*factor*R,dual_armijo_without_margin=dl>=.1*aL*factor*R and df>=.1*aF*factor*R))
      del elnew,gnew
     result['joint_trials']=joint
   # Semantic audit keeps original whole-prompt definition.
   x=inp['x'].to(device=device,dtype=dtype)
   with torch.no_grad():
    sourceimage=sc.image(x);brightness=[]
    for factor in [.7,1.,1.3]:
     logs=sc.logits_image((factor*sourceimage).clamp(0,1),e)[0]
     brightness.append(dict(factor=factor,target_logit=float(logs[0]),source_logit=float(logs[1]),target_minus_source=float(logs[0]-logs[1])))
   result['brightness_diagnostic']=brightness
   torch.save(dict(semantic_direction=gd.cpu(),direction_correction=None if b is None else b.detach().cpu(),reference_direction=ur.cpu()),OUT/f'{args.pair}_step{args.step}_{args.dtype}_{args.resize}_tensors.pt')
 except Exception:result.update(error=traceback.format_exc())
 result.update(elapsed_seconds=time.monotonic()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device))
 save(f'{args.pair}_step{args.step}_{args.dtype}_{args.resize}'+('_hvp' if args.hvp else '')+'.json',result)
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
