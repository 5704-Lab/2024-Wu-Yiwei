"""Factorized SD3 joint attention: source image relations, target text route.

No spatial editing masks. Local target image/text probability mass and text
contribution are retained at the intervention, but later layers can change them.
Source branches are unmodified. Not claimed to guarantee semantic correctness.
"""
import sys,json,time,math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel,AutoencoderKL
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,write,save_image,seed_setup,PAIRS
sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from FlowEdit_utils import calc_v_sd3

class Routing:
 def __init__(self,original):self.original=original;self.strength=0.;self.source_values=0.;self.last={}
 def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,*args,**kwargs):
  if not self.strength:return self.original(attn,hidden_states,encoder_hidden_states,attention_mask,*args,**kwargs)
  b,n,_=hidden_states.shape
  if b!=4 or encoder_hidden_states is None:raise RuntimeError('Routing requires source U,C and target U,C batch order')
  def heads(z):return z.view(b,-1,attn.heads,z.shape[-1]//attn.heads).transpose(1,2)
  q,k,v=[heads(f(hidden_states)) for f in [attn.to_q,attn.to_k,attn.to_v]]
  if attn.norm_q is not None:q=attn.norm_q(q)
  if attn.norm_k is not None:k=attn.norm_k(k)
  tq,tk,tv=[heads(f(encoder_hidden_states)) for f in [attn.add_q_proj,attn.add_k_proj,attn.add_v_proj]]
  if attn.norm_added_q is not None:tq=attn.norm_added_q(tq)
  if attn.norm_added_k is not None:tk=attn.norm_added_k(tk)
  query=torch.cat([q,tq],2);key=torch.cat([k,tk],2);value=torch.cat([v,tv],2)
  result=F.scaled_dot_product_attention(query,key,value,dropout_p=0.,is_causal=False)
  scale=1/math.sqrt(q.shape[-1])
  source_logits=(q[:2]@k[:2].transpose(-1,-2))*scale
  target_logits=(q[2:]@k[2:].transpose(-1,-2))*scale
  text_logits=(q[2:]@tk[2:].transpose(-1,-2))*scale
  log_i=torch.logsumexp(target_logits,-1,keepdim=True)
  log_t=torch.logsumexp(text_logits,-1,keepdim=True)
  mass=torch.sigmoid(log_i-log_t)
  ps=torch.softmax(source_logits,-1);pt=torch.softmax(target_logits,-1)
  chosen_v=(1-self.source_values)*v[2:]+self.source_values*v[:2]
  correction=self.strength*mass*(ps@chosen_v-pt@v[2:])
  result=result.clone();result[2:,:,:n]+=correction
  self.last=dict(image_mass_mean=float(mass.mean()),correction_rms=float(correction.square().mean().sqrt()))
  result=result.transpose(1,2).reshape(b,-1,attn.heads*q.shape[-1]).to(q.dtype)
  image,text=result[:,:n],result[:,n:]
  if not attn.context_pre_only:text=attn.to_add_out(text)
  image=attn.to_out[1](attn.to_out[0](image))
  return image,text

def main():
 seed_setup();device=torch.device('cuda:0');out=HERE/'outputs/routing'
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
 layers=[6,9,12,15];processors=[]
 for k in layers:
  a=model.transformer_blocks[k].attn;p=Routing(a.processor);a.set_processor(p);processors.append(p)
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=device)
 write(out/'protocol.json',dict(layers=layers,variants=dict(P_ONLY=dict(strength=.75,source_values=0),P_SOURCE_V=dict(strength=.75,source_values=.5)),
    scope='free-caption-native branch, no automatic fact parser; shape changes are stress tests'))
 for pair in PAIRS:
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=data['x'].to(device)
  es,et=[tuple(t.to(device) for t in emb) for emb in data['embeddings']]
  seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
  gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(device) for _ in range(33)]
  saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
  for method,sv in [('P_ONLY',0.),('P_SOURCE_V',.5)]:
   dest=out/pair/method;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'complete.json').exists():continue
   start=time.monotonic();rows=[];states=[x.cpu()];y=x.clone()
   for p in processors:p.strength=.75;p.source_values=sv
   for k in range(17,50):
    t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn)
    qs=(1-t)*x+t*noise[k-17];qa=qs+y-x;inputs=torch.cat([qs,qs,qa,qa])
    with torch.no_grad():
     if k==17:
      for p in processors:p.strength=0
      ref_s,ref_t=calc_v_sd3(pipe,inputs,seq,pool,3.5,13.5,times[k])
      for p in processors:p.strength=.75
     vs,vt=calc_v_sd3(pipe,inputs,seq,pool,3.5,13.5,times[k])
     if k==17:
      source_error=float((vs-ref_s).abs().max());assert source_error<1e-6
     y=y-h*(vt-vs)
    assert torch.isfinite(y).all()
    rows.append(dict(step=k,source_error_first=source_error,attention=[p.last for p in processors]))
    states.append(y.cpu())
    if k in [17,33,49]:
     with torch.no_grad():image=((vae.decode(y/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
     save_image(image,dest/f'step{k}.png')
    if k%8==1:print(pair,method,k,time.monotonic()-start,flush=True)
   save_image(image,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
   write(dest/'complete.json',dict(pair=pair,method=method,rows=rows,seconds=time.monotonic()-start))
 print('ROUTING_COMPLETE',flush=True)
if __name__=='__main__':main()
