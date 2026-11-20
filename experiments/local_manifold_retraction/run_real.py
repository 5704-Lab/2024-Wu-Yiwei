"""Shared-prior low-noise retraction and double-scale diagnostic editing."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
from diffusers import SD3Transformer2DModel,AutoencoderKL

ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;OUT=BASE/'outputs';OUT.mkdir(exist_ok=True)
PAIRS=['bear__00','brown_owl__00','bus__00','cat_crown__00']
CONFIG=dict(sigmas=[.025,.05,.1],edit_sigmas=[.025,.05],beta=.05,eta=1.,geometry_seed=20261009,geometry_repeat_seed=20261010,generation_seed=0,geometry_cfg=1.,geometry_condition='cached empty prompt',antithetic_pairs=1,full_methods=['SINGLE','DUAL'],precision='float32',source_cfg=3.5,target_cfg=13.5,optimization_every_active_step=True)

def project(r,radius):
 norm=float(r.norm());return r*min(1.,radius/max(norm,1e-30))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['local','full'],required=True);args=ap.parse_args();start_all=time.monotonic()
 if not json.loads((OUT/'toy.json').read_text())['passed']:raise RuntimeError('toy gate not passed')
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0')
 protocol=dict(config=CONFIG,stage=args.stage,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=str(torch.__version__),scope=f'{len(PAIRS)} development sources; no true manifold or independent quality oracle')
 (OUT/f'protocol_{args.stage}.json').write_text(json.dumps(protocol,indent=2)+'\n')
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 def geom(y,eps,cond,sigmas):
  qs=[];ts=[]
  for sigma in sigmas:
   tau=sigma/(1+sigma);alpha=1/(1+sigma)
   for sign in [1.,-1.]:qs.append(alpha*y+sign*tau*eps);ts.append(1000*tau)
  seq,pool=cond;n=len(qs)
  with torch.no_grad():
   q=torch.cat(qs);tt=torch.tensor(ts,device=device,dtype=y.dtype)
   v=model(hidden_states=q,timestep=tt,encoder_hidden_states=seq.expand(n,-1,-1),pooled_projections=pool.expand(n,-1),joint_attention_kwargs=None,return_dict=False)[0]
   m=q-tt[:,None,None,None]/1000*v
  if not torch.isfinite(m).all():raise RuntimeError('nonfinite geometric map')
  return [m[2*i:2*i+2].mean(dim=0,keepdim=True) for i in range(len(sigmas))]
 def image(y):
  with torch.no_grad():p=((vae.decode(y/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
  return Image.fromarray((p[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))
 def canvas(pair,rec,cols,name):
  columns=[('Source RGB',Image.open(ROOT/rec['source_image']).convert('RGB'))]+cols
  out=Image.new('RGB',(256*len(columns),300),'white');dr=ImageDraw.Draw(out)
  for i,(label,im) in enumerate(columns):out.paste(im.resize((256,256)),(256*i,35));dr.text((256*i+4,7),label,fill='black')
  dr.text((4,282),pair+' | '+name,fill='black');out.save(OUT/f'{pair}_{name}.png')
 def cosine(a,b):return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
 results=[];saved={}
 if args.stage=='local':
  for pair in PAIRS:
   inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);states=torch.load(ROOT/'experiments/siglip_state_response/baseline_states'/f'{pair}.pt',weights_only=True);es,et=[tuple(t.to(device) for t in e) for e in inp['embeddings']]
   conds=dict(EMPTY=(es[1],es[3]),SOURCE=(es[0],es[2]),TARGET=(et[0],et[2]));x=inp['x'].to(device)
   points=[('source',x),('step33',next(s for s in states['snapshots'] if s['step']==33)['z'].to(device)),('step47',next(s for s in states['snapshots'] if s['step']==47)['z'].to(device)),('final',states['baseline_final'].to(device))]
   gen=torch.Generator('cpu').manual_seed(CONFIG['geometry_seed']);eps=torch.randn(x.shape,generator=gen).to(device)
   for label,y in points:
    for cname in (['EMPTY','SOURCE','TARGET'] if label=='step33' else ['EMPTY']):
     started=time.monotonic();f=geom(y,eps,conds[cname],CONFIG['sigmas']);dual=(4*f[0]-f[1])/3;dual_hi=(4*f[1]-f[2])/3;d12=f[1]-f[0];d23=f[2]-f[1]
     denom=lambda scale:float(d23.norm()+scale*d12.norm())
     row=dict(pair=pair,point=label,condition=cname,single_residual_norm=float((f[0]-y).norm()),dual_residual_norm=float((dual-y).norm()),single_dual_cosine=cosine(f[0]-y,dual-y),first_order_residual=float((d23-2*d12).norm())/max(denom(2),1e-30),second_order_residual=float((d23-4*d12).norm())/max(denom(4),1e-30),difference_norm_ratio=float(d23.norm()/d12.norm().clamp_min(1e-30)),difference_cosine=cosine(d12,d23),dual_scale_difference=float((dual_hi-dual).norm()),dual_scale_relative_difference=float((dual_hi-dual).norm()/(dual-y).norm().clamp_min(1e-30)),seconds=time.monotonic()-started)
     if label=='step33' and cname=='EMPTY':
      g2=torch.Generator('cpu').manual_seed(CONFIG['geometry_repeat_seed']);eps2=torch.randn(x.shape,generator=g2).to(device);f2=geom(y,eps2,conds[cname],CONFIG['sigmas']);dual2=(4*f2[0]-f2[1])/3
      row.update(repeat_dual_cosine=cosine(dual-y,dual2-y),repeat_dual_relative_difference=float((dual2-dual).norm()/(dual-y).norm().clamp_min(1e-30)))
     results.append(row);saved[f'{pair}_{label}_{cname}']=dict(y=y.cpu(),F=[t.cpu() for t in f],dual=dual.cpu(),dual_high=dual_hi.cpu())
     print(json.dumps(row),flush=True)
    (OUT/'local.json').write_text(json.dumps(dict(protocol=protocol,rows=results),indent=2)+'\n')
   y=points[1][1];d=next(s for s in states['snapshots'] if s['step']==33)['zE'].to(device)-y
   loc=saved[f'{pair}_step33_EMPTY'];canvas(pair,inp['record'],[('Original FlowEdit step33',image(y)),('Original FlowEdit final',image(states['baseline_final'].to(device))),('Single raw retraction',image(loc['F'][0].to(device))),('Dual raw retraction',image(loc['dual'].to(device)))],'local_diagnostic')
  torch.save(saved,OUT/'local_states.pt')
 else:
  sys.path.insert(0,str(ROOT/'vendor/flowedit'));from FlowEdit_utils import calc_v_sd3
  pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True);meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text());times=torch.tensor(meta['timesteps'],device=device)
  for pair in PAIRS:
   inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);states=torch.load(ROOT/'experiments/siglip_state_response/baseline_states'/f'{pair}.pt',weights_only=True);x=inp['x'].to(device);es,et=[tuple(t.to(device) for t in e) for e in inp['embeddings']]
   seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]]);cond=(es[1],es[3])
   rng=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=rng).to(device) for _ in range(33)]
   aux=torch.Generator('cpu').manual_seed(CONFIG['geometry_seed']);auxnoise=[torch.randn(x.shape,generator=aux).to(device) for _ in range(33)]
   finals={};methods=(['BASE'] if pair=='bear__00' else [])+CONFIG['full_methods']
   for method in methods:
    started=time.monotonic();y=x.clone();rows=[]
    for k in range(17,50):
     t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn);qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
     with torch.no_grad():vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k]);d=-h*(va-vs);z=y+d
     R=CONFIG['beta']*float(d.norm())
     if method=='BASE':r=torch.zeros_like(d);w=r;geom_samples=0
     else:
      fs=geom(z,auxnoise[k-17],cond,CONFIG['edit_sigmas'] if method=='DUAL' else CONFIG['edit_sigmas'][:1]);retract=(4*fs[0]-fs[1])/3 if method=='DUAL' else fs[0];r=retract-z;w=project(r,R);geom_samples=len(fs)*2
     yn=z+w;wa=yn-z;actual_d=yn-y;finite=bool(torch.isfinite(yn).all())
     if not finite:raise RuntimeError('nonfinite accepted state')
     rows.append(dict(step=k,t=float(t),h=h,optimization_calls=int(method!='BASE'),geometry_samples=geom_samples,raw_retraction_norm=float(r.norm()),actual_correction_norm=float(wa.norm()),correction_ratio=float(wa.norm()/d.norm().clamp_min(1e-30)),increment_ratio=float(actual_d.norm()/d.norm().clamp_min(1e-30)),correction_base_cosine=cosine(wa,d),radius=R,capped=bool(float(r.norm())>R),finite=finite))
     if float(wa.norm())>R+1e-4:raise RuntimeError('actual correction exceeded budget tolerance')
     y=yn.detach()
     if k%10==7:print(json.dumps(dict(pair=pair,method=method,step=k)),flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-started;err=float((y.cpu().double()-states['baseline_final'].double()).norm()/states['baseline_final'].double().norm())
    result=dict(pair=pair,method=method,seconds=seconds,steps=33,optimization_calls=sum(r['optimization_calls'] for r in rows),nonzero_corrections=sum(r['actual_correction_norm']>0 for r in rows),geometry_samples=sum(r['geometry_samples'] for r in rows),relative_to_baseline=err,mean_increment_ratio=float(np.mean([r['increment_ratio'] for r in rows])),mean_correction_ratio=float(np.mean([r['correction_ratio'] for r in rows])),capped_steps=sum(r['capped'] for r in rows),all_finite=all(r['finite'] for r in rows))
    results.append(result);finals[method]=y.cpu();torch.save(dict(protocol=protocol,record=inp['record'],summary=result,rows=rows,final=y.cpu()),OUT/f'{pair}_{method}_trajectory.pt');image(y).save(OUT/f'{pair}_{method}_512.png')
    (OUT/'full.json').write_text(json.dumps(dict(protocol=protocol,results=results),indent=2)+'\n');print(json.dumps(result),flush=True)
   baseline=states['baseline_final'].to(device);image(baseline).save(OUT/f'{pair}_FlowEdit_512.png');image(x).save(OUT/f'{pair}_SourceVAE_512.png')
   canvas(pair,inp['record'],[('Source VAE',image(x)),('Original FlowEdit final',image(baseline))]+[(m+' every-step',image(finals[m].to(device))) for m in CONFIG['full_methods']],'full_comparison')
 (OUT/f'time_{args.stage}.json').write_text(json.dumps(dict(seconds_including_load=time.monotonic()-start_all),indent=2)+'\n')
 print('EXPERIMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
