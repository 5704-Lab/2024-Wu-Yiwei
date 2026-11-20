"""Ten direct clean-latent SigLIP target-gradient moves, without generation."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from torch.nn.attention import sdpa_kernel,SDPBackend

ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;OUT=BASE/'outputs';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'experiments/siglip_state_response'));from run_precheck import CleanScorer
ap=argparse.ArgumentParser();ap.add_argument('--step-norm',type=float,default=.25);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
if args.step_norm<=0:raise ValueError('step-norm must be positive')
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
PAIRS=['bear__00','brown_owl__00','bus__00','cat_crown__00'];device=torch.device(args.device)
protocol=dict(pairs=PAIRS,steps=10,step_l2_norm=args.step_norm,gradient='grad_y SigLIP2 target logit',normalized=True,recompute_each_step=True,initial='cached SD3 VAE source posterior-mode latent x0',generation_velocity_used=False,noising_used=False,parameters_frozen=True,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');sc=CleanScorer(device,torch.float32,'native');all_data=[];records=[]
def image(y):
 with torch.no_grad():pixels=sc.image(y)[0].permute(1,2,0).cpu().numpy()
 return Image.fromarray(np.clip(np.rint(pixels*255),0,255).astype('uint8'))
with sdpa_kernel(SDPBackend.MATH):
 for pair in PAIRS:
  started=time.monotonic();dest=OUT/pair;dest.mkdir(exist_ok=True);inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);rec=inp['record'];x=inp['x'].to(device);y=x.clone();e,lengths=sc.encode([rec['target_prompt'],rec['source_prompt']]);ims=[];rows=[];states=[x.cpu()]
  with torch.no_grad():initial_logits=sc.logits(y,e)[0].cpu().tolist()
  source=image(x);source.save(dest/'source_vae.png')
  for i in range(1,11):
   q=y.detach().requires_grad_(True);ell=sc.logits(q,e[:1])[0,0];g=torch.autograd.grad(ell,q)[0].detach();norm=float(g.norm())
   if not torch.isfinite(g).all() or norm<=1e-8:raise RuntimeError('unresolved semantic gradient')
   yn=(y+args.step_norm*g/norm).detach()
   with torch.no_grad():vals=sc.logits(yn,e)[0].cpu().tolist()
   im=image(yn);im.save(dest/f'move_{i:02d}.png');ims.append(im);states.append(yn.cpu())
   rows.append(dict(move=i,target_logit_before=float(ell.detach()),target_logit_after=vals[0],source_logit_after=vals[1],gradient_norm=norm,actual_step_norm=float((yn-y).norm()),distance_to_source_latent=float((yn-x).norm()),predicted_logit_gain=args.step_norm*norm,actual_logit_gain=vals[0]-float(ell.detach())))
   y=yn
  record=dict(pair=pair,record=rec,protocol=protocol,initial_target_logit=initial_logits[0],initial_source_logit=initial_logits[1],text_token_lengths=lengths,text_truncated=[v>64 for v in lengths],rows=rows,seconds=time.monotonic()-started)
  (dest/'metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');torch.save(dict(states=torch.stack(states),record=rec),dest/'states.pt');records.append(record)
  old=Image.open(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'frames/step_49.png').convert('RGB');cfg1=Image.open(ROOT/'experiments/flowedit_frequency_trajectory/outputs_cfg1'/pair/'frames/step_49.png').convert('RGB')
  all_data.append(dict(pair=pair,record=rec,source_rgb=Image.open(ROOT/rec['source_image']).convert('RGB'),source_vae=source,flowedit=old,flowedit_cfg1=cfg1,ims=ims,rows=rows));print(json.dumps(dict(pair=pair,initial_target_logit=initial_logits[0],final_target_logit=rows[-1]['target_logit_after'],seconds=record['seconds'])),flush=True)

font=ImageFont.truetype('DejaVuSans.ttf',15);small=ImageFont.truetype('DejaVuSans.ttf',12);size=192;stride=size+112;canvas=Image.new('RGB',(14*size,len(all_data)*stride+40),'white');draw=ImageDraw.Draw(canvas)
for j,a in enumerate(all_data):
 top=j*stride;draw.text((5,top+3),a['pair']+' | Recomputed target gradient; L2 step='+str(args.step_norm),font=font,fill='black');draw.text((5,top+27),'SOURCE: '+a['record']['source_prompt'],font=small,fill='#965400');draw.text((5,top+45),'TARGET: '+a['record']['target_prompt'],font=small,fill='#005b99')
 columns=[('Source RGB',a['source_rgb']),('Source VAE / x0',a['source_vae']),('FlowEdit CFG3.5/13.5',a['flowedit']),('FlowEdit CFG1/1',a['flowedit_cfg1'])]+[(f"Move {i+1} | logit {a['rows'][i]['target_logit_after']:.2f}",im) for i,im in enumerate(a['ims'])]
 for i,(label,im) in enumerate(columns):draw.text((i*size+2,top+68),label,font=small,fill='black');canvas.paste(im.resize((size,size)),(i*size,top+87))
draw.text((5,len(all_data)*stride+8),'Direct VAE -> SigLIP gradient ascent; no diffusion/FlowEdit update. Score increase is not editing success.',font=font,fill='black');canvas.save(OUT/'ten_moves_comparison.png')
(OUT/'summary.json').write_text(json.dumps(dict(protocol=protocol,records=records),ensure_ascii=False,indent=2)+'\n');print('TEN_MOVES_COMPLETE',flush=True)
