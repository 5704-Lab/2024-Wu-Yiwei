"""Local-state previews always include original FlowEdit, never claim final quality."""
from pathlib import Path
import json
import torch
from PIL import Image,ImageDraw
from diffusers import AutoencoderKL
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;OUT=BASE/'outputs'
torch.set_num_threads(4);device=torch.device('cuda:0')
vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
def image(y):
 with torch.no_grad():pixels=((vae.decode(y.to(device)/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
 return Image.fromarray((pixels[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))
for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
 states=torch.load(BASE/'baseline_states'/f'{pair}.pt',weights_only=True);s=next(s for s in states['snapshots'] if s['step']==33)
 rec=states['record'];r=json.loads((OUT/f'{pair}_step33_float32_native_hvp.json').read_text())
 ts=torch.load(OUT/f'{pair}_step33_float32_native_tensors.pt',weights_only=True)
 columns=[('Source',Image.open(ROOT/rec['source_image']).convert('RGB').resize((256,256))),('Original FlowEdit step33',image(s['zE']).resize((256,256))),('Original FlowEdit final',image(states['baseline_final']).resize((256,256)))]
 jt=r.get('joint_trials',[])
 if jt:
  radius=jt[0]['norm'];gu=ts['semantic_direction'];b=ts['direction_correction'];pd=gu+b/b.norm();pd=pd/pd.norm()
  columns += [('SigLIP G, same local norm',image(s['zE']+radius*gu).resize((256,256))),('SigLIP D, local diagnostic',image(s['zE']+radius*pd).resize((256,256)))]
 canvas=Image.new('RGB',(256*len(columns),300),'white');draw=ImageDraw.Draw(canvas)
 for k,(label,im) in enumerate(columns):canvas.paste(im,(256*k,40));draw.text((256*k+4,8),label,fill='black')
 draw.text((4,281),pair+' | local previews are not full corrected trajectories',fill='black')
 canvas.save(OUT/f'{pair}_local_comparison.png')
print('COMPARISON_PREVIEWS_COMPLETE')
