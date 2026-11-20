"""Uniform endpoint comparison and actual trajectory / tube audit."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,Phase,seed_setup,write,panel,save_image
sys.path.insert(0,str(HERE.parent/'joint_permission'))
from geometry import Geometry
from run import Relation,CFG
from material import Material
from types import SimpleNamespace

def main():
 seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');out=HERE/'outputs/analysis'
 short=torch.load(ROOT/'experiments/paired_relation_validation/outputs/short_condition_encodings.pt',weights_only=True)['short']
 rows=[];audits=[]
 for pair in ['bear__00','brown_owl__00']:
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=data['x'].to(sc.device)
  with torch.no_grad():src=sc.image(x)
  geom=Geometry(src);phase=Phase(src)
  attrs=['A bear with black fur.','A bear with brown fur.'] if pair=='bear__00' else ['An owl with white feathers.','An owl with brown feathers.']
  texts=[data['record']['target_prompt'],data['record']['source_prompt'],short[pair][1],short[pair][0],*attrs]
  emb,_=sc.encode(texts)
  orig=ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt'
  candidates=[('SOURCE',None,x),('FlowEdit',orig,None)]
  for old in ['achromatic_strong','constrained_fast']:
   candidates.append(('previous/'+old,ROOT/'experiments/joint_permission/outputs'/old/pair/'trajectory.pt',None))
  for tag in ['v2','v3']:
   for p in sorted((HERE/'outputs'/tag/'full'/pair).glob('*/trajectory.pt')):
    candidates.append((tag+'/'+p.parent.name,p,None))
  for name in ['material_pilot','material_full_chroma']:
   p=HERE/'outputs'/name/pair/('trajectory.pt' if 'full' in name else 'state.pt')
   if p.exists():candidates.append((name,p,None))
  for group in ['guided_full','guided_paired','transport_full','transport_roundtrip_control','transport_paired']:
   for p in sorted((HERE/'outputs'/group/pair).glob('*/trajectory.pt')):
    candidates.append((group+'/'+p.parent.name,p,None))
  for p in sorted((HERE/'outputs/conditions'/pair).glob('seed*/*/trajectory.pt')):
   candidates.append(('conditions/'+p.parent.parent.name+'/'+p.parent.name,p,None))
  columns=[]
  for name,path,y in candidates:
   if path:
    d=torch.load(path,weights_only=True);y=d['final'].to(sc.device)
    if str(path).startswith(str(HERE)) and 'states' in d:
     result=json.loads((path.parent/'complete.json').read_text());states=d['states'];steps=result['rows']
     a=dict(path=str(path.relative_to(ROOT)),states=len(states),updates=len(steps),finite=bool(torch.isfinite(states).all()),
      source_error=float((states[0]-data['x']).abs().max()),endpoint_error=float((states[-1]-d['final']).abs().max()))
     a['passed']=bool(a['finite'] and a['states']==34 and a['updates']==33 and a['source_error']==0 and a['endpoint_error']==0
       and [r['step'] for r in steps]==list(range(17,50)))
     if name=='v3/BASE':
      a['baseline_max_error']=max(r['baseline_step_error'] for r in steps);a['passed'] &= a['baseline_max_error']==0
     if name in ['transport_paired/NO_WARP','conditions/seed0/BASE']:
      baseline=torch.load(orig,weights_only=True)
      a['zero_intervention_all_state_error']=float((states-baseline['states'][17:]).abs().max())
      a['passed'] &= a['zero_intervention_all_state_error']==0
     if name=='v3/TUBE':
      ctx=dict(source=src,x=x,e=emb[:4]);rel=Relation(SimpleNamespace(sc=sc),ctx,'TUBE');actual=[]
      with torch.no_grad():
       for i,r in enumerate(steps):
        th=torch.tensor(r['theta'],device=sc.device);im=sc.image(states[i+1].to(sc.device))
        actual.append(float((rel.observe(im)[0]-rel.reference(th)[0]).norm()))
      a['max_actual_tube_norm']=max(actual);a['tube_radius']=result['config']['tube_radius']
      a['tube_log_recomputation_error']=max(abs(v*v-r['loss_after']) for v,r in zip(actual,steps))
      a['holds']=sum(r['status']=='tube_hold' for r in steps)
      a['task_floor_failures']=sum(not r['task_floor_satisfied'] for r in steps)
      a['passed'] &= a['max_actual_tube_norm']<=a['tube_radius']+1e-6 and a['tube_log_recomputation_error']<1e-5
     if name=='material_full_chroma':
      payload=torch.load(path.parent/'state.pt',weights_only=True)
      mat=Material(src,pair,clusters=result['clusters'],chroma_only=True);mat.centers=payload['centers'].to(sc.device)
      features=mat.features(src);dist=(features[:,None]-mat.centers[None,:,:,None,None]).square().sum(2)
      mat.weights=torch.softmax(-dist/.12**2,1);th=payload['theta'].to(sc.device,dtype=x.dtype)
      def encode(im):
       z=sc.vae.encode(im*2-1).latent_dist.mode();return (z-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor
      with torch.no_grad():expected=x+encode(mat.image(th))-encode(src)
      a['family_reconstruction_error']=float((expected-y).abs().max());a['passed'] &= a['family_reconstruction_error']<2e-5
      upper=torch.tensor(result['parameter_upper'])
      a['parameters_in_bounds']=all(bool((torch.tensor(r['theta'])>=0).all() and (torch.tensor(r['theta'])<=upper+1e-6).all()) for r in steps)
      a['passed'] &= a['parameters_in_bounds']
     assert a['passed'],a
     audits.append(a)
   with torch.no_grad():
    im=sc.image(y);s=sc.logits_image(torch.cat([im,im.flip(-1)]),emb)
    row=dict(pair=pair,method=name,full_target=float(s[0,0]),short_contrast=float((s[:,2]-s[:,3]).mean()),
     attribute_contrast=float((s[:,4]-s[:,5]).mean()),geometry=float(geom.loss(geom(im),40)),
     rgb_rmse=float((im-src).square().mean().sqrt()),background_top64_rmse=float((im[:,:,:64]-src[:,:,:64]).square().mean().sqrt()),
     **phase.diagnostics(im));rows.append(row)
   if name in ['SOURCE','FlowEdit','v2/SHARED_FACT','v3/TUBE','material_full_chroma','transport_paired/SOURCE_NORM_SAFE','guided_paired/guided_64','conditions/seed0/ATTRIBUTE_POSE']:
    columns.append((name,im))
  panel(columns,out/f'{pair}_comparison.png',title=pair,size=320)
  # A contact sheet preserves all evaluated failures without a score-based ranking.
  selected=[r for r in candidates if r[0]!='SOURCE']
  thumbs=[]
  for name,path,_ in selected:
   if path is None:continue
   final_path=path.parent/'final.png'
   if final_path.exists():thumbs.append((name,final_path))
  sheets=[]
  for start in range(0,len(thumbs),4):
   dest=out/f'{pair}_contact_row{start//4}.png';panel(thumbs[start:start+4],dest,title=pair,size=256);sheets.append(Image.open(dest).copy())
  canvas=Image.new('RGB',(1024,321*len(sheets)),'white')
  for i,sheet in enumerate(sheets):canvas.paste(sheet,(0,321*i))
  canvas.save(out/f'{pair}_all_methods.png')
 write(out/'endpoints.json',dict(rows=rows,caveat='same frozen scorer diagnostics; not ground truth or held-out evaluation. Top 64 rows are evaluation-only background strips.'))
 fields=list(rows[0])
 with (out/'endpoints.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 write(out/'audit.json',dict(passed=all(r['passed'] for r in audits),completed_trajectories=len(audits),
  active_updates=sum(r['updates'] for r in audits),trajectories=audits,
  code_hashes_at_audit={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
 print('ANALYSIS_COMPLETE',len(audits),sum(r['updates'] for r in audits),flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
