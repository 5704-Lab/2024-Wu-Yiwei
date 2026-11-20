"""Uniform endpoint measurements and structural audit, including failed methods.

Scores and image distances are diagnostics, not automatic semantic success labels.
All completed trajectories are included; interrupted trials are separately listed.
"""
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,Phase,write,panel,save_image,seed_setup
from geometry import Geometry

OUT=HERE/'outputs'
BRANCHES=['control/full','manifold','manifold_semantic','constrained_fast','routing','achromatic','achromatic_strong','achromatic_neutral']

def main():
 seed_setup()
 sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
 short=torch.load(ROOT/'experiments/paired_relation_validation/outputs/short_condition_encodings.pt',weights_only=True)['short']
 endpoints=[];audits=[]
 complete=sorted(p for branch in BRANCHES for p in (OUT/branch).rglob('complete.json'))
 for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
  x=data['x'].to(sc.device)
  with torch.no_grad():source=sc.image(x)
  save_image(source,OUT/'analysis'/f'{pair}_source.png')
  texts=[data['record']['target_prompt'],data['record']['source_prompt'],short[pair][1],short[pair][0]]
  attr=None
  if pair=='bear__00':attr=['A bear with black fur.','A bear with brown fur.']
  if pair=='brown_owl__00':attr=['An owl with white feathers.','An owl with brown feathers.']
  embeddings,_=sc.encode(texts+(attr or []))
  geometry=Geometry(source);phase=Phase(source)
  columns=[('Source',source)]
  base=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
  candidates=[('SOURCE',x,None),('FlowEdit',base['final'].to(sc.device),None)]
  for path in complete:
   if pair not in path.parts:continue
   result=json.loads(path.read_text());tr=torch.load(path.parent/'trajectory.pt',weights_only=True)
   name=str(path.parent.relative_to(OUT)).replace('/'+pair,'')
   states=tr['states'];rows=result['rows']
   audit=dict(path=str(path.relative_to(ROOT)),states=len(states),updates=len(rows),
       finite=bool(torch.isfinite(states).all()),source_error=float((states[0]-data['x']).abs().max()),
       endpoint_error=float((states[-1]-tr['final']).abs().max()),steps=[r['step'] for r in rows])
   audit['passed']=bool(audit['finite'] and audit['states']==34 and audit['updates']==33
       and audit['source_error']<1e-6 and audit['endpoint_error']==0 and audit['steps']==list(range(17,50)))
   if 'source_error_first' in rows[0]:
    audit['routing_source_error']=rows[0]['source_error_first'];audit['passed'] &= audit['routing_source_error']<1e-6
   if name.startswith('achromatic'):
    theta=np.asarray(rows[-1]['theta']);audit['parameter_min']=float(theta.min());audit['parameter_max']=float(theta.max())
    audit['passed'] &= bool(theta.min()>=-1e-6 and theta.max()<=1+1e-6)
    with torch.no_grad():
     order=result['config']['order'];m=order**2;p=torch.tensor(theta,device=sc.device,dtype=x.dtype)
     brightness=F.interpolate(p[:m].reshape(1,1,order,order),source.shape[-2:],mode='bilinear',align_corners=False)
     desaturation=F.interpolate(p[m:].reshape(1,1,order,order),source.shape[-2:],mode='bilinear',align_corners=False)
     gray=(source*torch.tensor([.299,.587,.114],device=sc.device)[None,:,None,None]).sum(1,keepdim=True)
     neutral=(1-desaturation)*source+desaturation*gray
     sign=-1. if pair=='bear__00' else 1.
     if result['config'].get('family','exposure')=='neutralmix':
      photograph=(1-brightness)*neutral+brightness*(0. if sign<0 else 1.)
     else:photograph=(neutral*torch.exp(sign*result['config']['brightness_log_bound']*brightness)).clamp(0,1)
     def encode(im):
      z=sc.vae.encode(im*2-1).latent_dist.mode();return (z-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor
     expected=x+encode(photograph)-encode(source)
     audit['family_latent_error']=float((expected.cpu()-tr['final']).abs().max())
     audit['passed'] &= audit['family_latent_error']<2e-4
     audit['finite_task_checks']=all((not r['accepted']) or r['attempts'][-1]['actual']>=r['attempts'][-1]['requested']-result['config']['semantic_tolerance']-1e-7 for r in rows)
     audit['passed'] &= audit['finite_task_checks']
     save_image(photograph,path.parent/'final_photograph.png')
   assert audit['passed'],audit
   audits.append(audit);candidates.append((name,tr['final'].to(sc.device),result))
  for name,y,result in candidates:
   with torch.no_grad():
    im=sc.image(y);logs=sc.logits_image(torch.cat([im,im.flip(-1)]),embeddings)
    row=dict(pair=pair,method=name,full_target=float(logs[0,0]),full_source=float(logs[0,1]),
      short_contrast=float((logs[:,2]-logs[:,3]).mean()),
      attribute_contrast=float((logs[:,4]-logs[:,5]).mean()) if attr else None,
      geometry_mean=float(geometry.loss(geometry(im),0)),geometry_tail=float(geometry.loss(geometry(im),40)),
      rgb_rmse=float((im-source).square().mean().sqrt()),
      background_top64_rmse=float((im[:,:,:64]-source[:,:,:64]).square().mean().sqrt()) if attr else None,
      rgb_mean=[float(v) for v in im.mean((0,2,3))],**phase.diagnostics(im))
    if result:
     row['seconds']=result['seconds'];row['accepted_steps']=sum(r.get('accepted',True) for r in result['rows'])
    endpoints.append(row)
   if name in ['FlowEdit','control/full/RF','constrained_fast','achromatic','achromatic_strong','achromatic_neutral']:
    columns.append((name,im))
  panel(columns,OUT/'analysis'/f'{pair}_comparison.png',title=pair,size=320)
  if pair=='cat_crown__00':
   panel([('Source',source)]+[(name,OUT/name.split('/')[0]/pair/name.split('/')[-1]/'final.png') for name,_,_ in candidates if name.startswith('routing/')],
         OUT/'analysis'/'cat_routing.png',title=pair,size=320)
 partial=[]
 for p in OUT.rglob('progress.json'):
  if not (p.parent/'complete.json').exists():
   d=json.loads(p.read_text());partial.append(dict(path=str(p.relative_to(ROOT)),updates=len(d['rows']),completed=False))
 write(OUT/'analysis'/'endpoints.json',dict(rows=endpoints,caveat='Scores and distances are diagnostic; visual inspection is required. No held-out quality claim. Top 64 rows are a post hoc background-only diagnostic on bear/owl, never used during inference.'))
 with (OUT/'analysis'/'endpoints.csv').open('w',newline='') as f:
  fields=sorted(set().union(*(r.keys() for r in endpoints)));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(endpoints)
 write(OUT/'analysis'/'audit.json',dict(passed=all(a['passed'] for a in audits),completed_trajectories=len(audits),
   completed_active_updates=sum(a['updates'] for a in audits),trajectories=audits,interrupted=partial,
   code_sha256_at_audit={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
 print('ANALYSIS_COMPLETE',len(audits),sum(a['updates'] for a in audits),flush=True)

if __name__=='__main__':
 with sdpa_kernel(SDPBackend.MATH):main()
