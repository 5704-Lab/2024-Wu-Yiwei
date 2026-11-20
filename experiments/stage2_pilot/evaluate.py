"""Frozen independent SigLIP2 scores; descriptive four-source pilot only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import torch
from transformers import AutoModel,AutoProcessor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/stage2_pilot/outputs'
def correlation(a,b):
 if len(a)<3 or np.std(a)<1e-20 or np.std(b)<1e-20:return None
 return float(spearmanr(a,b).statistic)
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--partial',action='store_true');args=p.parse_args()
 protocol=json.loads((OUT.parent/'protocol.json').read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 path=ROOT/'pretrained_models/siglip2-base-patch16-256';model=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False).to(args.device);processor=AutoProcessor.from_pretrained(path,local_files_only=True)
 cases=[];records=[]
 for rec in protocol['pairs']:
  dest=OUT/rec['pair_id']
  if args.partial and not (dest/'complete.json').exists():continue
  generated=json.loads((dest/'complete.json').read_text());labels=generated['arms']+['source']
  images=[Image.open(dest/f'{label}.png').convert('RGB') for label in labels]
  with torch.no_grad():
   tokens=processor(text=[rec['source_prompt'],rec['target_prompt']],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(args.device)
   text=model.get_text_features(**tokens);text=text/text.norm(dim=-1,keepdim=True)
   all_features=[]
   for start in range(0,len(images),4):
    inputs=processor(images=images[start:start+4],return_tensors='pt').to(args.device)
    feats=model.get_image_features(**inputs);all_features.append(feats/feats.norm(dim=-1,keepdim=True))
   image=torch.cat(all_features);similarities=(image@text.T).double().cpu().numpy();features=image.cpu()
  scores={label:{'source':float(similarities[k,0]),'target':float(similarities[k,1]),'preference':float(similarities[k,1]-similarities[k,0])} for k,label in enumerate(labels)}
  base=scores['baseline'];preds={r['arm']:r for r in generated['predictions'] if not r['missing']}
  latents=torch.load(dest/'terminal_latents.pt',weights_only=True);pair_rows=[]
  for direction in protocol['directions']:
   plus,minus=direction+'_plus',direction+'_minus'
   if plus not in preds or minus not in preds:
    pair_rows.append({'pair_id':rec['pair_id'],'direction':direction,'missing':True});continue
   po,mo=preds[plus],preds[minus];local=po['local_work']-mo['local_work'];finite=po['finite_work']-mo['finite_work'];scale=po['prediction_scale']+mo['prediction_scale']
   dtarget=scores[plus]['target']-scores[minus]['target'];dpref=scores[plus]['preference']-scores[minus]['preference']
   row={'pair_id':rec['pair_id'],'direction':direction,'missing':False,'local_prediction':local,'finite_prediction':finite,'prediction_scale':scale,'delta_target_plus_minus':dtarget,'delta_preference_plus_minus':dpref,
    'plus_vs_baseline_target':scores[plus]['target']-base['target'],'minus_vs_baseline_target':scores[minus]['target']-base['target'],'plus_vs_baseline_preference':scores[plus]['preference']-base['preference'],'minus_vs_baseline_preference':scores[minus]['preference']-base['preference'],
    'terminal_plus_minus_latent_norm':float((latents[plus].double()-latents[minus].double()).norm())}
   for method,pred in [('local',local),('finite',finite)]:
    for outcome,delta in [('target',dtarget),('preference',dpref)]:
     valid=abs(pred)>protocol['prediction_abstain_abs_cosine']*scale and abs(delta)>protocol['endpoint_abstain_abs_delta']
     row[f'{method}_{outcome}_valid']=bool(valid);row[f'{method}_{outcome}_sign_correct']=bool(pred*delta>0) if valid else None
   pair_rows.append(row);records.append(row)
  group={}
  for method in ['local','finite']:
   for outcome in ['target','preference']:
    vals=[r for r in pair_rows if not r['missing']];valid=[r for r in vals if r[f'{method}_{outcome}_valid']]
    group[f'{method}_{outcome}']={'spearman_descriptive':correlation([r[method+'_prediction'] for r in vals],[r['delta_'+outcome+'_plus_minus'] for r in vals]),'valid':len(valid),'sign_correct':sum(r[f'{method}_{outcome}_sign_correct'] for r in valid)}
  report={'record':rec,'scores':scores,'paired_interventions':pair_rows,'within_source_descriptive':group,'official_minus_unified':{key:scores['official'][key]-scores['baseline'][key] for key in ['target','preference']},'noise_unchanged':generated['noises_unchanged'],'generation_seconds':generated['elapsed_seconds']}
  (dest/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');cases.append(report)
  print(json.dumps({'case':rec['pair_id'],'baseline':base,'official_delta':report['official_minus_unified'],'within_source':group}),flush=True)
 totals={}
 for method in ['local','finite']:
  for outcome in ['target','preference']:
   valid=[r for r in records if r[f'{method}_{outcome}_valid']]
   totals[f'{method}_{outcome}']={'eligible_pairs':len(valid),'all_pairs':len(records),'sign_correct':sum(r[f'{method}_{outcome}_sign_correct'] for r in valid),'accuracy_descriptive':sum(r[f'{method}_{outcome}_sign_correct'] for r in valid)/len(valid) if valid else None,'source_groups':len(cases),'pooled_spearman_secondary':correlation([r[method+'_prediction'] for r in records],[r['delta_'+outcome+'_plus_minus'] for r in records])}
 modelmeta=json.loads((path/'download_manifest.json').read_text())
 summary={'stage':'2 feasibility pilot','complete':len(cases)==4,'source_groups':len(cases),'seeds':[0],'time_points':[33],'paired_directions':len(records),'main_endpoints':11*len(cases),'official_reference_endpoints':len(cases),'evaluator_device':args.device,'independent_evaluator':modelmeta['repository'],'evaluator_revision':modelmeta['revision'],'evaluator_sha256':next(f['sha256'] for f in modelmeta['files'] if f['path']=='model.safetensors'),'all_noise_unchanged':all(c['noise_unchanged'] for c in cases),'results':totals,'cases':cases,'scope':'Descriptive four-source mechanism pilot. No full Stage 2 pass, accurate semantic success rate, non-edit region metric or efficiency claim. No secondary evaluator or formal inference.','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(OUT/'paired_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));print(json.dumps({'complete':summary['complete'],'results':totals}),flush=True)
if __name__=='__main__':main()
