"""Stream full native descriptors; store maps/metrics instead of multi-GB features."""
import argparse
import re
import time
import numpy as np
import torch
from diffusers import SD3Transformer2DModel
from transformers import AutoTokenizer
from common import *
from tap import NativeTap
from match import descriptor,role_distribution,transport,measure,evaluate


def masks(tokenizer,text,words,n):
    tokens=tokenizer(text,padding='max_length',max_length=256,truncation=True,return_offsets_mapping=True,return_tensors='pt')
    out=[]
    for word in words:
        spans=[(m.start(),m.end()) for m in re.finditer(re.escape(word),text,re.I)]
        mask=torch.zeros(n,dtype=torch.bool)
        for i,(a,b) in enumerate(tokens.offset_mapping[0].tolist()):
            if a!=b and any(a<v and b>u for u,v in spans):mask[i+77]=True
        if not mask.any():raise ValueError((word,text))
        out.append(mask)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',default=PAIRS)
    ap.add_argument('--tag',default='native_v1');ap.add_argument('--variants',nargs='+')
    ap.add_argument('--dataset',default='data_v1')
    ap.add_argument('--selection',type=Path)
    args=ap.parse_args();setup();out=HERE/'outputs'/args.tag
    selected=json.loads(args.selection.read_text())['candidates'] if args.selection else None
    data_folder=HERE/'outputs'/args.dataset
    manifest=json.loads((data_folder/'manifest.json').read_text())
    protocol=dict(date='2026-11-04',script_sha256=sha(__file__),tap_sha256=sha(HERE/'tap.py'),match_sha256=sha(HERE/'match.py'),
        layers=LAYERS,probes=PROBES,noise_banks=['shared','independent_current_seed_plus_10000000'],
        facets=['O','post_adaln','K'],normalizations=['raw_cosine','source_centered_cosine'],
        matching=['nn_mutual_all_descriptors','sinkhorn40_temperature0.05_post_adaln_centered_l11_l20'],semantic_matching_weight=[0.,.2],
        semantic_weight_0p2_only_for='centered O/post_adaln at l11; ablation to bound compute before heldout',
        source_roles='manual target noun and context spans from existing text; original native C/O only; not anatomical keypoints',
        no_truth_or_deformation_penalty_in_matching=True,no_spatial_update_mask=True,extra_trained_modules=False,
        data_manifest_sha256=sha(data_folder/'manifest.json'),dataset=args.dataset,
        selected_candidates=selected,selection_file_sha256=sha(args.selection) if args.selection else None,
        compute='GPU0 original FP32 SD3; full 32x32 descriptors; native SDPA; stream similarities across four probes')
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==json.loads(json.dumps(protocol))
    else:write(out/'protocol.json',protocol)
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).to('cuda:0')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'pretrained_models/sd3-medium/tokenizer_3',local_files_only=True)
    tap=NativeTap(model);start=time.monotonic();states=0
    for row in manifest['records']:
        pair=row['pair']
        if pair not in args.pairs:continue
        inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        et=inp['embeddings'][1];condition=dict(seq=et[0],pool=et[2])
        role_masks=masks(tokenizer,row['record']['target_prompt'],ROLES[pair.split('__')[0]],et[0].shape[1])
        source=torch.load(data_folder/pair/'source.pt',weights_only=True)['y'].to('cuda:0')
        src=[]
        with torch.no_grad():
            for tau,seed in PROBES:
                eps=torch.randn(source.shape,generator=torch.Generator('cpu').manual_seed(seed)).to(source)
                features,pred=tap.observe(source,eps,tau,condition)
                src.append(features)
            if not (out/'interface.json').exists():
                tau,seed=PROBES[0];eps=torch.randn(source.shape,generator=torch.Generator('cpu').manual_seed(seed)).to(source)
                f,p=tap.observe(source,eps,tau,condition)
                plain=model(hidden_states=(1-tau)*source+tau*eps,timestep=torch.tensor([tau*1000],device=source.device),
                    encoder_hidden_states=condition['seq'].to(source),pooled_projections=condition['pool'].to(source),return_dict=False)[0]
                f2,p2=tap.observe(source,eps,tau,condition)
                write(out/'interface.json',dict(native_max_abs=float((plain-p).abs().max()),repeat_max_abs=max(float((f[l][k]-f2[l][k]).abs().max()) for l in LAYERS for k in f[l]),
                    residual_hook_count=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())))
        source_roles={l:torch.stack([role_distribution(s[l],role_masks)[:,0] for s in src]).mean(0) for l in LAYERS}
        for name,truth in row['variants'].items():
            if args.variants is not None and name not in args.variants:continue
            dest=out/pair/f'{name}.json'
            if dest.exists():continue
            y=torch.load(data_folder/pair/f'{name}.pt',weights_only=True)['y'].to(source)
            sims={};c_delta=[]
            with torch.no_grad():
                for bank in ['shared','independent']:
                    for (tau,seed),s in zip(PROBES,src):
                        eps=torch.randn(y.shape,generator=torch.Generator('cpu').manual_seed(seed+(10000000 if bank=='independent' else 0))).to(y)
                        f,pred=tap.observe(y,eps,tau,condition)
                        for l in LAYERS:
                            sr=role_distribution(s[l],role_masks);cr=role_distribution(f[l],role_masks)
                            role_sim=sr@cr.T
                            for facet in ['O','post_adaln','K']:
                                for norm in ['raw','centered']:
                                    if selected and not any(k.startswith(f'{bank}/{facet}/l{l}/{norm}/') for k in selected):continue
                                    a=descriptor(s[l][facet][0],s[l][facet][0],norm)
                                    b=descriptor(f[l][facet][0],s[l][facet][0],norm)
                                    sim=a@b.T
                                    for sem in [0.,.2]:
                                        if sem and not (l==11 and facet in ['O','post_adaln'] and norm=='centered'):continue
                                        key=f'{bank}/{facet}/l{l}/{norm}/sem{sem}'
                                        if selected and not any(k.startswith(key+'/') for k in selected):continue
                                        value=(sim+sem*role_sim)/len(PROBES)
                                        sims[key]=sims.get(key,0)+value
                            if bank=='shared':c_delta.append(float((f[l]['C']-s[l]['C']).square().mean()))
                result={};arrays={}
                for key,sim in sims.items():
                    layer=int(key.split('/')[2][1:])
                    for method in ['nn','sinkhorn']:
                        if selected and key+'/'+method not in selected:continue
                        if method=='sinkhorn' and not ('/post_adaln/' in key and '/centered/' in key and
                            ('/l11/' in key or '/l20/' in key)):continue
                        j,mutual,gap,alternative=transport(sim,method)
                        m=measure(j,mutual,gap,alternative,source_roles[layer])
                        metric_key=key+'/'+method
                        result[metric_key]=dict(**m,evaluation=evaluate(j,mutual,m,truth))
                        arrays[metric_key]=j.cpu().numpy().astype(np.uint16)
                        arrays[metric_key+'/mutual']=mutual.cpu().numpy()
            write(dest,dict(pair=pair,split=row['split'],variant=name,source_hash=digest(source),current_hash=digest(y),
                prompt_hash=digest(et[0])+digest(et[2]),full_context_change_mse=float(np.mean(c_delta)),results=result))
            np.savez_compressed(dest.with_suffix('.npz'),**arrays)
            states+=1;print('MEASURE',pair,name,states,round(time.monotonic()-start,2),flush=True)
        del src
    write(out/f'timing_{int(time.time())}.json',dict(states_this_run=states,seconds=time.monotonic()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
