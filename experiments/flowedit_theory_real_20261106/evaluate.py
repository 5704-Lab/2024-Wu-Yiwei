"""Offline image/proxy evaluation and numerical claim audit; never controls edits."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModel, AutoProcessor
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/run_v1'
sys.path.insert(0,str(ROOT/'experiments/sota_inversion_20261105'))
from common import write,sha,setup


def fidelity(image,reference):
    a=image[0].permute(1,2,0).float().numpy()
    b=reference[0].permute(1,2,0).float().numpy()
    mse=float(np.mean((a.astype('float64')-b.astype('float64'))**2))
    return dict(rmse=mse**.5,psnr_db=None if mse==0 else -10*np.log10(mse),
                ssim=float(structural_similarity(a,b,channel_axis=2,data_range=1.,
                    gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))


def panel(names,path,title,labels=None,side=320):
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    canvas=Image.new('RGB',(len(names)*side,side+95),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((8,6),title,fill='black',font=ImageFont.truetype(font,16))
    for i,name in enumerate(names):
        draw.text((i*side+7,34),(labels or {}).get(name,name),fill='black',font=ImageFont.truetype(font,13))
        image=Image.open(OUT/(name+'.png')).convert('RGB').resize((side,side),Image.Resampling.LANCZOS)
        canvas.paste(image,(i*side,58))
    draw.text((8,side+72),'Same 512 source / FLUX.1-dev. Source fidelity and proxy gain do not certify edit success.',
              fill='#444444',font=ImageFont.truetype(font,11))
    canvas.save(path)


@torch.inference_mode()
def main():
    setup()
    data=torch.load(OUT/'results.pt',map_location='cpu',weights_only=True)
    protocol=json.loads((OUT/'protocol.json').read_text())
    images=data['images'];states=data['states'];source=images['source']
    rows={}
    for name,image in images.items():
        delta=states[name].double()-states['source'].double()
        rows[name]=dict(vs_vae_source=fidelity(image,source),latent_displacement_rms=float(delta.square().mean().sqrt()))
        for region,(start,end) in protocol['windows'].items():
            rows[name][region]=fidelity(image[:,:,start:end,:],source[:,:,start:end,:])
    # Existing frozen scorer, used once offline after model inference. Prompt
    # margins are evidence about this scorer, not a semantic ground truth.
    processor=AutoProcessor.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',local_files_only=True)
    scorer=AutoModel.from_pretrained(ROOT/'pretrained_models/siglip2-base-patch16-256',
        local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    prompts=[protocol['source']['source_prompt'],protocol['source']['target_prompt'],
             'A photo of a brown bear.', 'A photo of a black bear.']
    for name in images:
        im=Image.open(OUT/(name+'.png')).convert('RGB')
        tokens=processor(text=prompts,images=im,padding='max_length',max_length=64,
                         truncation=True,return_tensors='pt')
        output=scorer(**{key:value.to('cuda:0') for key,value in tokens.items()})
        logits=output.logits_per_image[0].cpu().tolist()
        rows[name]['siglip_logits']=logits
        rows[name]['siglip_full_prompt_margin']=logits[1]-logits[0]
        rows[name]['siglip_short_color_margin']=logits[3]-logits[2]
    write(OUT/'image_metrics.json',dict(metrics=rows,scorer='existing frozen SigLIP2 base patch16 256',
        scorer_is_semantic_oracle=False,prompts=prompts,script_sha256=sha(__file__),
        windows='fixed offline source coordinates; not inference masks'))
    checks=[]
    for name in ['clean_full_gradient','clean_field_proposal']:
        trace=json.loads((OUT/(name+'_trace.json')).read_text())
        accepted=[r for r in trace['steps'] if r['status']=='accepted']
        checks.append(dict(name=name,accepted_steps=len(accepted),
            all_source_rms_within_budget=all(r['source_rms']<=.3500001 for r in accepted),
            all_actual_scalar_increased=all(r['gain']>=max(.25*r['directional_gain'],1e-7) for r in accepted),
            final_score=accepted[-1]['after'] if accepted else 0.,
            semantic_task_completed=False,termination=trace['termination']))
    write(OUT/'finite_gate_audit.json',dict(rows=checks,
        scope='recomputed actual scalar and total latent displacement only; floating-point measurements'))
    labels={'original':'Real source','source':'VAE source','flowedit_author':'FlowEdit author guidance',
            'flowedit_equal_g':'FlowEdit equal guidance','clean_full_gradient':'Clean: full gradient',
            'clean_field_proposal':'Clean: field proposal','path_reference':'Source path: Heun32',
            'path_euler2':'Source path: Euler2','path_euler8':'Source path: Euler8'}
    panel(['original','source','flowedit_equal_g','clean_full_gradient','clean_field_proposal'],
          OUT/'comparison.png','Real FLUX experiment: fixed clean objective, finite-value gate',labels)
    panel(['source','path_euler2','path_euler8','path_reference'],OUT/'source_coupling.png',
          'Source-coupled edits: approximate and numerical-reference ODE paths',labels)
    panel(['source','flowedit_author','flowedit_equal_g','clean_full_gradient'],OUT/'guidance_control.png',
          'All guidance configurations retained; budgets differ',labels)
    full_history=data['histories']['clean_full_gradient']
    indices=sorted(set([0,len(full_history)//2,len(full_history)-1]))
    panel(['clean_full_gradient_step'+f'{i:02d}' for i in indices],OUT/'clean_trajectory.png',
          'Accepted clean-state trajectory',side=320)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for name in ['clean_full_gradient','clean_field_proposal']:
        trace=json.loads((OUT/(name+'_trace.json')).read_text())
        accepted=[r for r in trace['steps'] if r['status']=='accepted']
        xs=np.arange(len(accepted)+1)
        axes[0].plot(xs,[0.]+[r['after'] for r in accepted],marker='o',label=name.replace('clean_',''))
        axes[1].plot(xs,[0.]+[r['source_rms'] for r in accepted],marker='o',label=name.replace('clean_',''))
    axes[0].set_title('Actual fixed scalar');axes[0].set_xlabel('Accepted steps');axes[0].legend(fontsize=7)
    axes[1].axhline(.35,color='black',linestyle='--');axes[1].set_title('Total latent displacement RMS')
    audit=json.loads((OUT/'source_path_audit.json').read_text())['rows']
    axes[2].loglog([r['euler_steps'] for r in audit],[r['max_path_rms_error_vs_heun32'] for r in audit],
                  marker='o',label='path error vs Heun32')
    axes[2].loglog([r['euler_steps'] for r in audit],[r['midpoint_defect_rms_max'] for r in audit],
                  marker='s',label='sampled defect')
    axes[2].set_title('Source path refinement');axes[2].set_xlabel('Euler steps');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'mechanism_checks.png',dpi=180);plt.close(fig)
    artifact_files=[p for p in OUT.iterdir() if p.suffix in ['.json','.png'] and p.name!='artifact_sha256.json']
    write(OUT/'artifact_sha256.json',{p.name:sha(p) for p in sorted(artifact_files)})
    print(json.dumps(dict(finite_gate=checks,image_metrics=rows),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
