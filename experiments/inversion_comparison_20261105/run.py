"""One real source: original FlowEdit versus ordinary source-conditioned ODE inversion.

No training, inversion improvements, attention injection, masks or new generated assets.
Outputs are immutable per --tag. Both reconstruction and target branches are saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL, SD3Transformer2DModel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from FlowEdit_utils import calc_v_sd3


def sha_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha_tensor(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--tag',default='bear_v1')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    out=HERE/'outputs'/args.tag
    if out.exists(): raise RuntimeError('Output exists. Use a fresh tag; do not overwrite experiments.')
    out.mkdir(parents=True)
    input_path=ROOT/'experiments/stage1_full/inputs/bear__00.pt'
    reference_path=ROOT/'experiments/stage1_full/outputs/bear__00__seed0/states.pt'
    schedule_path=ROOT/'experiments/stage1_full/inputs/manifest.json'
    data=torch.load(input_path,map_location='cpu',weights_only=True)
    meta=json.loads(schedule_path.read_text())
    times=torch.tensor(meta['timesteps'],dtype=torch.float32,device=device)
    sigmas=torch.cat([times/1000,torch.zeros(1,device=device)])
    assert len(times)==50 and bool(torch.all(sigmas[:-1]>sigmas[1:]))
    start_index=17
    x=data['x'].to(device)
    es,et=[tuple(v.to(device) for v in emb) for emb in data['embeddings']]
    seq=torch.cat([es[1],es[0],et[1],et[0]])
    pool=torch.cat([es[3],es[2],et[3],et[2]])
    protocol=dict(
        date='2026-11-05',pair='bear__00',source=data['record'],model='original SD3 Medium',
        precision='float32',resolution=512,T=50,scheduler='original shifted Euler grid, shift=3.0',
        source_cfg=3.5,target_cfg=13.5,negative_prompt='',flowedit_seed=0,n_avg=1,n_max=33,n_min=0,
        partial_start_index=start_index,partial_sigma=float(sigmas[start_index]),
        full_sigma=float(sigmas[0]),sigmas=sigmas.cpu().tolist(),
        inversion='ordinary source-conditioned forward ODE, explicit Euler evaluated at left (current) endpoint',
        source_reconstruction='reverse Euler with the same source prompt and source CFG=3.5',
        editing='reverse Euler with target prompt and target CFG=13.5',
        full_inversion_steps=50,partial_inversion_steps=33,
        inversion_numerics='Euler forward and backward are not exact inverses; reconstruction explicitly evaluated',
        scope='one existing natural source; illustrative comparison, not a general benchmark',
        spatial_mask=False,attention_injection=False,optimization=False,new_model=False,
        input_sha256=sha_file(input_path),reference_sha256=sha_file(reference_path),
        schedule_sha256=sha_file(schedule_path),script_sha256=sha_file(__file__),
        source_image_actual_sha256=sha_file(ROOT/data['record']['source_image']),
        source_latent_sha256=sha_tensor(x),condition_sha256=[sha_tensor(t) for e in [es,et] for t in e],
        model_config_sha256=sha_file(ROOT/'pretrained_models/sd3-medium/transformer/config.json'),
        torch_version=torch.__version__,device_name=torch.cuda.get_device_name(device))
    assert protocol['source_image_actual_sha256']==data['record']['source_sha256']
    write(out/'protocol.json',protocol)
    Image.open(ROOT/data['record']['source_image']).convert('RGB').resize(
        (512,512),Image.Resampling.BICUBIC).save(out/'source.png')
    clock=time.monotonic()
    print('LOAD_TRANSFORMER',flush=True)
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',
        local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
    counts=dict(transformer_calls=0,single_branch_evaluations=0)
    stage_stats={}
    states={}
    snapshots={}
    def progress(stage,step,latent):
        assert bool(torch.isfinite(latent).all()),(stage,step)
        print(json.dumps(dict(stage=stage,step=step,elapsed_seconds=round(time.monotonic()-clock,2),
                              latent_rms=float(latent.square().mean().sqrt())),ensure_ascii=False),flush=True)

    def paired(source_state,target_state,t):
        vs,vt=calc_v_sd3(pipe,torch.cat([source_state,source_state,target_state,target_state]),
                       seq,pool,3.5,13.5,t)
        counts['transformer_calls']+=1;counts['single_branch_evaluations']+=4
        return vs,vt

    # Recompute the nominal baseline; identical CPU noise bank to the retained seed-0 run.
    t0=time.monotonic();count0=counts.copy()
    gen=torch.Generator('cpu').manual_seed(0)
    y=x.clone();history=[y.cpu()]
    noises=[]
    for i in range(start_index,50):
        eps=torch.randn(x.shape,generator=gen,dtype=torch.float32).to(device)
        noises.append(sha_tensor(eps))
        qs=(1-sigmas[i])*x+sigmas[i]*eps
        qt=y+qs-x
        vs,vt=paired(qs,qt,times[i])
        y=y+(sigmas[i+1]-sigmas[i])*(vt-vs)
        history.append(y.cpu())
        progress('flowedit',i,y)
    states['flowedit']=y.cpu();snapshots['flowedit']=history
    ref=torch.load(reference_path,weights_only=True)
    reference=ref['baseline_final'].to(device)
    ref_error=float((y-reference).norm()/reference.norm())
    assert ref_error<1e-5,('Baseline replay differs',ref_error)
    assert noises==ref['noise_hashes']
    stage_stats['flowedit']=dict(seconds=time.monotonic()-t0,calls=counts['transformer_calls']-count0['transformer_calls'],
        branch_evaluations=counts['single_branch_evaluations']-count0['single_branch_evaluations'],
        retained_reference_relative_error=ref_error,noise_hashes_match=True)
    torch.save(dict(states=states,snapshots=snapshots),out/'flowedit_checkpoint.pt')

    # Ascend the SAME grid from clean sigma=0 to sigma=1. Index identifies grid coordinate.
    t0=time.monotonic();count0=counts.copy()
    ascending={50:x.cpu()}
    z=x.clone()
    source_seq=torch.cat([es[1],es[0]])
    source_pool=torch.cat([es[3],es[2]])
    for j in range(49,-1,-1):
        current=sigmas[j+1]
        pred=model(hidden_states=torch.cat([z,z]),timestep=(1000*current).expand(2),
                   encoder_hidden_states=source_seq,pooled_projections=source_pool,return_dict=False)[0]
        unconditional,conditional=pred.chunk(2)
        velocity=unconditional+3.5*(conditional-unconditional)
        counts['transformer_calls']+=1;counts['single_branch_evaluations']+=2
        z=z+(sigmas[j]-current)*velocity
        ascending[j]=z.cpu()
        progress('source_ode_inversion',j,z)
    snapshots['inversion_ascending_by_grid_index']=ascending
    stage_stats['source_inversion_full']=dict(seconds=time.monotonic()-t0,
        calls=counts['transformer_calls']-count0['transformer_calls'],
        branch_evaluations=counts['single_branch_evaluations']-count0['single_branch_evaluations'])
    torch.save(dict(ascending=ascending,source=x.cpu()),out/'inversion_checkpoint.pt')

    for name,first in [('ode_partial',start_index),('ode_full',0)]:
        t0=time.monotonic();count0=counts.copy()
        reconstruction=ascending[first].to(device);target=reconstruction.clone()
        rec_history=[reconstruction.cpu()];tar_history=[target.cpu()]
        for i in range(first,50):
            vs,vt=paired(reconstruction,target,times[i])
            delta=sigmas[i+1]-sigmas[i]
            reconstruction=reconstruction+delta*vs
            target=target+delta*vt
            rec_history.append(reconstruction.cpu());tar_history.append(target.cpu())
            progress(name,i,target)
        states[name]=target.cpu();states[name+'_reconstruction']=reconstruction.cpu()
        snapshots[name]=tar_history;snapshots[name+'_reconstruction']=rec_history
        stage_stats[name]=dict(seconds=time.monotonic()-t0,steps=50-first,
            calls_with_reconstruction=counts['transformer_calls']-count0['transformer_calls'],
            branch_evaluations_with_reconstruction=counts['single_branch_evaluations']-count0['single_branch_evaluations'],
            editing_branch_evaluations=2*(50-first),
            inversion_branch_evaluations=2*(50-first),
            reconstruction_branch_evaluations=2*(50-first),
            inversion_plus_edit_branch_evaluations=4*(50-first))
        torch.save(dict(states=states,snapshots=snapshots),out/'trajectory_checkpoint.pt')
    torch.cuda.synchronize(device)
    peak_model=torch.cuda.max_memory_allocated(device)
    model.cpu();del model,pipe
    torch.cuda.empty_cache()
    print('DECODE_RESULTS',flush=True)
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    def decode(latent):
        return ((vae.decode(latent.to(device)/vae.config.scaling_factor+vae.config.shift_factor,
                            return_dict=False)[0]+1)/2).clamp(0,1)
    def save_image(image,path):
        array=(image[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype('uint8')
        Image.fromarray(array).save(path)
    source_tensor=torch.from_numpy(np.array(Image.open(out/'source.png')).copy()).permute(2,0,1)[None].to(device).float()/255
    encoded=(vae.encode(source_tensor*2-1).latent_dist.mode()-vae.config.shift_factor)*vae.config.scaling_factor
    encoder_error=float((encoded-x).norm()/x.norm())
    assert encoder_error<1e-5,('Source encoding identity failed',encoder_error)
    states['vae_reconstruction']=x.cpu()
    decoded={}
    for name,latent in states.items():
        image=decode(latent);save_image(image,out/(name+'.png'));decoded[name]=image.cpu()
    vae_source=decoded['vae_reconstruction'].double()
    metrics={}
    for name,latent in states.items():
        image=decoded[name].double()
        mse=float((image-vae_source).square().mean())
        metrics[name]=dict(pixel_rmse_from_vae_source=float(mse**.5),
                           psnr_from_vae_source_db=None if mse==0 else float(-10*np.log10(mse)),
                           latent_rms_displacement=float((latent.double()-x.cpu().double()).square().mean().sqrt()))
    torch.save(dict(record=data['record'],states=states,snapshots=snapshots),out/'states.pt')
    write(out/'complete.json',dict(status='complete',protocol_file='protocol.json',stages=stage_stats,
        totals=counts,metrics=metrics,source_reencoding_relative_error=encoder_error,
        peak_model_gpu_bytes=peak_model,elapsed_seconds=time.monotonic()-clock,
        metric_caveat='Source difference is not edit quality; legitimate color/morphology also change pixels.',
        baseline_identity_verified=True,script_sha256=sha_file(__file__)))
    print('COMPARISON_COMPLETE',json.dumps(dict(seconds=time.monotonic()-clock,output=str(out))),flush=True)


if __name__=='__main__': main()
