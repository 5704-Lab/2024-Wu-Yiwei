"""Resume only the full reverse paths after interrupted initial run.

The source inversion, FlowEdit and partial inversion results are reused exactly.
Initial and recovery script identities are retained separately.
"""
import argparse
import json
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL,SD3Transformer2DModel

from run import ROOT,HERE,sha_file,sha_tensor,write,calc_v_sd3


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='bear_v1');args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');out=HERE/'outputs'/args.tag
    if (out/'complete.json').exists():raise RuntimeError('Already complete')
    protocol=json.loads((out/'protocol.json').read_text())
    assert sha_file(HERE/'run.py')==protocol['script_sha256']
    input_path=ROOT/'experiments/stage1_full/inputs/bear__00.pt'
    assert sha_file(input_path)==protocol['input_sha256']
    data=torch.load(input_path,map_location='cpu',weights_only=True)
    x=data['x'].to(device);assert sha_tensor(x)==protocol['source_latent_sha256']
    checkpoint=torch.load(out/'trajectory_checkpoint.pt',map_location='cpu',weights_only=True)
    states=checkpoint['states'];snapshots=checkpoint['snapshots']
    assert set(states)=={'flowedit','ode_partial','ode_partial_reconstruction'}
    ascending=torch.load(out/'inversion_checkpoint.pt',map_location='cpu',weights_only=True)['ascending']
    sigmas=torch.tensor(protocol['sigmas'],device=device,dtype=torch.float32)
    es,et=[tuple(v.to(device) for v in e) for e in data['embeddings']]
    seq=torch.cat([es[1],es[0],et[1],et[0]])
    pool=torch.cat([es[3],es[2],et[3],et[2]])
    initial_log=HERE/'run_bear_v1.log'
    logged=[json.loads(line) for line in initial_log.read_text().splitlines() if line.startswith('{"stage"')]
    full_done=[r for r in logged if r['stage']=='ode_full']
    start=time.monotonic()
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
    reconstruction=ascending[0].to(device);target=reconstruction.clone()
    rec_history=[reconstruction.cpu()];tar_history=[target.cpu()]
    for i in range(50):
        vs,vt=calc_v_sd3(pipe,torch.cat([reconstruction,reconstruction,target,target]),seq,pool,3.5,13.5,
                        1000*sigmas[i])
        dt=sigmas[i+1]-sigmas[i]
        reconstruction+=dt*vs;target+=dt*vt
        assert bool(torch.isfinite(reconstruction).all() and torch.isfinite(target).all())
        rec_history.append(reconstruction.cpu());tar_history.append(target.cpu())
        print('RESUME_FULL',i,round(time.monotonic()-start,2),flush=True)
    states['ode_full']=target.cpu();states['ode_full_reconstruction']=reconstruction.cpu()
    snapshots['ode_full']=tar_history;snapshots['ode_full_reconstruction']=rec_history
    peak=torch.cuda.max_memory_allocated(device)
    model.cpu();del model,pipe
    torch.cuda.empty_cache()
    print('RESUME_DECODE',flush=True)
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    source_tensor=torch.from_numpy(np.array(Image.open(out/'source.png')).copy()).permute(2,0,1)[None].to(device).float()/255
    encoded=(vae.encode(source_tensor*2-1).latent_dist.mode()-vae.config.shift_factor)*vae.config.scaling_factor
    encoding_error=float((encoded-x).norm()/x.norm());assert encoding_error<1e-5
    states['vae_reconstruction']=x.cpu();decoded={}
    for name,latent in states.items():
        im=((vae.decode(latent.to(device)/vae.config.scaling_factor+vae.config.shift_factor,
                        return_dict=False)[0]+1)/2).clamp(0,1)
        array=(im[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype('uint8')
        Image.fromarray(array).save(out/(name+'.png'));decoded[name]=im.cpu()
    reference_path=ROOT/'experiments/stage1_full/outputs/bear__00__seed0/states.pt'
    assert sha_file(reference_path)==protocol['reference_sha256']
    ref=torch.load(reference_path,map_location='cpu',weights_only=True)
    ref_error=float((states['flowedit']-ref['baseline_final']).norm()/ref['baseline_final'].norm())
    assert ref_error<1e-5
    gen=torch.Generator('cpu').manual_seed(0)
    noise_hashes=[sha_tensor(torch.randn(x.shape,generator=gen,dtype=torch.float32)) for _ in range(33)]
    assert noise_hashes==ref['noise_hashes']
    metrics={}
    for name,latent in states.items():
        mse=float((decoded[name].double()-decoded['vae_reconstruction'].double()).square().mean())
        metrics[name]=dict(pixel_rmse_from_vae_source=float(mse**.5),psnr_from_vae_source_db=None if mse==0 else float(-10*np.log10(mse)),
            latent_rms_displacement=float((latent.double()-x.cpu().double()).square().mean().sqrt()))
    torch.save(dict(record=data['record'],states=states,snapshots=snapshots),out/'states.pt')
    recovery=dict(initial_attempt_exit_code=143,initial_attempt_finished=False,
        recovered_completed_stages=['flowedit','source_inversion_full','ode_partial'],
        full_reverse_recomputed_from_original_inversion_state=True,
        interrupted_full_reverse_steps_logged=len(full_done),
        interrupted_extra_branch_evaluations_at_least=4*len(full_done),
        unlogged_inflight_work_unknown=True,
        initial_attempt_seconds_at_last_log=logged[-1]['elapsed_seconds'],
        resumed_elapsed_seconds=time.monotonic()-start,
        recovery_script_sha256=sha_file(__file__),
        initial_script_sha256=protocol['script_sha256'])
    stages=dict(flowedit=dict(retained_reference_relative_error=ref_error,noise_hashes_match=True,
                             calls=33,branch_evaluations=132),
                source_inversion_full=dict(calls=50,branch_evaluations=100),
                ode_partial=dict(steps=33,calls_with_reconstruction=33,branch_evaluations_with_reconstruction=132,
                                 inversion_plus_edit_branch_evaluations=132),
                ode_full=dict(steps=50,calls_with_reconstruction=50,branch_evaluations_with_reconstruction=200,
                              inversion_plus_edit_branch_evaluations=200))
    write(out/'complete.json',dict(status='complete_after_checkpoint_recovery',protocol_file='protocol.json',
        stages=stages,metrics=metrics,source_reencoding_relative_error=encoding_error,baseline_identity_verified=True,
        totals=dict(transformer_calls_for_retained_results=166,single_branch_evaluations_for_retained_results=564,
                    interrupted_work_also_performed=True),
        recovery=recovery,peak_model_gpu_bytes=peak,
        elapsed_seconds=logged[-1]['elapsed_seconds']+time.monotonic()-start,
        elapsed_seconds_scope='active initial and recovery stages; excludes gap between processes',
        metric_caveat='Source difference is not edit quality; legitimate color/morphology change pixels.',
        script_sha256=protocol['script_sha256']))
    print('RESUME_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
