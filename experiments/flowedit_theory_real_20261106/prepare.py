"""Freeze the protocol and encode the existing bear source at 512, before inference."""
from pathlib import Path
import sys
import time
import json
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKL

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/sota_inversion_20261105'))
from common import setup, write, sha, save_image


@torch.inference_mode()
def main():
    setup()
    out = Path(__file__).resolve().parent / 'outputs/run_v1'
    if out.exists():
        raise RuntimeError('Never overwrite a registered experiment')
    out.mkdir(parents=True)
    start = time.monotonic()
    oldpath = ROOT / 'experiments/sota_inversion_20261105/outputs/inputs_v1/inputs.pt'
    old = torch.load(oldpath, map_location='cpu', weights_only=True)
    protocol = dict(
        date='2026-11-06', model='same local FLUX.1-dev weights', source=old['source'],
        resolution=512, source_count=1, seed=20261106,
        scope='mechanism and finite-value validation; not a semantic preservation certificate',
        precision='BF16 audit, then FP32 computation using the same stored weights on two GPUs',
        weights_frozen=True, tf32=False, input_text_cache_sha256=sha(oldpath),
        scales=[.2, .5, 1.], main_sigma=.5, quadrature_nodes=2,
        objective='fixed radial midpoint scalar divided by latent dimension',
        source_target_guidance=3.5, radial_source_fixed=True,
        guard=dict(source_displacement_rms_max=.35, proposal_rms=.08, steps=8,
                   armijo=.25, backtracking_factors=[1., .5, .25, .125],
                   objective_measured_gain_min=1e-7,
                   constraint='exact Euclidean projection onto source RMS ball',
                   task_threshold=None),
        ablation='cheap field proposals with the same scalar finite-value gate',
        finite_difference=dict(probe_displacement_rms=.08, relative_tolerance=.05,
                               perturbation_rms=[.0005, .002, .008]),
        baselines=[dict(name='flowedit_equal_g', T=28, n_max=24, gs=3.5, gt=3.5),
                   dict(name='flowedit_author', T=28, n_max=24, gs=1.5, gt=5.5)],
        source_path=dict(T=.3, approximate_euler_steps=[2, 4, 8],
                         reference_heun_steps=[16, 32], defect='cubic Hermite midpoint residual',
                         edit_steps=8, displacement_scale=1., source_target_guidance=3.5,
                         certified_network_L=False),
        audit=dict(noise_candidates=8, noise_time=1/3, offline_only=True),
        windows=dict(top_background=[0,110], bottom_water=[425,512]),
        prohibited_inference_dependencies=['image scoring windows', 'SigLIP evaluation'],
        preparation_script_sha256=sha(__file__),
        registered_before_transformer_execution=True,
    )
    write(out / 'protocol.json', protocol)
    source = Image.open(ROOT / old['source']['source_image']).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
    source.save(out / 'original.png')
    vae = AutoencoderKL.from_pretrained(ROOT/'pretrained_models/flux1-dev/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    pixels = torch.from_numpy(np.array(source).copy()).permute(2,0,1)[None].float().to('cuda:0')/255
    latent = (vae.encode(2*pixels-1).latent_dist.mode()-vae.config.shift_factor)*vae.config.scaling_factor
    packed = rearrange(latent,'b c (h ph) (w pw)->b (h w) (c ph pw)',ph=2,pw=2).cpu()
    rec = ((vae.decode(latent/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
    save_image(rec,out/'vae_source.png')
    ids=torch.zeros(1,32,32,3)
    ids[...,1]=torch.arange(32)[:,None];ids[...,2]=torch.arange(32)[None,:]
    inputs={}
    for name,condition in old['inputs'].items():
        inputs[name]={key:value.clone() for key,value in condition.items() if key not in ['img','img_ids']}
        inputs[name].update(img=packed.clone(),img_ids=ids.reshape(1,1024,3))
    torch.save(dict(inputs=inputs,source=old['source'],size=512),out/'inputs.pt')
    write(out/'prepared.json',dict(seconds=time.monotonic()-start,inputs_sha256=sha(out/'inputs.pt'),
                                  original_sha256=sha(out/'original.png')))
    print('PREPARED',out,flush=True)


if __name__=='__main__':
    main()
