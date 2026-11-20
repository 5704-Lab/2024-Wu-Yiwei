"""Fresh factor-controlled sources and original VAE latents. Offline truth only."""
import hashlib
import json
import sys
import time
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL
from probe import ROOT, HERE, write
sys.path.insert(0, str(ROOT/'experiments/semantic_feedback_20261028'))
from dataset import render, VARIANTS


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = HERE/'outputs/fresh_v1'
    if (out/'manifest.json').exists():
        raise RuntimeError('Immutable fresh dataset exists')
    variants = dict(VARIANTS, wrong_head_front=dict(color=1., morph=1., head=-.85),
        wrong_heading_front=dict(color=1., morph=1., heading=-.85), pose_front_only=dict(head=-.85))
    vae = AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',
        local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    start = time.monotonic()
    rows = []
    for scene in range(8):
        seed = 286031+scene
        camera_side = 1. if scene%2 == 0 else -1.
        row = dict(scene=scene, seed=seed, camera_side=camera_side, split='fresh_heldout', variants={})
        for name, factors in variants.items():
            p = out/'images'/f'scene{scene:02d}'/f'{name}.png'
            p.parent.mkdir(parents=True, exist_ok=True)
            image = render(seed, camera_side=camera_side, **factors)
            image.save(p)
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float().to('cuda:0')/255
            with torch.no_grad():
                z = vae.encode(2*pixels-1).latent_dist.mode()
                y = (z-vae.config.shift_factor)*vae.config.scaling_factor
                decoded = (vae.decode(z, return_dict=False)[0]+1)/2
            lp = out/'latents'/f'scene{scene:02d}'/f'{name}.pt'
            lp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(y=y.cpu(), vae_rmse=float((decoded-pixels).square().mean().sqrt())), lp)
            row['variants'][name] = dict(file=str(p.relative_to(HERE)), factors=factors,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        rows.append(row)
        print('FRESH', scene, round(time.monotonic()-start, 2), flush=True)
    write(out/'manifest.json', dict(records=rows, scope='8 fresh controlled sources; 15 states each; camera sides balanced; factors used only in offline evaluation',
        source_independent_of_previous_datasets=True, renderer_sha256=hashlib.sha256((ROOT/'experiments/semantic_feedback_20261028/dataset.py').read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
