"""Same-family numerical refinement and actual frozen-VAE feasibility checks."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from probe import CleanScorer,Observers,read_image,seed_setup,panel,save_image,write
from material_exact import fit_material_exact


def encode(sc,image):
    latent=sc.vae.encode(image*2-1).latent_dist.mode()
    return (latent-sc.vae.config.shift_factor)*sc.vae.config.scaling_factor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='v3');args=ap.parse_args()
    seed_setup();out=HERE/'outputs'/args.tag
    protocol=dict(scope='Same-family oracle fitting and frozen VAE diagnostics; no deployed text-to-edit success',
                  args=vars(args),code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                  reference_tolerance=2e-4,fit_scenes=list(range(4,8))+list(range(16,40)),vae_scenes=list(range(32,36)))
    if (out/'protocol_refine.json').exists():raise RuntimeError('Use a fresh tag')
    write(out/'protocol_refine.json',protocol)
    for name in ['refine.py','material.py','material_exact.py','probe.py','dataset.py']:
        (out/('snapshot_'+name)).write_bytes((HERE/name).read_bytes())
    sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
    manifest=json.loads((out/'dataset/manifest.json').read_text())
    rows=[];vae_rows=[];started=time.monotonic()
    for record in manifest['records']:
        scene=record['scene']
        if scene not in protocol['fit_scenes']:continue
        source=read_image(HERE/record['variants']['source']['file'],sc.device)
        target=read_image(HERE/record['variants']['correct']['file'],sc.device)
        fitted,info=fit_material_exact(source,target)
        row=dict(scene=scene,split=record['split'],within_reference_tolerance=info['rgb_mse']<=2e-4,**info)
        rows.append(row);dest=out/'scenes'/f'scene{scene:02d}'
        save_image(fitted,dest/'fit_MATERIAL_EXACT.png');write(dest/'fit.json',row)
        write(out/'refine_results.json',dict(protocol=protocol,rows=rows,vae_rows=vae_rows,seconds=time.monotonic()-started))
        if scene in protocol['vae_scenes']:
            with torch.no_grad():
                x=encode(sc,source);xt=encode(sc,target);imx=sc.image(x);imt=sc.image(xt)
                roundtrip=encode(sc,imx)
                identity_increment=encode(sc,imx)-roundtrip
            fitted_vae,fit_vae=fit_material_exact(imx,imt)
            with torch.no_grad():
                lifted=x+encode(sc,fitted_vae)-roundtrip
                imlift=sc.image(lifted)
                chord=xt-x;half=sc.image(x+.5*chord)
            _,jvp=torch.func.jvp(sc.image,(x,),(chord,))
            finite=[]
            for fraction in [.001,.005,.02]:
                with torch.no_grad():fd=(sc.image(x+fraction*chord)-sc.image(x-fraction*chord))/(2*fraction)
                finite.append(dict(chord_fraction=fraction,relative_jvp_error=float((fd-jvp).norm()/jvp.norm().clamp_min(1e-12))))
            record_vae=dict(scene=scene,source_reconstruction_mse=float((imx-source).square().mean()),
                            target_reconstruction_mse=float((imt-target).square().mean()),
                            identity_increment_max=float(identity_increment.abs().max()),
                            exact_chord_endpoint_max=float((sc.image(x+chord)-imt).abs().max()),
                            full_chord_linearization_relative_error=float((jvp-(imt-imx)).norm()/(imt-imx).norm()),
                            lifted_mse_to_matched_vae_target=float((imlift-imt).square().mean()),
                            lifted_mse_to_rendered_target=float((imlift-target).square().mean()),
                            decoded_source_family_fit=fit_vae,jvp_finite_checks=finite)
            vae_rows.append(record_vae)
            panel([('Source rendered',source),('Source VAE',imx),('Legal rendered',target),('Legal VAE',imt),
                   ('Latent halfway',half),('Material lift',imlift)],dest/'vae_comparison.png',title=f'scene {scene}',size=192)
            torch.save(dict(x=x.cpu(),target_latent=xt.cpu(),material_lift=lifted.cpu()),dest/'vae_states.pt')
            write(out/'refine_results.json',dict(protocol=protocol,rows=rows,vae_rows=vae_rows,seconds=time.monotonic()-started))
        print('REFINE',scene,record['split'],info['rgb_mse'],info['projected_gradient_inf'],round(time.monotonic()-started,1),flush=True)
    write(out/'refine_results.json',dict(protocol=protocol,rows=rows,vae_rows=vae_rows,seconds=time.monotonic()-started))
    print('REFINE_COMPLETE',len(rows),len(vae_rows),flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH):main()
