"""Restricted real-photo global-pose recovery, with fixed luminance edit target.

Not a full FlowEdit semantic-category method. Uses source/current/native text,
no correct edited reference at runtime. References below are offline scores only.
Correction is lifted into the actual y through paired VAE encodings; neutral
action is exactly zero. Finite candidate observations include the full matcher.
"""
import argparse
import math
import time
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL,SD3Transformer2DModel
from transformers import AutoTokenizer
from common import *
from reader import SelectedReader


def warp(im,deg):
    if deg==0:return im
    a=math.radians(deg)
    inverse=torch.tensor([[math.cos(a),math.sin(a),0.],[-math.sin(a),math.cos(a),0.]],device=im.device,dtype=im.dtype)[None]
    grid=F.affine_grid(inverse,im.shape,align_corners=False)
    return F.grid_sample(im,grid,align_corners=False,mode='bilinear',padding_mode='reflection')


def encode(vae,im):
    z=vae.encode(im*2-1).latent_dist.mode()
    return (z-vae.config.shift_factor)*vae.config.scaling_factor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00','dogs__00','dogs2__00','penguins__00','japanese_castle__00'])
    ap.add_argument('--tag',default='repair_v2')
    ap.add_argument('--bank',choices=['independent','shared'],default='independent')
    args=ap.parse_args();setup();out=HERE/'outputs'/args.tag
    if (out/'summary.json').exists():raise RuntimeError('Immutable recovery experiment exists')
    selection=json.loads((HERE/'outputs/analysis_v3/selection.json').read_text());key=selection['primary']
    key=args.bank+'/'+key.split('/',1)[1]
    limit=json.loads((HERE/'outputs/analysis_v3/thresholds.json').read_text())['limits'][key]
    calibration=[];truth_rows=json.loads((HERE/'outputs/data_v1/manifest.json').read_text())['records']
    truth={(r['pair'],name):v for r in truth_rows for name,v in r['variants'].items()}
    for p in (HERE/'outputs/calibration_weighted_v1').glob('*__00/*.json'):
        row=json.loads(p.read_text())
        gt=truth.get((row['pair'],row['variant']))
        if gt and 'matrix' in gt:
            import numpy as np
            if np.linalg.det(gt['matrix'])>0:
                f=row['results'][key]['global_frame'];a=np.array(gt['matrix']);singular=np.linalg.svd(a,compute_uv=False)
                calibration.append(dict(translation_error=float(np.linalg.norm(np.array(f['translation'])-gt['translation'])),
                    stretch_error=float(np.max(np.abs(np.array(f['stretch_eigenvalues'])-singular)))))
    limit['photometric_translation_tolerance']=max(r['translation_error'] for r in calibration)+.001
    limit['photometric_stretch_tolerance']=max(r['stretch_error'] for r in calibration)+.001
    write(out/'protocol.json',dict(date='2026-11-04',key=key,limit=limit,pairs=args.pairs,
        task='retain prescribed global luminance ratio 0.6 while recovering added whole-image rotation',
        task_ratio_tolerance=.05,recovery_factor_per_iteration=.5,maximum_iterations=4,
        action_grid='-measured_angle * [0.5,1,1.5], clipped to +/-10 degrees; choose minimum actual latent norm among feasible',
        oracle_reference_at_runtime=False,external_masks=False,nominal_velocity='zero; post-edit finite recovery pilot, not full FE integration',
        correct_target_reference='offline dark-source evaluation only; never passed to reader or candidate selector',
        code_hashes={p.name:sha(p) for p in [Path(__file__),HERE/'reader.py',HERE/'weighted_frames.py']}))
    write(out/'measurement_error_calibration.json',dict(source_count=4,states=len(calibration),errors=calibration,
        translation_error_envelope=limit['photometric_translation_tolerance'],stretch_eigenvalue_error_envelope=limit['photometric_stretch_tolerance'],
        semantic_tolerance_translation=0.,semantic_tolerance_stretch=0.,statistical_coverage_guarantee=False,
        reason='source-only photometric calibration ignored orientation-dependent correspondence errors; use known affine residuals, not current bad state to authorize extra shape'))
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).to('cuda:0')
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).to('cuda:0')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'pretrained_models/sd3-medium/tokenizer_3',local_files_only=True)
    summary=[];start=time.monotonic()
    with torch.no_grad():
        for index,pair in enumerate(args.pairs):
            dest=out/pair
            if (dest/'result.json').exists():continue
            reader=SelectedReader(model,tokenizer,pair,key);x=reader.x
            source=image(vae,x);desired_lum=.6*float(source.mean())
            # Known perturbation is generated here but not supplied to feedback.
            added_angle=12. if index%2==0 else -12.
            initial=encode(vae,warp(source*.6,added_angle));y=initial.clone()
            save_image(image(vae,y),dest/'initial.png');save_image(source,dest/'source.png')
            reference=torch.load(HERE/'outputs/data_v1'/pair/'dark.pt',weights_only=True)['y'].to(x)
            ref_image=image(vae,reference)
            init_observer=reader(y);init_audit=reader(y,True);logs=[];total_action=0.
            for step in range(4):
                current=reader(y);f=current['frame'];angle=f.get('rotation_deg')
                if not f.get('valid') or angle is None:
                    logs.append(dict(step=step,status='unsupported'));break
                beta=limit['rotation_tolerance_degrees']
                if abs(angle)<=beta:
                    logs.append(dict(step=step,status='inside_measured_tube',angle=angle));break
                cap=beta+.5*(abs(angle)-beta)
                im=image(vae,y);base_encode=encode(vae,im)
                neutral=y+(encode(vae,warp(im,0.))-base_encode)
                assert float((neutral-y).abs().max())==0.,'non-neutral latent lift'
                angles=sorted(set(float(max(-10.,min(10.,-angle*k))) for k in [.5,1.,1.5]),key=abs)
                trials=[];candidates=[]
                for correction in angles:
                    d=encode(vae,warp(im,correction))-base_encode
                    candidate=y+d;obs=reader(candidate);cf=obs['frame']
                    ci=image(vae,candidate);lum=float(ci.mean())
                    quality=cf.get('valid',False) and not cf.get('reflection',True) and obs['mutual_fraction']>=limit['mutual_floor'] and cf['fit_rmse']<=limit['fit_rmse_cap']
                    geometry=quality and sum(x*x for x in cf['translation'])**.5<=limit['photometric_translation_tolerance'] and max(abs(x-1) for x in cf['stretch_eigenvalues'])<=limit['photometric_stretch_tolerance']
                    passed=geometry and cf['rotation_deg'] is not None and abs(cf['rotation_deg'])<=cap and abs(lum-desired_lum)/float(source.mean())<=.05
                    trial=dict(correction_degrees=correction,latent_norm=float(d.norm()),observer=obs,
                        luminance_ratio=lum/float(source.mean()),passed=passed);trials.append(trial)
                    if passed:candidates.append((float(d.norm()),candidate,correction))
                if not candidates:
                    logs.append(dict(step=step,status='finite_grid_infeasible',before=current,trials=trials));break
                _,y,last_action=min(candidates,key=lambda x:x[0]);y=y.detach()
                total_action+=last_action
                logs.append(dict(step=step,status='accepted',before=current,cap_degrees=cap,accepted_correction_degrees=last_action,trials=trials))
                save_image(image(vae,y),dest/f'step{step}.png')
            final=image(vae,y);fo=reader(y);audit=reader(y,True)
            # Equal-direction negative control and source reversion are separate
            # endpoints; neither is substituted for a failed controlled run.
            im0=image(vae,initial);base0=encode(vae,im0)
            wrong_angle=math.copysign(abs(total_action) if total_action else 6.,added_angle)
            wrong=initial+(encode(vae,warp(im0,wrong_angle))-base0)
            oracle=initial+(encode(vae,warp(im0,-added_angle))-base0)
            endpoint=dict(controlled=y,unchanged=initial,wrong_direction=wrong,source_revert=x,oracle_inverse=oracle)
            evaluations={}
            for mode,z in endpoint.items():
                ii=image(vae,z);error=(ii[:,:,80:-80,80:-80]-ref_image[:,:,80:-80,80:-80]).square().mean().sqrt()
                evaluations[mode]=dict(reference_center_rmse=float(error),luminance_ratio=float(ii.mean()/source.mean()))
                save_image(ii,dest/f'{mode}.png')
            save_image(ref_image,dest/'offline_reference_dark.png')
            torch.save(dict(initial=initial.cpu(),final=y.cpu()),dest/'states.pt')
            result=dict(pair=pair,split=split(pair),added_angle_truth_offline_only=added_angle,initial_observer=init_observer,
                final_observer=fo,initial_independent_audit=init_audit,final_independent_audit=audit,steps=logs,
                accepted_steps=sum(r['status']=='accepted' for r in logs),evaluations=evaluations,
                bound_is_semantic_certificate=False,scope='global affine pose and luminance mechanism only')
            write(dest/'result.json',result);summary.append(result)
            print('REPAIR',pair,result['accepted_steps'],evaluations,round(time.monotonic()-start,2),flush=True)
            del reader
    write(out/'summary.json',dict(results=summary,seconds=time.monotonic()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),full_category_editing_validated=False))


if __name__=='__main__':main()
