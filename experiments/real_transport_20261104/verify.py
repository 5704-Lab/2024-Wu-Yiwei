"""Audit real artifacts and accepted finite steps; does not certify semantics."""
import ast
import hashlib
import platform
import numpy as np
import torch
import diffusers
from common import *


def main():
    out=HERE/'outputs/verification_v1'
    if (out/'result.json').exists():raise RuntimeError('Immutable verification exists')
    checks=[]
    def check(name,condition,**metrics):
        checks.append(dict(name=name,passed=bool(condition),**metrics))
        if not condition:raise AssertionError(name)
    setup()
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    for folder,n in [('development_v1',56),('calibration_v1',56),('heldout_v1',168),('trajectory_v1',60)]:
        paths=list((HERE/'outputs'/folder).glob('*__00/*.json'))
        check(folder+'_record_count',len(paths)==n,records=len(paths))
        interface=json.loads((HERE/'outputs'/folder/'interface.json').read_text())
        check(folder+'_read_only_identity',interface['native_max_abs']==0 and interface['repeat_max_abs']==0 and interface['residual_hook_count']==0)
    # Verify matrix errors use the refitted texture frame rather than stale
    # uniform-frame values. Early weighted v1 development is not production.
    manifest=json.loads((HERE/'outputs/data_v1/manifest.json').read_text())
    truth={(r['pair'],n):v for r in manifest['records'] for n,v in r['variants'].items()}
    mismatch=0;matrices=0
    for folder in ['development_weighted_v2','calibration_weighted_v1','heldout_weighted_v1']:
        for p in (HERE/'outputs'/folder).glob('*__00/*.json'):
            row=json.loads(p.read_text());gt=truth[(row['pair'],row['variant'])]
            if 'matrix' not in gt:continue
            for r in row['results'].values():
                if not r['global_frame'].get('valid'):continue
                expected=np.linalg.norm(np.array(r['global_frame']['matrix'])-gt['matrix'])
                actual=r['evaluation'].get('global_frame_matrix_error')
                if actual is not None:
                    mismatch+=abs(expected-actual)>1e-10;matrices+=1
    check('weighted_matrix_evaluation_matches_actual_frame',mismatch==0 and matrices>0,frames=matrices)
    accepted=0;forwards=0
    for tag in ['repair_v1','repair_v2']:
        protocol=json.loads((HERE/'outputs'/tag/'protocol.json').read_text());beta=protocol['limit']['rotation_tolerance_degrees']
        for p in (HERE/'outputs'/tag).glob('*__00/result.json'):
            r=json.loads(p.read_text());forwards+=24+4*len(r['steps'])+4*sum(len(s.get('trials',[])) for s in r['steps'])
            for step in r['steps']:
                if step['status']!='accepted':continue
                trial=next(t for t in step['trials'] if abs(t['correction_degrees']-step['accepted_correction_degrees'])<1e-10)
                check(tag+'_'+r['pair']+'_finite_accept_'+str(step['step']),trial['passed'] and
                    abs(trial['observer']['frame']['rotation_deg'])<=step['cap_degrees']+1e-10 and abs(trial['luminance_ratio']-.6)<=.05+1e-8)
                accepted+=1
    rgb=json.loads((HERE/'outputs/repair_rgb_audit_v1/summary.json').read_text())['rows']
    check('independent_rgb_registration_improves_all_six',all(abs(r['scores']['controlled']['offline_residual_angle_deg'])<abs(r['scores']['unchanged']['offline_residual_angle_deg']) for r in rgb))
    result=dict(passed=len(checks),total=len(checks),checks=checks,accepted_finite_steps=accepted,
        scope='artifact, numerical acceptance and independent known-global-motion checks only; not semantic-pose protection',
        descriptor_forward_count_main=340*8+40*4+4*3,recovery_forward_count_from_logs=forwards)
    write(out/'result.json',result)
    weight=ROOT/'pretrained_models/sd3-medium/transformer/diffusion_pytorch_model.safetensors'
    h=hashlib.sha256()
    with weight.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    check('original_transformer_weight_fingerprint',h.hexdigest()=='8b0e357dd04755fbfb067c63f4a883ac1de2942dd4a3240069874a17022e5418')
    snapshot=out/'code_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for p in HERE.glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
    write(out/'provenance.json',dict(date='2026-11-04',transformer_weights_sha256=h.hexdigest(),
        model_config_sha256=sha(ROOT/'pretrained_models/sd3-medium/transformer/config.json'),
        original_model_upstream_identity=json.loads((ROOT/'pretrained_models/sd3-medium/upstream_identity.json').read_text()),
        libraries=dict(torch=torch.__version__,diffusers=diffusers.__version__,cuda=torch.version.cuda,python=platform.python_version()),
        probe_noise_hashes={str(seed):digest(torch.randn((1,16,64,64),generator=torch.Generator('cpu').manual_seed(seed))) for seed in [20261104,20261105,30261104,30261105,40261104,40261105,50261104,50261105]},
        source_only_weights='fixed source image texture and native token role measurements; never spatially gate update',
        independent_sources='20 existing benchmark sources, split 4/4/12 for this experiment; not newly collected images',
        backend='original native SDPA, FP32, TF32 disabled; exact kernel not separately profiled',
        shared_gpu_timing='physical GPU0; other users GPU jobs present; no exclusive latency benchmark claimed',
        current_code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        observed_control_policy='finite candidate compatibility check with empirical calibration; not all-hypotheses robust safety certificate'))
    write(out/'strict_robust_scope.json',dict(translation_semantic_tolerance=0.,stretch_semantic_tolerance=0.,
        empirical_error_envelopes=json.loads((HERE/'outputs/repair_v2/measurement_error_calibration.json').read_text()),
        statement='abs(estimated residual)<=empirical error is compatible with no change but does not certify true no change. With nonzero error and zero semantic tolerance, strict upper-bound safe-set certification is infeasible.',
        full_robust_theory_implemented=False))
    print(json.dumps(dict(passed=len(checks),total=len(checks),accepted_steps=accepted,main_forwards=result['descriptor_forward_count_main'],repair_forwards=forwards)))


if __name__=='__main__':main()
