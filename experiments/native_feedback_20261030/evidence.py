"""Paired and camera-stratified diagnostic evidence; no threshold refit."""
import json
import math
import statistics
from pathlib import Path
import hashlib
import torch
from probe import HERE, ROOT, write, digest


def main():
    out = HERE/'outputs/evidence_v1'
    if out.exists():
        raise RuntimeError('Immutable evidence exists')
    d = json.loads((HERE/'outputs/analysis_v1/native_fresh_v1_audit.json').read_text())
    limits = json.loads((HERE/'outputs/analysis_v1/limits.json').read_text())
    index = {}
    for r in d['rows']:
        index.setdefault(r['scene'],{})[r['variant']] = r['scores']
    keys = ['raw/image_O/l11','raw/context_C/l11','dynamic/cos_competition/l11/angles','anchored/qk/l11/angles']
    results = {}
    for key in keys:
        rows = {}
        for name in ['valid','color_only','wrong_head','wrong_head_front']:
            values = [r[name][key] for r in index.values()]
            rows[name] = dict(median_energy=statistics.median(values),
                median_rms_degrees=statistics.median(math.sqrt(v)*180/math.pi for v in values) if '/angles' in key else None)
        paired = {name:sum(r[name][key]>r['valid'][key] for r in index.values()) for name in ['wrong_head','wrong_head_front']}
        sides = {}
        for parity,label in [(0,'camera_plus'),(1,'camera_minus')]:
            subset = {s:r for s,r in index.items() if s%2==parity}
            sides[label] = dict(sources=len(subset),legal_accepted=sum(r[v][key]<=limits[key] for r in subset.values()
                for v in ['source','color_only','partial','valid','valid_alt']),legal_total=5*len(subset),
                head_rejected=sum(r[v][key]>limits[key] for r in subset.values() for v in ['wrong_head','wrong_head_front']),
                head_total=2*len(subset))
        results[key] = dict(by_variant=rows,paired_ordering_over_valid=paired,stratified=sides,
            energy_threshold=limits[key],rms_angle_threshold_degrees=math.sqrt(limits[key])*180/math.pi if '/angles' in key else None)
    write(out/'paired.json',dict(results=results,source_count=8,threshold_refit=False,
        paired_ordering_meaning='each wrong state differs from matched full valid edit only by head rotation; higher source distance is a weak diagnostic, not a full semantic certificate'))
    # Complete cache/execution provenance, supplemental to per-run protocols.
    protocol = json.loads((HERE/'outputs/native_v3/protocol.json').read_text())
    noise = {str(seed):digest(torch.randn((1,16,64,64),generator=torch.Generator('cpu').manual_seed(seed)))
             for seed in [19571030,19571031]}
    identity = json.loads((ROOT/'pretrained_models/sd3-medium/upstream_identity.json').read_text())
    write(out/'provenance.json',dict(model_weights_sha256_actually_streamed=protocol['model_weights_sha256'],
        pretrained_upstream_identity=identity,latent_noise_tensor_hashes=noise,
        source_and_current_latent_identity='source_hash and latent_hash stored in every native record',
        no_source_cache_persistently_reused_across_runs=True,
        backend='original JointAttnProcessor2_0, default PyTorch SDPA automatic kernel; exact kernel was not separately profiled',
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        observation_control_admission='failed; no hard closed-loop image control',
        source_code_snapshots=True))
    snap = out/'snapshots';snap.mkdir()
    for p in HERE.glob('*.py'):
        (snap/p.name).write_bytes(p.read_bytes())


if __name__ == '__main__':
    main()
