"""Locked native relation metrics: development -> calibration -> fresh audit.

No learned linear readout, no oracle reference/landmark enters the measurement.
Geometric truth may be used by a separate offline landmark diagnostic only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from probe import ROOT, HERE, LAYERS, write
from run import PROBES

LEGAL = ['source','color_only','partial','valid','valid_alt']
ERRORS = ['wrong_heading','wrong_head','wrong_background','wrong_object','both_objects','shifted',
          'wrong_head_front','wrong_heading_front']
MAPS = ['qk', 'cos_spatial', 'cos_competition', 'dot_competition']
BRANCHES = ['dynamic', 'anchored']


def spatial(row, branch, mode):
    if mode == 'qk':
        return row[branch+'_probability'][0].double()
    if mode == 'cos_spatial':
        return (row[branch+'_cosine_map'][0].double()/.1).softmax(-1)
    # ConceptAttention-style competition is over concepts at each image patch,
    # not over patches. Normalize spatially only to compute conditional moments.
    logits = row[branch+('_dot_map' if mode == 'dot_competition' else '_cosine_map')][0].double()
    if mode == 'cos_competition':
        logits = logits/.1
    p = logits.softmax(0)
    return p/p.sum(-1, keepdim=True).clamp_min(1e-15)


def geometry(p):
    side = math.isqrt(p.shape[-1])
    axis = (torch.arange(side).double()+.5)/side-.5
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    centers = p @ torch.stack([xx, yy],-1).reshape(-1,2)
    h, n, b = centers[11], centers[12], centers[13]
    hr, nr, br = centers[14], centers[15], centers[16]
    vectors = torch.stack([n-h, h-b, nr-hr, hr-br])
    angles = torch.atan2(vectors[:,1], vectors[:,0])
    # Two projected instance centers and four unit directions. No latent updates.
    return centers, angles, vectors.norm(dim=-1)


def wrapped(a):
    return torch.atan2(a.sin(), a.cos())


def metrics(item, source):
    scores = {}
    support = {}
    for layer in LAYERS:
        for branch in BRANCHES:
            for mode in MAPS:
                center_delta, angle_delta, distribution = [], [], []
                for a, s in zip(item['observations'], source['observations']):
                    p = spatial(a['layers'][layer], branch, mode)
                    q = spatial(s['layers'][layer], branch, mode)
                    c, angle, lengths = geometry(p)
                    cs, sa, sl = geometry(q)
                    center_delta.append((c[[13,16]]-cs[[13,16]]).flatten())
                    angle_delta.append(wrapped(angle-sa))
                    distribution.append(((p[[7,10]]-q[[7,10]])**2).sum(-1))
                stem = f'{branch}/{mode}/l{layer}'
                # Mean squared per-probe angle, not angle of an averaged vector.
                scores[stem+'/angles'] = float(torch.stack(angle_delta).square().mean())
                scores[stem+'/centers'] = float(torch.stack(center_delta).square().mean())
                scores[stem+'/scene_right'] = float(torch.stack(distribution).mean())
                supports = []
                for s in source['observations']:
                    c, _, lengths = geometry(spatial(s['layers'][layer], branch, mode))
                    separation = float((c[13]-c[16]).norm())
                    # Fixed geometric support heuristic, not a measured accuracy guarantee.
                    supports.append(dict(min_part_separation=float(lengths.min()),
                        instance_separation=separation, left_order=bool(c[13,0] < c[16,0]),
                        supported=bool(lengths.min() >= .01 and separation >= .1 and c[13,0] < c[16,0])))
                support[stem] = supports
        for facet in ['image_O','image_K','image_V','context_C']:
            values = []
            for a, s in zip(item['observations'], source['observations']):
                v, u = a['layers'][layer][facet].double(), s['layers'][layer][facet].double()
                values.append((v-u).square().mean())
            scores[f'raw/{facet}/l{layer}'] = float(torch.stack(values).mean())
    return scores, support


def summary(rows, limits):
    result = {}
    for key, limit in limits.items():
        legal = [r for r in rows if r['variant'] in LEGAL]
        wrong = [r for r in rows if r['variant'] in ERRORS]
        by_variant = {}
        for name in ERRORS:
            subset = [r for r in wrong if r['variant'] == name]
            if subset:
                by_variant[name] = dict(rejected=sum(r['scores'][key] > limit for r in subset), total=len(subset))
        result[key] = dict(legal_accepted=sum(r['scores'][key] <= limit for r in legal), legal_total=len(legal),
            errors_rejected=sum(r['scores'][key] > limit for r in wrong), errors_total=len(wrong), by_error=by_variant)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['dev','audit'], required=True)
    ap.add_argument('--tag', default='native_v3')
    ap.add_argument('--output', default='analysis_v1')
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = HERE/'outputs'/args.output
    if (out/(args.stage+'.json')).exists():
        raise RuntimeError('Immutable analysis exists')
    folder = HERE/'outputs'/args.tag
    rows, supports = [], {}
    for scene_folder in sorted(folder.glob('scene*')):
        scene = int(scene_folder.name[5:])
        if args.stage == 'dev' and scene >= 8:
            continue
        source = torch.load(scene_folder/'source.pt', weights_only=True)
        for path in sorted(scene_folder.glob('*.pt')):
            item = torch.load(path, weights_only=True)
            score, support = metrics(item, source)
            rows.append(dict(scene=scene, variant=path.stem, split='development' if scene<4 else 'calibration', scores=score))
            supports[scene] = support
        print('ANALYZE', args.stage, scene, flush=True)
    if args.stage == 'dev':
        limits = {key: max(r['scores'][key] for r in rows if r['split']=='calibration' and r['variant'] in LEGAL)*1.000001+1e-14
                  for key in rows[0]['scores']}
        write(out/'limits.json', limits)
        write(out/'protocol.json', dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            calibration='4 separate sources: max over five legally admissible progress states; fixed before fresh extraction',
            independence='sources not per-state observations; frozen variants/maps/layers; all configurations reported',
            deployment='no configuration is admitted to hard control before source binding and held-out selectivity pass'))
    else:
        limits = json.loads((out/'limits.json').read_text())
    write(out/(args.stage+'.json'), dict(rows=rows, support=supports, results=summary(rows, limits)))
    report = summary(rows, limits)
    print('BEST_DIAGNOSTIC', sorted(((v['errors_rejected'],k,v['legal_accepted'],v['legal_total']) for k,v in report.items()), reverse=True)[:10], flush=True)


if __name__ == '__main__':
    main()
