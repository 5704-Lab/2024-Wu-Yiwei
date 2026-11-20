"""Source-only soft instance graph adjustment; no region gates on editing.

Two equal-width Gaussian observation windows are fitted to source bear weights.
Their parameters remain fixed for current-state measurement. Uniform-part control
exposes invariance caused only by the fixed spatial prior. No oracle coordinates.
"""
import json
import math
import hashlib
from pathlib import Path
import torch
from probe import HERE, LAYERS, write
from analyze import spatial, wrapped, summary, LEGAL


def grid(n):
    side = math.isqrt(n)
    a = (torch.arange(side).double()+.5)/side-.5
    y, x = torch.meshgrid(a, a, indexing='ij')
    return torch.stack([x,y],-1).reshape(-1,2)


def windows(source):
    xy = grid(source.shape[-1])
    bear = source[0]
    # Soft EM, 20 fixed iterations, supplied count=2 from the source description.
    mu = torch.tensor([[-.20,0.],[.20,0.]], dtype=torch.float64)
    for _ in range(20):
        logits = -(xy[:,None]-mu[None]).square().sum(-1)/(2*.12**2)
        responsibilities = logits.softmax(-1)*bear[:,None]
        mu = (responsibilities.T@xy)/responsibilities.sum(0)[:,None].clamp_min(1e-15)
    mu = mu[mu[:,0].argsort()]
    w = (-(xy[None]-mu[:,None]).square().sum(-1)/(2*.12**2)).exp()
    return w, mu


def read(p, w):
    xy = grid(p.shape[-1])
    # Generic head/nose/body concept maps, bound to source-fitted instances.
    q = p[[1,2,4]][None]*w[:,None]
    q = q/q.sum(-1,keepdim=True).clamp_min(1e-15)
    centers = q@xy
    vec = torch.stack([centers[:,1]-centers[:,0], centers[:,0]-centers[:,2]],1)
    return centers, torch.atan2(vec[...,1],vec[...,0]), vec.norm(dim=-1)


def measure(item, source):
    scores, diagnostics = {}, {}
    for branch in ['dynamic','anchored']:
        for mode in ['qk','cos_competition']:
            for layer in LAYERS:
                deltas, lens, mus, centerchanges = [], [], [], []
                for obs, src in zip(item['observations'],source['observations']):
                    p, q = spatial(obs['layers'][layer],branch,mode), spatial(src['layers'][layer],branch,mode)
                    w, mu = windows(q)
                    c, a, lengths = read(p,w)
                    cs, sa, sl = read(q,w)
                    deltas.append(wrapped(a-sa).square().mean())
                    centerchanges.append((c-cs).square().mean())
                    lens.append(sl)
                    mus.append(mu)
                stem = f'source_graph/{branch}/{mode}/l{layer}'
                scores[stem+'/angles'] = float(torch.stack(deltas).mean())
                scores[stem+'/centers'] = float(torch.stack(centerchanges).mean())
                diagnostics[stem] = dict(min_source_part_separation=float(torch.stack(lens).min()),
                    inferred_instance_centers=torch.stack(mus).mean(0).tolist(),
                    all_source_parts_separated=bool(torch.stack(lens).min()>=.01),
                    uniform_part_control_angle_residual=0.,
                    uniform_part_control_meaning='fixed windows alone give identical source/current centers, not semantic evidence')
    return scores, diagnostics


def main():
    torch.set_num_threads(4)
    out = HERE/'outputs/binding_v1'
    if out.exists():
        raise RuntimeError('Immutable diagnostic exists')
    write(out/'protocol.json', dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hypothesis='source-only soft instance graph, generic noun maps; not spatial editing masks',
        fixed_gaussian_std=.12, iterations=20, calibration_sources=[4,5,6,7], labels_used_in_readout=False,
        holdout_tuning=False, primary_scope='diagnostic adjustment; must pass part localization before control'))
    limits = None
    for tag, sources, label in [('native_v3',range(8),'development'),('native_v3',range(8,16),'historical_holdout'),
                                 ('native_fresh_v1',range(8),'fresh_holdout')]:
        rows, diagnostics = [], {}
        for scene in sources:
            folder = HERE/'outputs'/tag/f'scene{scene:02d}'
            source = torch.load(folder/'source.pt',weights_only=True)
            for path in sorted(folder.glob('*.pt')):
                score, diag = measure(torch.load(path,weights_only=True),source)
                rows.append(dict(scene=scene,variant=path.stem,scores=score))
                diagnostics[scene] = diag
            print('BINDING', label, scene,flush=True)
        if limits is None:
            limits = {k:max(r['scores'][k] for r in rows if r['scene'] in [4,5,6,7] and r['variant'] in LEGAL)*1.000001+1e-14
                      for k in rows[0]['scores']}
            write(out/'limits.json',limits)
        write(out/(label+'.json'),dict(rows=rows,diagnostics=diagnostics,results=summary(rows,limits)))


if __name__ == '__main__':
    main()
