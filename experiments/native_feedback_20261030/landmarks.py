"""Independent OFFLINE renderer landmark projections, never a runtime input."""
import json
import math
import numpy as np
import torch
from probe import HERE, ROOT, write
from analyze import spatial, geometry, wrapped
from binding import windows, read


def rotation(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def truth(seed, camera_side=1., **f):
    rng = np.random.default_rng(seed)
    camera = np.array([3.4*camera_side,2.1,7.5])+[rng.uniform(-.35,.35),rng.uniform(-.1,.15),0.]
    forward = np.array([0.,.75,0.])-camera
    forward /= np.linalg.norm(forward)
    side = np.cross(forward,[0.,1.,0.]); side /= np.linalg.norm(side)
    up = np.cross(side,forward)
    yaws = [rng.uniform(-.25,.15),rng.uniform(-.4,.15)]
    scales = [rng.uniform(.94,1.06),rng.uniform(.92,1.05)]
    positions = []
    for i in range(2):
        r = rotation(yaws[i]+(f.get('heading',0.) if i==0 else 0.))
        rh = rotation(f.get('head',0.) if i==0 else 0.)
        origin = np.array([[-1.05,1.05][i]+(f.get('shift',0.) if i==0 else 0.),0.,rng.uniform(-.04,.04)])
        neck = np.array([.49,1.12,0.])
        parts = [neck+rh@np.array([.05,.04,0.]),neck+rh@np.array([.47,.025,0.]),np.array([0.,.78,0.])]
        xyz = np.array([origin+scales[i]*(r@p) for p in parts])
        delta = xyz-camera
        z = delta@forward
        xy = np.stack([7.8*(delta@side)/z/4.85,-7.8*(delta@up)/z/4.85],-1)
        positions.append(xy)
    return torch.tensor(np.array(positions))


def direction(c):
    v = c[:,1]-c[:,0]
    return torch.atan2(v[:,1],v[:,0])


def main():
    torch.set_num_threads(4)
    out = HERE/'outputs/landmarks_v1'
    if out.exists():
        raise RuntimeError('Immutable diagnostic exists')
    rows = []
    manifest = json.loads((HERE/'outputs/fresh_v1/manifest.json').read_text())
    for entry in manifest['records']:
        scene = entry['scene']
        folder = HERE/'outputs/native_fresh_v1'/f'scene{scene:02d}'
        source = torch.load(folder/'source.pt',weights_only=True)
        expected_source = truth(entry['seed'],entry['camera_side'])
        for variant in ['source','valid','wrong_head','wrong_head_front']:
            item = torch.load(folder/f'{variant}.pt',weights_only=True)
            expected = truth(entry['seed'],entry['camera_side'],**entry['variants'][variant]['factors'])
            for branch in ['dynamic','anchored']:
                for mode in ['qk','cos_competition']:
                    measured, measured_source, graph, graph_source = [], [], [], []
                    for obs,src in zip(item['observations'],source['observations']):
                        p = spatial(obs['layers'][11],branch,mode)
                        q = spatial(src['layers'][11],branch,mode)
                        c = geometry(p)[0][[11,12,13,14,15,16]].reshape(2,3,2)
                        cs = geometry(q)[0][[11,12,13,14,15,16]].reshape(2,3,2)
                        w,_ = windows(q)
                        gc,_,_ = read(p,w)
                        gcs,_,_ = read(q,w)
                        measured.append(c); measured_source.append(cs); graph.append(gc); graph_source.append(gcs)
                    for method,values,srcvalues in [('phrase',measured,measured_source),('source_graph',graph,graph_source)]:
                        v,s = torch.stack(values),torch.stack(srcvalues)
                        measured_delta = wrapped(torch.stack([direction(c) for c in v])-torch.stack([direction(c) for c in s]))[:,0]
                        true_delta = wrapped(direction(expected)-direction(expected_source))[0]
                        rows.append(dict(scene=scene, variant=variant,method=method,branch=branch,map=mode,
                            mean_part_error_pixels=float((v-expected).norm(dim=-1).mean()*512),
                            part_within_32px_fraction=float(((v-expected).norm(dim=-1)*512<=32).double().mean()),
                            predicted_head_turn_radians=float(measured_delta.mean()), true_projected_head_turn_radians=float(true_delta),
                            head_turn_absolute_error_radians=float(wrapped(measured_delta-true_delta).abs().mean()),
                            source_part_locations=s.mean(0).tolist(), true_source_part_locations=expected_source.tolist()))
        print('LANDMARK',scene,flush=True)
    grouped = {}
    for r in rows:
        k = '/'.join([r['method'],r['branch'],r['map'],r['variant']])
        grouped.setdefault(k,[]).append(r)
    summary = {k:{metric:float(np.mean([r[metric] for r in v])) for metric in ['mean_part_error_pixels','part_within_32px_fraction','head_turn_absolute_error_radians']}
               for k,v in grouped.items()}
    write(out/'result.json',dict(rows=rows,summary=summary,
        boundary='analytic 3D part-center projection, no occlusion annotation; offline only; source/current measurements computed without these labels',
        pixel_grid='512x512, part-center projected from exact same seeded camera/animal transformations'))


if __name__ == '__main__':
    main()
