"""Frozen-model coverage/observation diagnostics, not an automatic editor."""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/null_joint_20261025'))
from patches import PatchRelation, photometries
sys.path.insert(0, str(ROOT/'experiments/phase_permission'))
from common import CleanScorer, Phase, panel, save_image, seed_setup, warp, write
sys.path.insert(0, str(ROOT/'experiments/joint_permission'))
from geometry import Geometry


def read_image(path, device):
    im = np.asarray(Image.open(path).convert('RGB')).copy().astype(np.float32)/255
    return torch.from_numpy(im).permute(2,0,1)[None].to(device)


def graph_basis(features, count=8):
    f = features[0].detach().double().cpu().numpy()
    affinity = np.exp((f @ f.T - 1)/.08)
    degree = affinity.sum(1)
    normalized = affinity/np.sqrt(degree[:,None]*degree[None,:])
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    basis = [np.ones(len(f))]
    for vector in eigenvectors[:, -2:-count-1:-1].T:
        v = vector/np.sqrt(degree)
        v -= v.mean(); v /= max(np.max(np.abs(v)), 1e-12)
        basis.append(v)
    side = int(round(math.sqrt(len(f))))
    assert side*side == len(f)
    tensor = torch.tensor(np.stack(basis).reshape(count,side,side), dtype=torch.float32)
    return tensor, eigenvalues[-count:].tolist()


def fit_family(source, target, kind, graph=None):
    """Oracle-target fit measures coverage; no deployment access is claimed."""
    side = source.shape[-1]
    if kind == 'GLOBAL':
        basis = torch.ones((1,side,side))
    elif kind == 'COORD':
        q = (torch.arange(side)+.5)/side
        basis = torch.stack([torch.cos(math.pi*i*q)[:,None]*torch.cos(math.pi*j*q)[None,:]
                             for i in range(2) for j in range(2)])
    else:
        basis = F.interpolate(graph[:,None], (side,side), mode='bilinear', align_corners=False)[:,0]
    k = len(basis)
    if kind == 'COORD':
        gb = np.array([1.2]+[.2/(k-1)]*(k-1)); ob = np.array([.12]+[.03/(k-1)]*(k-1))
    else:
        gb = np.array([1.5]+[1.5/max(k-1,1)]*(k-1))
        ob = np.array([.15]+[.15/max(k-1,1)]*(k-1))
    bounds = np.tile(np.r_[gb,ob], 3)
    sx = F.interpolate(source.cpu(), (64,64), mode='area')[0].double().numpy().reshape(3,-1)
    tx = F.interpolate(target.cpu(), (64,64), mode='area')[0].double().numpy().reshape(3,-1)
    phi = F.interpolate(basis[:,None], (64,64), mode='area')[:,0].double().numpy().reshape(k,-1)
    reg = 1e-7
    def objective(normalized):
        theta = (normalized*bounds).reshape(3,2,k)
        gains = np.exp(theta[:,0] @ phi)
        prediction = sx*gains + theta[:,1] @ phi
        residual = prediction-tx
        loss = .5*np.mean(residual**2)+.5*reg*np.mean(normalized**2)
        grad_gain = (residual*sx*gains) @ phi.T / residual.size
        grad_offset = residual @ phi.T / residual.size
        gradient = np.stack([grad_gain,grad_offset],1).reshape(-1)*bounds + reg*normalized/normalized.size
        return loss, gradient
    opt = minimize(objective, np.zeros(6*k), jac=True, method='L-BFGS-B', bounds=[(-1.,1.)]*(6*k),
                   options=dict(maxiter=240, ftol=1e-12, gtol=2e-8, maxls=30))
    loss, gradient = objective(opt.x)
    projected = opt.x-np.clip(opt.x-gradient, -1, 1)
    theta = torch.tensor(opt.x*bounds, dtype=torch.float32).reshape(3,2,k)
    fields = torch.einsum('cjk,khw->cjhw', theta, basis)
    fitted = (source.cpu()*fields[:,0][None].exp()+fields[:,1][None]).clamp(0,1).to(source.device)
    return fitted, dict(kind=kind, parameter_count=6*k, optimizer_success=bool(opt.success),
                        iterations=int(opt.nit), message=str(opt.message), train_loss=float(loss),
                        projected_gradient_inf=float(np.abs(projected).max()),
                        normalized_parameters=opt.x.tolist(), bounds=bounds.tolist(),
                        rgb_mse=float((fitted-target).square().mean()))


class Observers:
    def __init__(self, sc, source):
        self.sc = sc; self.source = source
        self.patch = PatchRelation(sc, source, rank=2)
        self.geometry = Geometry(source)
        self.phase = Phase(source)
        with torch.no_grad():
            f = self.patch.source
            responses = torch.cat([self.patch.features(im)-f for im in photometries(source)],0)
            matrix = responses.flatten(1).T.double()
            gram = matrix.T @ matrix
            values, vectors = torch.linalg.eigh(gram)
            keep = values > values.max()*1e-6
            self.shared = (matrix @ vectors[:,keep]/values[keep].sqrt()).float()
            coefficients = responses.flatten(1) @ self.shared
            self.shared_bound = coefficients.abs().amax(0)*1.5+1e-5
            self.shared_training_residual_max = float(self.shared_residual(responses).norm(dim=-1).max())
            photos = dict(dark=source*.45, white=source*.65+.35,
                          desaturate=source*.4+source.mean(1,keepdim=True)*.6,
                          warm=(source*source.new_tensor([1.1,.9,1.05])[None,:,None,None]).clamp(0,1),
                          spatial_light=(source*torch.exp(.15*torch.linspace(0,2*math.pi,source.shape[-1],device=source.device).cos()[None,None,None,:])).clamp(0,1))
            self.global_controls = {name:self.measure(im) for name,im in photos.items()}
            self.global_budget = max(2.5*max(v['patch_energy'] for v in self.global_controls.values()),1e-12)

    def shared_residual(self, responses):
        flat = responses.flatten(1)
        c = flat @ self.shared
        c = torch.maximum(torch.minimum(c,self.shared_bound),-self.shared_bound) if hasattr(self,'shared_bound') else c
        return flat-c @ self.shared.T

    def correspondence(self, features):
        source = self.patch.source[0]; current = features[0]
        n = len(source); side = int(math.sqrt(n))
        q = (torch.arange(side,device=source.device,dtype=source.dtype)+.5)/side
        yy,xx = torch.meshgrid(q,q,indexing='ij'); pos = torch.stack([xx,yy],-1).reshape(-1,2)
        ds = torch.cdist(pos,pos).square()
        costs = (2-2*source @ current.T).clamp_min(0)
        kernel = -(costs+.05*ds)/.05
        u = torch.zeros(n,device=source.device); v = torch.zeros_like(u)
        for _ in range(80):
            u = -math.log(n)-torch.logsumexp(kernel+v[None,:],1)
            v = -math.log(n)-torch.logsumexp(kernel+u[:,None],0)
        p = (kernel+u[:,None]+v[None,:]).exp()
        matched = (p @ pos)*n; displacement = matched-pos
        center = displacement.mean(0)
        oriented = 2*(displacement-center).square().sum(-1).mean()+center.square().sum()
        return dict(correspondence_feature=float((p*costs).sum()),
                    correspondence_position=float((p*ds).sum()),
                    correspondence_oriented=float(oriented),
                    correspondence_marginal_error=float((p.sum(1)-1/n).abs().max()))

    @torch.no_grad()
    def measure(self, image, features=None):
        f = self.patch.features(image) if features is None else features
        raw = f-self.patch.source
        prof = self.patch.profile(raw)
        excess = (prof.norm(dim=-1)-self.patch.tolerance[None]).clamp_min(0)
        return dict(patch_energy=float(.5*excess.square().mean()),
                    raw_patch_mse=float(raw.square().sum(-1).mean()),
                    shared_feature_mse=float(self.shared_residual(raw).square().sum()/f.shape[1]),
                    geometry=float(self.geometry.loss(self.geometry(image),40.)),
                    phase=float((self.phase(image)-self.phase.source_obs).norm()),
                    rgb_mse=float((image-self.source).square().mean()),
                    **self.correspondence(f))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v1'); ap.add_argument('--natural-only',action='store_true')
    ap.add_argument('--extended',action='store_true')
    args = ap.parse_args(); seed_setup(); out = HERE/'outputs'/args.tag
    protocol = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    local_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                    args=vars(args), scope='Offline controlled witnesses and natural fixed images; no full trajectory quality claim',
                    oracle_reference_mse_tolerance=2e-4, graph_rank=8, fit_maxiter=240,
                    source_budget_multiplier=2.5, global_text_floor_fraction=.5)
    protocol_path = out/('protocol_natural.json' if args.natural_only else 'protocol_probe.json')
    if protocol_path.exists(): raise RuntimeError('Use a fresh immutable tag')
    write(protocol_path,protocol)
    (out/('snapshot_natural.py' if args.natural_only else 'snapshot_probe.py')).write_bytes(Path(__file__).read_bytes())
    sc = CleanScorer(torch.device('cuda:0'),torch.float32,'native')
    started = time.monotonic(); all_rows=[]; records=[]
    if not args.natural_only:
        manifest=json.loads((out/'dataset/manifest.json').read_text())
        for record in manifest['records']:
            scene=record['scene']; dest=out/'scenes'/f'scene{scene:02d}'
            images={name:read_image(HERE/value['file'],sc.device) for name,value in record['variants'].items()}
            source=images['source']; target=images['correct']; observer=Observers(sc,source)
            basis,eigenvalues=graph_basis(observer.patch.source,8)
            fit_info={}
            for kind in ['GLOBAL','COORD','GRAPH']:
                fitted,info=fit_family(source,target,kind,basis)
                images['fit_'+kind]=fitted; fit_info[kind]=info
                save_image(fitted,dest/f'fit_{kind}.png')
            if args.extended:
                from material import fit_material
                large_basis,large_eigenvalues=graph_basis(observer.patch.source,32)
                fitted,info=fit_family(source,target,'GRAPH32',large_basis)
                images['fit_GRAPH32']=fitted;fit_info['GRAPH32']=info
                save_image(fitted,dest/'fit_GRAPH32.png')
                fitted,info=fit_material(source,target)
                images['fit_MATERIAL']=fitted;fit_info['MATERIAL']=info
                save_image(fitted,dest/'fit_MATERIAL.png')
            swapped=record['target_prompt'].replace('left','TEMP_SIDE').replace('right','left').replace('TEMP_SIDE','right')
            enc,_=sc.encode([record['target_prompt'],record['source_prompt'],swapped])
            source_logit=None;source_assignment=None;source_equivariant=None;rows=[]
            with torch.no_grad():
                for name,image in images.items():
                    f=observer.patch.features(image)
                    both_scores=sc.logits_image(torch.cat([image,image.flip(-1)]),enc)
                    scores=both_scores[0]
                    contrast=float(scores[0]-scores[1])
                    assignment=float(scores[0]-scores[2])
                    equivariant=.5*(assignment-float(both_scores[1,0]-both_scores[1,2]))
                    if name=='source':source_logit=contrast;source_assignment=assignment;source_equivariant=equivariant
                    row=dict(scene=scene,split=record['split'],case=name,identical_source_material=record['identical_source_material'],
                             text_contrast=contrast,text_gain=contrast-source_logit,
                             assignment=assignment,assignment_gain=assignment-source_assignment,
                             equivariant_assignment=equivariant,equivariant_gain=equivariant-source_equivariant,
                             oracle_reference_mse=float((image-target).square().mean()),
                             oracle_joint_within_tolerance=float((image-target).square().mean())<=2e-4,
                             constructed_joint_truth=record['variants'].get(name,{}).get('joint_success'),
                             **observer.measure(image,f))
                    rows.append(row);all_rows.append(row)
            info=dict(scene=scene,split=record['split'],global_budget=observer.global_budget,
                      global_controls=observer.global_controls,fits=fit_info,graph_eigenvalues=eigenvalues,
                      shared_feature_rank=observer.shared.shape[1],rows=rows)
            records.append(info);write(dest/'results.json',info)
            write(out/'probe_results.json',dict(protocol=protocol,records=records,rows=all_rows,seconds=time.monotonic()-started))
            displayed=['source','correct','turned','wrong_object','fit_GLOBAL','fit_COORD','fit_GRAPH']
            if args.extended:displayed=['source','correct','wrong_object','turned','fit_GRAPH32','fit_MATERIAL']
            panel([(name,images[name]) for name in displayed],
                  dest/'comparison.png',title=f'scene {scene} {record["split"]}',size=192)
            print('SCENE',scene,record['split'],'legal/budget',rows[1]['patch_energy'],observer.global_budget,
                  'fit_mse',{k:round(v['rgb_mse'],6) for k,v in fit_info.items()},'seconds',round(time.monotonic()-started,1),flush=True)
            del images,observer,basis;gc.collect();torch.cuda.empty_cache()
    natural=[]
    for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        with torch.no_grad(): source=sc.image(data['x'].to(sc.device))
        observer=Observers(sc,source)
        images={'source':source,'global_dark':source*.45,'global_white':source*.65+.35,
                'mirror':source.flip(-1),'shift16':warp(source,tx=16)}
        if pair in ['bear__00','brown_owl__00']:
            for mode in ['BASE','CF','JOINT']:
                saved=torch.load(ROOT/'experiments/null_joint_20261025/outputs/v1_full/full'/pair/'seed0'/mode/'trajectory.pt',weights_only=True)
                with torch.no_grad():images[mode]=sc.image(saved['final'].to(sc.device))
            saved=torch.load(ROOT/'experiments/null_joint_20261025/outputs'/('v5_adaptive_bear' if pair=='bear__00' else 'v5_adaptive_owl')/'full'/pair/'seed0/PATCH_ADAPT/trajectory.pt',weights_only=True)
            with torch.no_grad():images['ADAPT']=sc.image(saved['final'].to(sc.device))
        results=[]
        for name,im in images.items():
            results.append(dict(pair=pair,case=name,**observer.measure(im)))
        natural.append(dict(pair=pair,global_budget=observer.global_budget,rows=results))
        panel(list(images.items()),out/'natural'/f'{pair}.png',title=pair,size=192)
        write(out/'natural_results.json',dict(records=natural,scope='Known geometry controls and retained failures; global photometries are not task-valid local edits'))
        print('NATURAL',pair,'seconds',round(time.monotonic()-started,1),flush=True)
        del observer,images;gc.collect();torch.cuda.empty_cache()
    print('PROBE_COMPLETE',len(records),len(natural),round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH): main()
