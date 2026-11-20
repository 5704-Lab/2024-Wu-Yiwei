"""Content matching first; explicit source-frame deformation second.

No transform truth, geometry-to-be-protected, or correct target is used in match.
Mutual support is a heuristic; it is NOT a calibrated semantic confidence bound.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def coordinates(n,device='cpu'):
    side=math.isqrt(n);assert side*side==n
    a=(torch.arange(side,device=device,dtype=torch.float32)+.5)/side-.5
    yy,xx=torch.meshgrid(a,a,indexing='ij')
    return torch.stack([xx,yy],-1).reshape(-1,2)


def descriptor(x,source,mode):
    if mode=='centered':x=x-source.mean(0,keepdim=True)
    return F.normalize(x,dim=-1)


def role_distribution(row,masks):
    o=F.normalize(row['O'][0],dim=-1)
    c=torch.stack([row['C'][0,m.to(row['C'].device)].mean(0) for m in masks])
    return (o@F.normalize(c,dim=-1).T/.1).softmax(-1)


def estimate_frame(p,q,w):
    p=p.double();q=q.double();w=w.double()
    if float(w.sum())<1e-10:return dict(valid=False,reason='no_support')
    w=w/w.sum();eff=float(1/w.square().sum())
    if eff<8:return dict(valid=False,reason='low_effective_support',effective_points=eff)
    mp=(w[:,None]*p).sum(0);mq=(w[:,None]*q).sum(0)
    pc=p-mp;qc=q-mq
    cov=(pc.T*w)@pc
    eig=torch.linalg.eigvalsh(cov)
    if float(eig.min())<1e-5:return dict(valid=False,reason='rank_deficient',effective_points=eff)
    a=torch.linalg.solve(cov,(pc.T*w)@qc).T
    t=mq-a@mp
    u,s,vh=torch.linalg.svd(a)
    det=float(torch.linalg.det(a));rotation=u@vh
    reflection=det<0
    fit=((qc-pc@a.T).square().sum(-1)*w).sum().sqrt()
    return dict(valid=bool(float(s.min())>1e-3),effective_points=eff,source_cov_min=float(eig.min()),
        matrix=a.cpu().tolist(),translation=t.cpu().tolist(),determinant=det,reflection=reflection,
        rotation_deg=None if reflection else float(torch.atan2(rotation[1,0],rotation[0,0])*180/torch.pi),
        stretch_eigenvalues=s.cpu().tolist(),fit_rmse=float(fit))


def transport(similarity,method):
    if method=='nn':logp=similarity
    elif method=='sinkhorn':
        kernel=similarity/.05
        u=torch.zeros(kernel.shape[0],device=kernel.device);v=u.clone()
        logmass=-math.log(kernel.shape[0])
        for _ in range(40):
            u=logmass-torch.logsumexp(kernel+v[None],dim=1)
            v=logmass-torch.logsumexp(kernel+u[:,None],dim=0)
        logp=kernel+u[:,None]+v[None]
    else:raise ValueError(method)
    top=logp.topk(2,dim=1)
    j=top.indices[:,0];back=logp.argmax(0)
    mutual=back[j]==torch.arange(len(j),device=j.device)
    return j,mutual,top.values[:,0]-top.values[:,1],top.indices[:,1]


def measure(j,mutual,gap,alternative,role_weights):
    p=coordinates(len(j),j.device).double();q=p[j];w=mutual.double()
    weights=[w,w*role_weights]
    for y in [-.375,-.125,.125,.375]:
        for x in [-.375,-.125,.125,.375]:
            weight=torch.exp(-((p-torch.tensor([x,y],device=p.device))**2).sum(-1)/(2*.16**2))
            weights.append(w*weight)
    weights.append(w)
    weights=torch.stack(weights);total=weights.sum(-1);ww=weights/total[:,None].clamp_min(1e-20)
    qq=q[None].expand(19,-1,-1).clone();qq[-1]=p[alternative]
    mp=torch.einsum('kn,nd->kd',ww,p);mq=torch.einsum('kn,knd->kd',ww,qq)
    pc=p[None]-mp[:,None];qc=qq-mq[:,None]
    cov=torch.einsum('kn,kni,knj->kij',ww,pc,pc)
    cross=torch.einsum('kn,kni,knj->kij',ww,qc,pc)
    eigen=torch.linalg.eigvalsh(cov);effective=1/ww.square().sum(-1).clamp_min(1e-20)
    support=(total>1e-10)&(effective>=8)&(eigen[:,0]>=1e-5)
    cov=torch.where(support[:,None,None],cov,torch.eye(2,device=p.device,dtype=p.dtype)[None])
    a=torch.linalg.solve(cov,cross.transpose(1,2)).transpose(1,2)
    t=mq-torch.einsum('kij,kj->ki',a,mp)
    u,s,vh=torch.linalg.svd(a);rot=u@vh;det=torch.linalg.det(a)
    angle=torch.atan2(rot[:,1,0],rot[:,0,0])*180/torch.pi
    residual=qc-torch.einsum('kij,knj->kni',a,pc)
    rmse=(residual.square().sum(-1)*ww).sum(-1).sqrt()
    pack=torch.cat([support[:,None],effective[:,None],eigen[:,:1],a.flatten(1),t,det[:,None],angle[:,None],s,rmse[:,None]],1).cpu().numpy()
    frames=[]
    for z in pack:
        valid=bool(z[0] and z[12]>1e-3)
        frames.append(dict(valid=valid,effective_points=float(z[1]),source_cov_min=float(z[2]),
            matrix=z[3:7].reshape(2,2).tolist(),translation=z[7:9].tolist(),determinant=float(z[9]),
            reflection=bool(z[9]<0),rotation_deg=None if z[9]<0 else float(z[10]),
            stretch_eigenvalues=z[11:13].tolist(),fit_rmse=float(z[13])))
    local=[abs(r['rotation_deg']) for r in frames[2:18] if r['valid'] and r['rotation_deg'] is not None]
    return dict(mutual_fraction=float(w.mean()),mean_match_gap=float(gap.mean()),global_frame=frames[0],
        object_frame=frames[1],alternative_frame=frames[-1],source_object_weight_entropy=float(-(role_weights/role_weights.sum()).mul((role_weights/role_weights.sum()).clamp_min(1e-20).log()).sum()),
        local_valid_fraction=len(local)/16,local_rotation_abs_p90=float(np.percentile(local,90)) if local else None,
        local_rotation_abs_max=max(local) if local else None)


def evaluate(j,mutual,metrics,truth):
    """Truth is used only AFTER matching and frame fitting."""
    if 'matrix' not in truth:return None
    p=coordinates(len(j),j.device)
    a=torch.tensor(truth['matrix'],device=p.device);t=torch.tensor(truth['translation'],device=p.device)
    expected=p@a.T+t
    valid=(expected.abs().max(-1).values<=.5-1/64)
    dist=(p[j]-expected).norm(dim=-1)
    good=valid&mutual
    out=dict(visible_source_fraction=float(valid.float().mean()),pck32=float((dist[valid]<=32/512).float().mean()),
        median_epe_pixels=float(dist[valid].median()*512),mean_epe_pixels=float(dist[valid].mean()*512),
        mutual_visible_fraction=float(good.float().sum()/valid.float().sum()),
        mutual_pck32=float((dist[good]<=32/512).float().mean()) if good.any() else None)
    for key in ['global_frame','object_frame']:
        r=metrics[key]
        if not r.get('valid'):continue
        out[key+'_matrix_error']=float(np.linalg.norm(np.array(r['matrix'])-np.array(truth['matrix'])))
        out[key+'_translation_error']=float(np.linalg.norm(np.array(r['translation'])-np.array(truth['translation'])))
        if np.linalg.det(truth['matrix'])>0 and r['rotation_deg'] is not None:
            u,_,vh=np.linalg.svd(truth['matrix']);rr=u@vh
            angle=math.degrees(math.atan2(rr[1,0],rr[0,0]))
            out[key+'_angle_error_deg']=abs((r['rotation_deg']-angle+180)%360-180)
    return out
