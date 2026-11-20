"""Positive shared material-gain surrogate; no target ROI or velocity mask.

Source chromatic response functions and fixed Bernstein coordinate functions
define all coefficients. Oracle target fitting is diagnostic only. This is a
restricted photometric model, not an inferred physical scene or free-text binder.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


def material_basis(source, colors=8, degree=7):
    rgb=source[0].detach().cpu().double().numpy()
    chroma=rgb/np.maximum(rgb.sum(0,keepdims=True),1e-3)
    samples=chroma[:,::8,::8].reshape(3,-1).T
    centers=[samples[np.argmax(np.sum((samples-1/3)**2,axis=1))]]
    for _ in range(colors-1):
        distances=np.min(np.sum((samples[:,None]-np.array(centers)[None])**2,axis=-1),axis=1)
        centers.append(samples[np.argmax(distances)])
    centers=np.array(centers)
    ds=np.sum((chroma.transpose(1,2,0)[:,:,None]-centers[None,None])**2,axis=-1)
    logits=-ds/(2*.025**2);logits-=logits.max(-1,keepdims=True)
    chromatic=np.exp(logits);chromatic/=chromatic.sum(-1,keepdims=True)
    h,w=source.shape[-2:];xx=(np.arange(w)+.5)/w
    coordinates=np.stack([math.comb(degree,k)*xx**k*(1-xx)**(degree-k) for k in range(degree+1)])
    basis=np.einsum('hwc,rw->crhw',chromatic,coordinates).reshape(colors*(degree+1),h,w)
    assert np.max(np.abs(basis.sum(0)-1))<1e-10
    return torch.tensor(basis,dtype=torch.float32),centers.tolist()


def fit_material(source,target,maxiter=500):
    basis,centers=material_basis(source)
    k=len(basis)
    sx=F.interpolate(source.cpu(),(96,96),mode='area')[0].double().numpy().reshape(3,-1)
    tx=F.interpolate(target.cpu(),(96,96),mode='area')[0].double().numpy().reshape(3,-1)
    phi=F.interpolate(basis[:,None],(96,96),mode='area')[:,0].double().numpy().reshape(k,-1)
    params=[];logs=[];regularization=1e-9
    for c in range(3):
        design=(phi*sx[c][None]).T
        gram=design.T@design/design.shape[0]+regularization*np.eye(k)
        rhs=design.T@tx[c]/design.shape[0]+regularization*np.ones(k)
        objective=lambda a:(.5*a@gram@a-rhs@a,gram@a-rhs)
        opt=minimize(objective,np.ones(k),jac=True,method='L-BFGS-B',bounds=[(.05,12.)]*k,
                     options=dict(maxiter=maxiter,ftol=1e-14,gtol=1e-9,maxls=40))
        _,grad=objective(opt.x)
        pg=opt.x-np.clip(opt.x-grad,.05,12.)
        logs.append(dict(success=bool(opt.success),iterations=int(opt.nit),message=str(opt.message),projected_gradient_inf=float(np.max(np.abs(pg)))))
        params.append(opt.x)
    coefficients=torch.tensor(np.stack(params),dtype=torch.float32)
    gains=torch.einsum('ck,khw->chw',coefficients,basis)
    predicted=(source.cpu()*gains[None]).clamp(0,1).to(source.device)
    return predicted,dict(kind='MATERIAL',parameter_count=3*k,colors=8,bernstein_degree=7,
                          minimum_gain=float(gains.min()),maximum_gain=float(gains.max()),
                          identity_max_error=float((source.cpu()*basis.sum(0)[None,None]-source.cpu()).abs().max()),
                          chromatic_centers=centers,coefficients=coefficients.tolist(),solvers=logs,
                          rgb_mse=float((predicted-target).square().mean()),
                          projected_gradient_inf=max(r['projected_gradient_inf'] for r in logs))
