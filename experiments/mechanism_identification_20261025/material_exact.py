"""Exact native-pixel convex coverage fit for the unchanged material family."""
import numpy as np
from scipy.optimize import lsq_linear
import torch

from material import material_basis


def fit_material_exact(source,target,global_offset=False):
    basis,centers=material_basis(source)
    phi=basis.double().numpy().reshape(len(basis),-1)
    sx=source[0].detach().cpu().double().numpy().reshape(3,-1)
    tx=target[0].detach().cpu().double().numpy().reshape(3,-1)
    n=sx.shape[1];k=len(basis);regularization=1e-9
    params=[];logs=[]
    for c in range(3):
        design=phi*sx[c][None]
        if global_offset:
            design=np.r_[design,np.ones((1,n))]
        count=design.shape[0]
        prior=np.r_[np.ones(k),0.] if global_offset else np.ones(k)
        gram=design@design.T/n+regularization*np.eye(count)
        rhs=design@tx[c]/n+regularization*prior
        lower=np.linalg.cholesky(gram)
        transformed=np.linalg.solve(lower,rhs)
        lo=np.r_[np.full(k,.05),-.25] if global_offset else np.full(k,.05)
        hi=np.r_[np.full(k,12.),.75] if global_offset else np.full(k,12.)
        opt=lsq_linear(lower.T,transformed,bounds=(lo,hi),method='bvls',tol=1e-11,max_iter=500)
        gradient=gram@opt.x-rhs
        projected=opt.x-np.clip(opt.x-gradient,lo,hi)
        params.append(opt.x)
        logs.append(dict(success=bool(opt.success),iterations=int(opt.nit),status=int(opt.status),
                         projected_gradient_inf=float(np.abs(projected).max()),optimality=float(opt.optimality)))
    coefficients=torch.tensor(np.stack(params),dtype=torch.float32)
    gains=torch.einsum('ck,khw->chw',coefficients[:,:k],basis)
    offsets=coefficients[:,k] if global_offset else torch.zeros(3)
    unclamped=source.cpu()*gains[None]+offsets[None,:,None,None]
    predicted=unclamped.clamp(0,1).to(source.device)
    return predicted,dict(kind='MATERIAL_OFFSET' if global_offset else 'MATERIAL_EXACT',parameter_count=3*(k+int(global_offset)),colors=8,bernstein_degree=7,
                          minimum_gain=float(gains.min()),maximum_gain=float(gains.max()),
                          identity_max_error=float((source.cpu()*basis.sum(0)[None,None]-source.cpu()).abs().max()),
                          chromatic_centers=centers,coefficients=coefficients.tolist(),global_offsets=offsets.tolist(),solvers=logs,
                          rgb_mse=float((predicted-target).square().mean()),
                          unclamped_rgb_mse=float((unclamped.to(target.device)-target).square().mean()),
                          projected_gradient_inf=max(r['projected_gradient_inf'] for r in logs))
