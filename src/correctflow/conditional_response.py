"""Consistent single-sample velocity and scale-separated automatic derivatives."""
from contextlib import contextmanager
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel


def norm(t):return float(t.detach().double().norm())

def predicted(model,q,tn,sequence,pooled):
    if q.shape[0]!=1:raise ValueError('The fixed numerical interface requires batch=1')
    dtype=next(model.parameters()).dtype
    with sdpa_kernel(SDPBackend.MATH):
        return model(hidden_states=q.to(dtype),timestep=tn.reshape(1),encoder_hidden_states=sequence,pooled_projections=pooled,return_dict=False)[0]


def velocity(model,q,tn,embeddings,beta):
    sequence,negative,pool,negative_pool=embeddings
    with torch.no_grad():
        vn=predicted(model,q,tn,negative,negative_pool)
        vc=predicted(model,q,tn,sequence,pool)
        return vn+beta*(vc-vn)


def anchor(model,q,tn,t,source,target):
    with torch.no_grad():
        vs=predicted(model,q,tn,source[0],source[2])
        va=predicted(model,q,tn,target[0],target[2])
        return -(1-t)/t*(va-vs)


def response(model,q,tn,embeddings,beta,h,seed=20261005,audit=True,probe_override=None):
    seq,negative,pool,negative_pool=embeddings
    with torch.no_grad():uncond=predicted(model,q,tn,negative,negative_pool)
    def fn(sequence,pooled):
        return uncond+beta*(predicted(model,q,tn,sequence,pooled)-uncond)
    with torch.enable_grad(),sdpa_kernel(SDPBackend.MATH):
        v,pullback=torch.func.vjp(fn,seq,pool)
        raw_probe=(-h*v.detach()) if probe_override is None else probe_override.detach()
        probe_norm=norm(raw_probe)
        if not probe_norm>1e-14:raise ValueError('Weak velocity probe; direction unavailable')
        unit_probe=(raw_probe/probe_norm).detach()
        g=pullback(unit_probe)
        gnorm=float(torch.sqrt(sum(gg.detach().double().square().sum() for gg in g)))
        if not gnorm>1e-14:raise ValueError('Weak text response; direction unavailable')
        unit_g=tuple((gg.detach()/gnorm).detach() for gg in g)
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        _,r=torch.func.jvp(fn,(seq,pool),unit_g)
    metrics={'probe_norm':probe_norm,'normalized_vjp_norm':gnorm,'response_scale':probe_norm*gnorm,'scaled_response_norm':norm(r)}
    if audit:
        gen=torch.Generator('cpu').manual_seed(seed)
        # Same raw tangent as the old audit, to keep its scalar statistic comparable.
        tangents=tuple(torch.randn(vv.shape,generator=gen,dtype=torch.float32).to(device=q.device,dtype=seq.dtype) for vv in (seq,pool))
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            _,jdelta=torch.func.jvp(fn,(seq,pool),tangents)
        lhs=sum((gg.detach().double()*de.double()).sum() for gg,de in zip(g,tangents));rhs=(unit_probe.double()*jdelta.double()).sum()
        tangent_norm=float(torch.sqrt(sum(de.double().square().sum() for de in tangents)))
        denom=gnorm*tangent_norm+norm(unit_probe)*norm(jdelta)
        work=float((unit_probe.double()*r.detach().double()).sum())
        metrics.update(normwise_dual_residual=float(abs(lhs-rhs)/max(denom,1e-30)),scaled_scalar_dual_residual=float(abs(lhs-rhs)/(1+abs(lhs)+abs(rhs))),
            restored_original_scalar_dual_residual=float(abs(lhs-rhs)*probe_norm/(1+(abs(lhs)+abs(rhs))*probe_norm)),energy_identity_normwise=abs(work-gnorm)/max(gnorm+norm(unit_probe)*norm(r),1e-30),energy_identity_relative=abs(work-gnorm)/max(gnorm+abs(work),1e-30),unit_probe_response_work=work)
    tensors={'velocity':v.detach().cpu(),'response':r.detach().cpu(),'unit_probe':unit_probe.cpu(),'g_sequence_normalized_probe':g[0].detach().cpu(),'g_pool_normalized_probe':g[1].detach().cpu()}
    metrics['finite']=all(bool(torch.isfinite(t).all()) for t in tensors.values())
    del pullback,v,g,r
    return metrics,tensors
