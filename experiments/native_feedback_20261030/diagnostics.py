"""Derivative audit + source-only instance binding adjustment.

The adjusted soft graph is a measurement, never an image-update gate. It is
diagnostic and is not admitted as a protected fact without offline validation.
"""
import argparse
import json
import math
import time
import torch
from diffusers import SD3Transformer2DModel
from probe import ROOT, HERE, NativeProbe, write
from analyze import geometry, spatial


def gradient_audit():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = HERE/'outputs/derivative_v1'
    if out.exists():
        raise RuntimeError('Immutable diagnostic exists')
    bank = torch.load(HERE/'outputs/bank_v1/bank.pt', weights_only=True)
    cond = torch.load(HERE/'outputs/bank_v1/conditions.pt', weights_only=True)['conditions']['shape']
    model = SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',
        torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to('cuda:0')
    probe = NativeProbe(model, bank)
    srcpath = ROOT/'experiments/semantic_feedback_20261028/outputs/features_v2/scene00'
    source = torch.load(srcpath/'source.pt', weights_only=True)['y'].to('cuda:0')
    y = torch.load(srcpath/'wrong_head.pt', weights_only=True)['y'].to('cuda:0')
    eps = torch.randn(y.shape, generator=torch.Generator('cpu').manual_seed(19571030)).to(y)
    with torch.no_grad():
        queries = probe.observe(source, eps, .35, cond)['queries']
    def scalar(point, mode):
        obs = probe.observe(point, eps, .35, cond, queries)
        p = obs['layers'][11][mode+'_probability'][0]
        axis = (torch.arange(32, device=p.device, dtype=p.dtype)+.5)/32-.5
        xx = axis.repeat(32)
        # Continuous, nonconstant location measurement; no semantic certification.
        return (p[12]*xx).sum()
    rows = []
    start = time.monotonic()
    for mode in ['dynamic','anchored']:
        q = y.detach().requires_grad_(True)
        value = scalar(q, mode)
        grad = torch.autograd.grad(value, q)[0]
        u = torch.randn(y.shape, generator=torch.Generator('cpu').manual_seed(42571030)).to(y)
        u = u/u.square().mean().sqrt()  # RMS 1 direction, epsilon in latent RMS units
        pred = float((grad*u).sum())
        checks = []
        for delta in [.02, .01, .005]:
            with torch.no_grad():
                fd = float((scalar(y+delta*u, mode)-scalar(y-delta*u, mode))/(2*delta))
            checks.append(dict(epsilon_rms=delta, autodiff=pred, central_difference=fd,
                relative_error=abs(fd-pred)/max(abs(fd),abs(pred),1e-12)))
        rows.append(dict(mode=mode, value=float(value.detach()), gradient_norm=float(grad.norm()), checks=checks))
        print('DERIVATIVE', rows[-1], flush=True)
    write(out/'result.json', dict(rows=rows, seconds=time.monotonic()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated(), latent_input='actual clean wrong_head latent',
        semantic_validity_certified=False, source_queries_detached=True, model_parameter_grads=any(p.grad is not None for p in model.parameters())))


if __name__ == '__main__':
    gradient_audit()
