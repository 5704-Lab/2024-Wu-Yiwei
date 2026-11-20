"""FP64 analytical audit only; no pretrained model or image editing run."""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
gen = torch.Generator().manual_seed(20261012)
OUT = Path(__file__).resolve().parent / "outputs"


def unit(x):
    return x / x.norm()


def response(f, e):
    a = f @ e
    q = f - a * e
    return a, q / q.norm()


def solve(H, A, err, g, lam, zeta=0.0):
    M = (1 + zeta) * H + lam * A.T @ A
    b = lam * A.T @ err
    r = torch.linalg.solve(M, -b)
    nu = 0.0
    if float(g @ r) < 0:
        w = torch.linalg.solve(M, g)
        nu = float(-(g @ r) / (g @ w))
        r = r + nu * w
    return r, nu, M, b


def main():
    rows = []
    for seed in range(40):
        dim, latent = 24, 12
        eye = torch.eye(dim)
        es = unit(torch.randn(dim, generator=gen))
        et = unit(torch.randn(dim, generator=gen))
        fs = unit(torch.randn(dim, generator=gen))
        K = torch.outer(et, es) - torch.outer(es, et)
        R = eye + K + K @ K / (1 + es @ et)
        Ps, Pt = eye - torch.outer(es, es), eye - torch.outer(et, et)
        source_a, ns = response(fs, es)
        m = R @ ns
        rot_error = max(float((R.T @ R - eye).abs().max()),
                        float((R @ es - et).abs().max()),
                        float((Pt @ R - R @ Ps).abs().max()))
        assert rot_error < 1e-12
        free_error = 0.0
        for a in torch.linspace(-0.8, 0.8, 17):
            fa = a * et + torch.sqrt(1 - a * a) * m
            actual_a, na = response(fa, et)
            free_error = max(free_error, abs(float(actual_a - a)),
                             abs(float(fa.norm() - 1)), float((na - m).abs().max()))
        assert free_error < 1e-12

        W = torch.randn(dim, latent, generator=gen) / np.sqrt(latent)
        B = torch.randn(dim, latent, generator=gen) / np.sqrt(latent)
        C = torch.randn(latent, latent, generator=gen) / np.sqrt(latent)
        bias = torch.randn(dim, generator=gen)
        y = torch.randn(latent, generator=gen)

        def feature(q):
            return unit(W @ q + 0.08 * B @ torch.sin(C @ q) + bias)

        def pair(q):
            return response(feature(q), et)

        def loss(q):
            return 0.5 * (pair(q)[1] - m).square().sum()

        f = feature(y)
        a, n = pair(y)
        rho = torch.sqrt(1 - a * a)
        J = torch.autograd.functional.jacobian(feature, y)
        A = (eye - torch.outer(n, n)) @ Pt @ J / rho
        A_ad = torch.autograd.functional.jacobian(lambda q: pair(q)[1], y)
        g = J.T @ et
        g_ad = torch.autograd.functional.jacobian(lambda q: pair(q)[0], y)
        gradK = A.T @ (n - m)
        gradK_ad = torch.autograd.functional.jacobian(loss, y)
        chain_error = max(float((A - A_ad).abs().max()),
                          float((g - g_ad).abs().max()),
                          float((gradK - gradK_ad).abs().max()))
        assert chain_error < 1e-12
        # Independent state finite difference; text vectors never perturbed.
        probe = unit(torch.randn(latent, generator=gen))
        eps = 1e-5
        fd = (pair(y + eps * probe)[1] - pair(y - eps * probe)[1]) / (2 * eps)
        fd_error = float((fd - A @ probe).abs().max())
        assert fd_error < 1e-8
        spherical = et - a * f
        sphere_error = float(((eye - torch.outer(n, n)) @ Pt @ spherical / rho).norm())
        assert sphere_error < 1e-12
        # Potential-space gradient generally does change residual direction.
        latent_cross = float((A @ g).norm())

        Z = torch.randn(latent, latent, generator=gen)
        H = Z.T @ Z / latent + 0.2 * torch.eye(latent)
        lam = 0.7
        r, nu, M, b = solve(H, A, n - m, g, lam)
        active = nu > 0
        kkt = float((M @ r + b - nu * g).abs().max())
        assert kkt < 1e-12 and float(g @ r) > -1e-12
        assert abs(nu * float(g @ r)) < 1e-12
        descent_identity = abs(float(gradK @ r + r @ M @ r / lam))
        assert descent_identity < 1e-12
        assert float(gradK @ r) < 0

        # Solve a genuinely active H-norm budget through its dual multiplier.
        delta = 0.3 * float(torch.sqrt(r @ H @ r))
        lo, hi = 0.0, 1.0
        while True:
            trial, *_ = solve(H, A, n - m, g, lam, hi)
            if float(torch.sqrt(trial @ H @ trial)) <= delta:
                break
            hi *= 2
        for _ in range(70):
            mid = (lo + hi) / 2
            trial, *_ = solve(H, A, n - m, g, lam, mid)
            if float(torch.sqrt(trial @ H @ trial)) > delta:
                lo = mid
            else:
                hi = mid
        rb, nub, Mb, bb = solve(H, A, n - m, g, lam, hi)
        budget_error = abs(float(torch.sqrt(rb @ H @ rb)) - delta)
        assert budget_error < 1e-12
        assert float(g @ rb) > -1e-12
        assert abs(float(gradK @ rb + rb @ Mb @ rb / lam)) < 1e-12

        invg = torch.linalg.solve(H, g)
        den = g @ invg
        Q = torch.eye(latent) - torch.outer(invg, g) / den
        projection_error = max(float((Q @ Q - Q).abs().max()),
                               float((Q.T @ H - H @ Q).abs().max()),
                               float((g @ Q).abs().max()))
        assert projection_error < 1e-11
        v = -Q @ torch.linalg.solve(H, gradK)
        feedback_error = abs(float(gradK @ v + v @ H @ v))
        assert feedback_error < 1e-12

        alpha = 0.05
        correction = 0.0
        if active:
            w = invg / den
            for _ in range(12):
                point = (y + alpha * r + correction * w).detach().requires_grad_(True)
                val = pair(point)[0]
                dval = torch.autograd.grad(val, point)[0] @ w
                gap = float(val - a)
                if abs(gap) < 1e-13:
                    break
                correction -= gap / float(dval)
            accepted = y + alpha * r + correction * w
        else:
            accepted = y + alpha * r
        target_change = float(pair(accepted)[0] - a)
        keep_change = float(loss(accepted) - loss(y))
        assert target_change >= -1e-12 and keep_change < 0

        rows.append(dict(seed=seed, rotation_error=rot_error, free_strength_error=free_error,
                         chain_error=chain_error, finite_difference_error=fd_error,
                         sphere_task_null_error=sphere_error, latent_cross_coupling=latent_cross,
                         task_constraint_active=active, kkt_error=kkt,
                         descent_identity_error=descent_identity, budget_error=budget_error,
                         projection_error=projection_error, continuous_feedback_error=feedback_error,
                         finite_target_change=target_change, finite_keep_change=keep_change,
                         finite_retraction_coefficient=correction, source_pair_strength=float(source_a)))
    assert any(r["task_constraint_active"] for r in rows)
    assert any(not r["task_constraint_active"] for r in rows)
    assert max(r["latent_cross_coupling"] for r in rows) > 1e-3
    summary = dict(scope="Synthetic FP64 mathematics only; no pretrained/image validation",
                   cases=len(rows), active_constraints=sum(r["task_constraint_active"] for r in rows),
                   maxima={k:max(abs(r[k]) for r in rows) for k in rows[0]
                           if k.endswith("error")},
                   max_latent_cross_coupling=max(r["latent_cross_coupling"] for r in rows),
                   checks=["text-coordinate orthogonal rotation", "free target strength at zero keep loss",
                           "state chain rule and independent finite difference", "feature-space task null",
                           "latent cross-coupling counterexample", "QP KKT and descent",
                           "active dual norm budget", "weighted projection and feedback identity",
                           "finite step with active-score retraction"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
