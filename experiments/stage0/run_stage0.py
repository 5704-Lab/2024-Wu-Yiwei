"""CPU/FP64 mathematical audit with exact densities and an independent QP solver."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cvxpy as cp
import numpy as np
from scipy.integrate import quad
import torch
from threadpoolctl import threadpool_limits

from correctflow.local_correction import (
    kkt_residuals, low_rank_inverse_action, objective, solve_zero_threshold_qp,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class Mixture:
    def __init__(self, components, dimension, seed):
        gen = torch.Generator().manual_seed(seed)
        self.means = torch.randn(components, dimension, generator=gen) * (0.6 if seed % 2 else 2.0)
        self.condition_map = torch.randn(components, 3, generator=gen)
        self.sigma0 = 0.4

    def log_components(self, q, t, e):
        variance = (1 - t) ** 2 * self.sigma0 ** 2 + t ** 2
        residual = q[None, :] - (1 - t) * self.means
        log_gaussian = -q.numel() / 2 * math.log(2 * math.pi * variance) - residual.square().sum(-1) / (2 * variance)
        return torch.log_softmax(self.condition_map @ e, 0) + log_gaussian

    def logp(self, q, t, e):
        return torch.logsumexp(self.log_components(q, t, e), 0)

    def score_closed(self, q, t, e):
        variance = (1 - t) ** 2 * self.sigma0 ** 2 + t ** 2
        posterior = torch.softmax(self.log_components(q, t, e), 0)
        return (posterior[:, None] * ((1 - t) * self.means - q) / variance).sum(0)

    def mean_closed(self, q, t, e):
        variance = (1 - t) ** 2 * self.sigma0 ** 2 + t ** 2
        posterior = torch.softmax(self.log_components(q, t, e), 0)
        component_mean = self.means + (1 - t) * self.sigma0 ** 2 / variance * (q - (1 - t) * self.means)
        return (posterior[:, None] * component_mean).sum(0)

    def velocity(self, q, t, e):
        return (q - self.mean_closed(q, t, e)) / t

    def anchor(self, q, t, es, ea):
        return -(1 - t) / t * (self.velocity(q, t, ea) - self.velocity(q, t, es))

    def curvature_bound(self, t):
        diameter = torch.cdist((1 - t) * self.means, (1 - t) * self.means).max()
        variance = (1 - t) ** 2 * self.sigma0 ** 2 + t ** 2
        return float(diameter ** 2 / (2 * variance ** 2))


def relative_error(actual, expected):
    return float(torch.linalg.vector_norm(actual - expected) / (1 + torch.linalg.vector_norm(expected)))


def midpoint_work(model, q, displacement, t, es, ea, m):
    return sum(float(model.anchor(q + (j + 0.5) / m * displacement, t, es, ea) @ displacement) for j in range(m)) / m


def run_analytic():
    rows = []
    for k in (2, 4):
        for dimension in (2, 16):
            for seed in range(20):
                model = Mixture(k, dimension, seed)
                gen = torch.Generator().manual_seed(seed + 9000)
                es = torch.randn(3, generator=gen)
                ea = es.clone() if seed % 5 == 0 else torch.randn(3, generator=gen)
                x = model.means[seed % k] + model.sigma0 * torch.randn(dimension, generator=gen)
                eps = torch.randn(dimension, generator=gen)
                offset = torch.randn(dimension, generator=gen) * 0.15
                for t in (0.2, 0.5, 0.8):
                    q = (1 - t) * x + t * eps
                    qa = q + offset
                    v = model.velocity(q, t, es)
                    score = torch.func.grad(lambda qq: model.logp(qq, t, es))(q)
                    score_from_v = -(q + (1 - t) * v) / t
                    anchor = model.anchor(qa, t, es, ea)
                    grad_ratio = torch.func.grad(lambda qq: model.logp(qq, t, ea) - model.logp(qq, t, es))(qa)
                    errors = {
                        "score_velocity_error": relative_error(score_from_v, score),
                        "score_closed_error": relative_error(model.score_closed(q, t, es), score),
                        "mean_velocity_error": relative_error(q - t * v, model.mean_closed(q, t, es)),
                        "density_ratio_error": relative_error(anchor, grad_ratio),
                    }
                    for condition, state in ((es, q), (ea, qa)):
                        fn = lambda ee: model.velocity(state, t, ee)
                        velocity, vjp_fn = torch.func.vjp(fn, condition)
                        h = 0.03
                        probe = (-h * velocity).detach()
                        g = vjp_fn(probe)[0]
                        explicit_j = torch.func.jacfwd(fn)(condition)
                        explicit_g = explicit_j.T @ probe
                        tangent = torch.randn(3, generator=gen)
                        _, jvp_tangent = torch.func.jvp(fn, (condition,), (tangent,))
                        lhs, rhs = g @ tangent, probe @ jvp_tangent
                        _, response = torch.func.jvp(fn, (condition,), (g.detach(),))
                        energy_grad = torch.func.grad(lambda ee: -h / 2 * fn(ee).square().sum())(condition)
                        random_probe = torch.randn(dimension, generator=gen)
                        random_g = vjp_fn(random_probe)[0]
                        _, random_response = torch.func.jvp(fn, (condition,), (random_g.detach(),))
                        errors["vjp_explicit_error"] = max(errors.get("vjp_explicit_error", 0), relative_error(g, explicit_g))
                        errors["vjp_energy_error"] = max(errors.get("vjp_energy_error", 0), relative_error(g, energy_grad))
                        errors["response_explicit_error"] = max(errors.get("response_explicit_error", 0), relative_error(response, explicit_j @ explicit_j.T @ probe))
                        errors["dual_error"] = max(errors.get("dual_error", 0), float(abs(lhs - rhs) / (1 + abs(lhs) + abs(rhs))))
                        errors["psd_violation"] = max(errors.get("psd_violation", 0), max(-float(random_probe @ random_response), 0.0))
                    displacement = torch.randn(dimension, generator=gen) * 0.04
                    exact_work = float(model.logp(qa + displacement, t, ea) - model.logp(qa + displacement, t, es) - model.logp(qa, t, ea) + model.logp(qa, t, es))
                    numerical_work, quad_error = quad(
                        lambda eta: float(model.anchor(qa + eta * displacement, t, es, ea) @ displacement),
                        0, 1, epsabs=1e-12, epsrel=1e-12,
                    )
                    errors["integral_identity_error"] = abs(numerical_work - exact_work) / (1 + abs(exact_work))
                    bound_violations = []
                    midpoint_errors = {}
                    for m in (1, 2, 4, 8):
                        estimate = midpoint_work(model, qa, displacement, t, es, ea, m)
                        bound = model.curvature_bound(t) * float(displacement @ displacement) / (4 * m)
                        midpoint_errors[str(m)] = abs(estimate - exact_work)
                        bound_violations.append(max(abs(estimate - exact_work) - bound, 0.0))
                    errors["midpoint_bound_violation"] = max(bound_violations)
                    rows.append({
                        "components": k, "dimension": dimension, "seed": seed, "t": t,
                        "same_condition": bool(torch.equal(es, ea)),
                        **errors, "midpoint_errors": midpoint_errors, "quad_error_estimate": quad_error,
                    })
    return rows


def run_qp():
    rows = []
    for dimension in (2, 16, 64):
        for seed in range(20):
            rng = np.random.default_rng(seed + 100 * dimension)
            drift, a = rng.normal(size=dimension), rng.normal(size=dimension)
            source = rng.normal(size=dimension)
            target = source.copy() if seed % 4 == 0 else rng.normal(size=dimension)
            if seed % 7 == 0:
                source = None
            if seed % 9 == 0:
                target = None
            qs = np.zeros((dimension, dimension)) if source is None else np.outer(source, source) / (source @ source)
            qa = np.zeros((dimension, dimension)) if target is None else np.outer(target, target) / (target @ target)
            ls = (0, 1, 8, 100)[seed % 4]
            lu = 0 if target is None else (1, 2, 10, 0)[seed % 4]
            weight = np.eye(dimension) + ls * qs + lu * (np.eye(dimension) - qa)
            alpha, rho = (0.1, 0.3, 1.0)[seed % 3], 1.0
            y = rng.normal(size=dimension)
            direct = np.linalg.solve(alpha * np.eye(dimension) + rho * weight, y)
            lowrank = low_rank_inverse_action(y, source, target, alpha=alpha, rho=rho, lambda_s=ls, lambda_u=lu)
            lowrank_error = float(np.linalg.norm(lowrank - direct) / (1 + np.linalg.norm(direct)))
            for factor in (0.0, 0.02, 0.2, 2.0):
                radius = factor * np.linalg.norm(drift)
                result = solve_zero_threshold_qp(drift, a, weight, alpha=alpha, rho=rho, radius=radius)
                w = result.displacement
                residual = kkt_residuals(drift, weight, result, radius, alpha, rho) if radius else {}
                residual["source_distance_violation"] = max(float(np.linalg.norm(drift + w) ** 2 - drift @ drift), 0.0) / (1 + drift @ drift)
                residual["objective_violation"] = max(objective(drift, w, weight, alpha, rho) - objective(drift, np.zeros(dimension), weight, alpha, rho), 0.0) / (1 + objective(drift, np.zeros(dimension), weight, alpha, rho))
                residual["lowrank_inverse_error"] = lowrank_error
                residual["scaled_source_violation"] = max(
                    max(float(np.linalg.norm(drift + scale * w) ** 2 - drift @ drift), 0.0) / (1 + drift @ drift)
                    for scale in (0.0, 0.25, 0.5, 1.0)
                )
                oracle_status = "zero_radius_exact"
                if radius:
                    variable = cp.Variable(dimension)
                    normal = a / np.linalg.norm(a)
                    problem = cp.Problem(
                        cp.Minimize(alpha / 2 * cp.sum_squares(drift + variable) + rho / 2 * cp.quad_form(variable, cp.psd_wrap(weight))),
                        [normal @ variable >= 0, cp.norm(variable) <= radius],
                    )
                    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)
                    oracle_status = problem.status
                    if oracle_status not in ("optimal", "optimal_inaccurate"):
                        raise RuntimeError(f"independent solver failed: {oracle_status}")
                    oracle = np.asarray(variable.value)
                    residual["oracle_objective_gap"] = abs(objective(drift, w, weight, alpha, rho) - objective(drift, oracle, weight, alpha, rho)) / (1 + abs(problem.value))
                    residual["oracle_solution_error"] = float(np.linalg.norm(w - oracle) / (1 + np.linalg.norm(oracle)))
                rows.append({
                    "dimension": dimension, "seed": seed, "radius_factor": factor,
                    "halfspace_active": result.halfspace_multiplier > 1e-10,
                    "ball_active": result.ball_multiplier > 1e-10,
                    "condition_number": float(np.linalg.cond(alpha * np.eye(dimension) + rho * weight)),
                    "oracle_status": oracle_status, **residual,
                })
    return rows


def run_counterexamples():
    e = torch.tensor([0.7])
    fn = lambda ee: torch.stack((torch.cos(ee[0]), torch.sin(ee[0])))
    velocity, vjp = torch.func.vjp(fn, e)
    radial = vjp(velocity.detach())[0]
    task = vjp(torch.tensor([1.0, 0.0]))[0]
    drift = np.array([0.0, 1.0])
    result = solve_zero_threshold_qp(drift, np.array([1.0, 0.0]), np.eye(2), alpha=1.0, radius=2.0)
    w = result.displacement
    quadratic_gain = -w[1] ** 2 + w[0]
    quadratic_work = sum(np.array([1.0, -2 * eta * w[1]]) @ w for eta in (0.25, 0.75)) / 2
    # A known aliasing counterexample demonstrates that two agreeing quadratures
    # cannot certify an unknown neural field's unsampled curvature.
    alias_a = lambda eta: -1 + 2 * math.cos(8 * math.pi * eta)
    alias_m1 = alias_a(0.5)
    alias_m2 = (alias_a(0.25) + alias_a(0.75)) / 2
    alias_true_gain = -1.0
    known_bound = 16 * math.pi / 8  # L*||w||^2/(4*m), with m=2, ||w||=1
    weak = solve_zero_threshold_qp(np.ones(2), np.zeros(2), np.eye(2), radius=1.0)
    return {
        "rotation_radial_norm": float(torch.linalg.vector_norm(radial)),
        "rotation_task_norm": float(torch.linalg.vector_norm(task)),
        "quadratic_qp_displacement": w.tolist(),
        "quadratic_exact_gain": float(quadratic_gain),
        "quadratic_midpoint_work": float(quadratic_work),
        "quadratic_check_rejects": bool(quadratic_work < 0),
        "alias_m1": alias_m1,
        "alias_m2": alias_m2,
        "alias_exact_gain": alias_true_gain,
        "alias_heuristic_false_acceptance": bool(alias_m1 == alias_m2 and alias_m2 > 0),
        "alias_certified_lower_bound": alias_m2 - known_bound,
        "weak_anchor_fallback_zero": bool(np.all(weak.displacement == 0)),
        "interpretation": "Aliasing is an expected limitation diagnostic, not a claimed neural certification or semantic result.",
    }


def main():
    start = time.monotonic()
    output = ROOT / "experiments/stage0/outputs"
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        analytic = run_analytic()
        print(f"analytic cases completed: {len(analytic)}", flush=True)
        qp = run_qp()
        print(f"QP cases completed: {len(qp)}", flush=True)
        examples = run_counterexamples()
    maxima = defaultdict(float)
    for row in analytic + qp:
        for key, value in row.items():
            if key.endswith(("_error", "_violation", "_complementarity", "_gap")) or key in ("stationarity", "primal"):
                maxima[key] = max(maxima[key], float(value))
    gates = {
        "analytic_equalities": all(maxima[k] <= 1e-8 for k in (
            "score_velocity_error", "score_closed_error", "mean_velocity_error",
            "density_ratio_error", "vjp_explicit_error", "vjp_energy_error",
            "response_explicit_error", "dual_error", "integral_identity_error",
        )),
        "psd_and_integral_bounds": maxima["psd_violation"] <= 1e-8 and maxima["midpoint_bound_violation"] <= 1e-8,
        "qp_kkt_and_invariants": all(maxima[k] <= 1e-7 for k in (
            "stationarity", "primal", "halfspace_complementarity", "ball_complementarity",
            "source_distance_violation", "objective_violation", "scaled_source_violation",
        )),
        "independent_qp_objective": maxima["oracle_objective_gap"] <= 1e-7,
        "lowrank_inverse": maxima["lowrank_inverse_error"] <= 1e-8,
        "counterexamples": (
            examples["rotation_radial_norm"] <= 1e-12
            and examples["rotation_task_norm"] > 0.1
            and examples["quadratic_check_rejects"]
            and examples["alias_heuristic_false_acceptance"]
            and examples["alias_certified_lower_bound"] < 0
            and examples["weak_anchor_fallback_zero"]
        ),
    }
    summary = {
        "stage": 0, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "dtype": "float64", "training_steps": 0,
        "analytic_cases": len(analytic), "qp_cases": len(qp),
        "independent_qp_cases": sum(r["radius_factor"] != 0 for r in qp),
        "max_errors": dict(maxima),
        "gates": gates, "passed": all(gates.values()),
        "counterexamples": examples,
        "elapsed_seconds": time.monotonic() - start,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy", "cvxpy", "clarabel")},
        "code_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "src/correctflow/local_correction.py"]
        },
        "scope": "Only mathematical/computational correctness; no real-image editing quality or originality validation.",
    }
    for name, records in (("analytic_cases", analytic), ("qp_cases", qp)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
