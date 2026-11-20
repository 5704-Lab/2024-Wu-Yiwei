"""Low-dimensional reference operators for the 2026-11-06 theory study.

No FLUX model is loaded. Bounds are caller-supplied mathematical assumptions,
not estimated or certified automatically. The functions deliberately distinguish
an actual finite-value acceptance check from a local convex model.
"""

from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import numpy as np
import torch


def clean_field(y: torch.Tensor, sigma: float, source_velocity: Callable,
                target_velocity: Callable) -> torch.Tensor:
    """Same-state, deterministic RF query; exact heat-score ratio only ideally."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be strictly positive and finite")
    alpha = 1 / (1 + sigma)
    time = sigma / (1 + sigma)
    query = alpha * y
    return alpha**2 / time * (source_velocity(query, time) - target_velocity(query, time))


def radial_potential(y: torch.Tensor, source: torch.Tensor, field: Callable,
                     quadrature_nodes: int = 8) -> torch.Tensor:
    """A fully specified scalar of the actual field, with midpoint quadrature.

Do NOT detach field outputs: the claimed gradient includes their derivatives.
Keep source, quadrature rule, field, scales and weights fixed during a run.
"""
    if quadrature_nodes < 1 or int(quadrature_nodes) != quadrature_nodes:
        raise ValueError("quadrature_nodes must be a positive integer")
    if source.shape != y.shape:
        raise ValueError("source and y must have identical shapes")
    source = source.detach()
    delta = y - source
    terms = [torch.sum(field(source + ((j + 0.5) / quadrature_nodes) * delta) * delta)
             for j in range(quadrature_nodes)]
    return torch.stack(terms).mean()


def value_gradient(y: np.ndarray, scalar: Callable) -> tuple[float, np.ndarray]:
    variable = torch.tensor(np.asarray(y), dtype=torch.float64, requires_grad=True)
    value = scalar(variable)
    if value.ndim != 0:
        raise ValueError("the objective must be a scalar")
    gradient, = torch.autograd.grad(value, variable)
    return float(value.detach()), gradient.detach().numpy()


def defect_certificate(times, residual_norms, field_lipschitz_bound: float,
                       residual_lipschitz_bound: float):
    """Conditional sup-norm source-path bound, given valid uniform L and M.

Sampling residuals alone is not a certificate. M must bound the continuous-time
Lipschitz constant of the residual on the entire sampled interval.
"""
    times = np.asarray(times, dtype=float)
    residuals = np.asarray(residual_norms, dtype=float)
    L, M = field_lipschitz_bound, residual_lipschitz_bound
    if times.ndim != 1 or len(times) < 2 or residuals.shape != times.shape:
        raise ValueError("times and residual_norms must be matching vectors")
    if not np.all(np.isfinite(np.r_[times, residuals, L, M])):
        raise ValueError("all inputs must be finite")
    if times[0] != 0 or np.any(np.diff(times) <= 0) or np.any(residuals < 0) or min(L, M) < 0:
        raise ValueError("invalid mesh, residual norms, or bounds")
    delta = float(residuals.max() + M * np.diff(times).max() / 2)
    radius = delta * (np.expm1(L * times) / L if L > 0 else times)
    return dict(conditional_residual_bound=delta, conditional_path_radii=radius,
                assumes_uniform_field_L=L, assumes_uniform_residual_M=M)


def propagated_edit_bound(step_sizes, source_path_errors, source_L, target_L,
                          displacement_scales=None):
    """Two Euler edits with the same steps, fields and initial clean state."""
    hs, es = np.asarray(step_sizes, float), np.asarray(source_path_errors, float)
    scales = np.ones_like(hs) if displacement_scales is None else np.asarray(displacement_scales, float)
    if hs.shape != es.shape or hs.shape != scales.shape or hs.ndim != 1:
        raise ValueError("step sizes, errors and scales must have matching vector shapes")
    if not np.all(np.isfinite(np.r_[hs, es, scales, source_L, target_L])) or min(source_L, target_L) < 0:
        raise ValueError("finite nonnegative Lipschitz bounds required")
    if np.any(hs < 0) or np.any(es < 0):
        raise ValueError("step sizes and source errors must be nonnegative")
    e, history = 0.0, [0.0]
    for h, source_error, a in zip(hs, es, scales):
        e = (1 + h * target_L * abs(a)) * e + h * (source_L + target_L) * source_error
        history.append(float(e))
    return np.array(history)


@dataclass
class StepResult:
    status: str
    delta: np.ndarray | None
    model_residual: float | None = None
    directional_gain: float | None = None


def guarded_step(gradient, constraint_values, constraint_jacobian,
                 curvature_bounds, eta, radius, metric=None, tolerance=1e-8):
    """Convex upper-model step. This is a QCQP/SOCP, not a linear CBF QP.

The separate finite_accept gate is mandatory when curvature bounds are estimates.
"""
    g, h = np.asarray(gradient, float), np.asarray(constraint_values, float)
    J, L = np.asarray(constraint_jacobian, float), np.asarray(curvature_bounds, float)
    n = g.size
    metric = np.eye(n) if metric is None else np.asarray(metric, float)
    if g.shape != (n,) or h.ndim != 1 or J.shape != (h.size, n) or L.shape != h.shape or metric.shape != (n, n):
        raise ValueError("incompatible dimensions")
    if not np.all(np.isfinite(np.r_[g, h, J.ravel(), L, metric.ravel(), eta, radius])):
        raise ValueError("finite inputs required")
    if eta <= 0 or radius <= 0 or np.any(L < 0) or np.any(h > 0):
        raise ValueError("positive eta/radius, nonnegative curvature and feasible initial state required")
    if not np.allclose(metric, metric.T) or np.linalg.eigvalsh(metric).min() <= 0:
        raise ValueError("metric must be symmetric positive definite")
    delta = cp.Variable(n)
    constraints = [cp.norm(delta) <= radius]
    constraints += [h[i] + J[i] @ delta + L[i] / 2 * cp.sum_squares(delta) <= 0 for i in range(h.size)]
    problem = cp.Problem(cp.Minimize(cp.quad_form(delta, metric) / 2 - eta * g @ delta), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    if delta.value is None:
        return StepResult("solver_no_candidate", None)
    d = np.asarray(delta.value)
    residuals = h + J @ d + L / 2 * np.dot(d, d)
    worst = float(max(np.max(residuals, initial=-np.inf), np.linalg.norm(d) - radius))
    gain = float(g @ d)
    if not np.all(np.isfinite(d)) or worst > tolerance:
        return StepResult("local_model_rejected", None, worst, gain)
    if np.linalg.norm(d) < 1e-9 or gain <= 1e-12:
        return StepResult("small_step_or_no_ascent", None, worst, gain)
    # A numerical candidate is not assigned a proof solely from solver.status.
    if gain + tolerance < float(d @ metric @ d) / eta:
        return StepResult("variational_inequality_rejected", None, worst, gain)
    return StepResult("candidate_requires_finite_check", d, worst, gain)


def finite_accept(y, delta, gradient, objective, constraints, armijo=0.25,
                  objective_error_bound=0.0, constraint_error_bound=0.0):
    """Accept actual finite outputs. Error margins concern function evaluation.

They do not certify semantic correctness, gradient accuracy, or model likelihood.
"""
    if not 0 < armijo < 1 or min(objective_error_bound, constraint_error_bound) < 0:
        raise ValueError("invalid acceptance settings")
    y, d, g = np.asarray(y), np.asarray(delta), np.asarray(gradient)
    before, after = float(objective(y)), float(objective(y + d))
    values = np.asarray(constraints(y + d), float)
    gain = float(g @ d)
    finite = np.all(np.isfinite(np.r_[before, after, values, gain]))
    accepted = bool(finite and gain > 0 and np.all(values + constraint_error_bound <= 0)
                    and after - before >= armijo * gain + 2 * objective_error_bound)
    return dict(accepted=accepted, measured_gain=after - before,
                required_gain=armijo * gain + 2 * objective_error_bound,
                constraint_values=values.tolist())


def entropy_noise_weights(costs, prior, temperature):
    """Exact finite-support Gibbs tilt; changes the original empirical measure."""
    costs, prior = np.asarray(costs, float), np.asarray(prior, float)
    if costs.shape != prior.shape or costs.ndim != 1 or len(costs) == 0:
        raise ValueError("matching nonempty vectors required")
    if not np.all(np.isfinite(np.r_[costs, prior, temperature])) or np.any(prior <= 0) or temperature <= 0:
        raise ValueError("finite costs, positive prior masses and temperature required")
    if not np.isclose(prior.sum(), 1):
        raise ValueError("prior must sum to one")
    logits = np.log(prior) - costs / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()
