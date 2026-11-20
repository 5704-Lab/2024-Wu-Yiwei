"""Low-dimensional reference solver for the native-observation feedback design.

This module has no SD3, image, language parsing, or semantic calibration code.
It checks how already-defined scalar observations enter a step-consistent
convex controller. Production latent-space solves should use JVP/VJP.
"""

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np


@dataclass
class StepResult:
    status: str
    increment: Optional[np.ndarray]
    velocity: Optional[np.ndarray]
    diagnostics: dict


def recovery_caps(residual, tolerance, recovery_rate, step_width):
    residual = np.asarray(residual, dtype=float)
    tolerance = np.broadcast_to(np.asarray(tolerance, dtype=float), residual.shape)
    recovery_rate = np.broadcast_to(np.asarray(recovery_rate, dtype=float), residual.shape)
    if step_width <= 0 or np.any(tolerance < 0) or np.any(recovery_rate < 0):
        raise ValueError("Positive step width and nonnegative tolerances/rates required")
    violation = np.maximum(np.abs(residual) - tolerance, 0.0)
    return tolerance + np.exp(-recovery_rate * step_width) * violation


def scalar_rate_bounds(residual, tolerance, recovery_rate, step_width):
    """Stable bounds on J v, avoiding cancellation outside a scalar tube."""
    residual = np.asarray(residual, dtype=float)
    tolerance = np.broadcast_to(np.asarray(tolerance, dtype=float), residual.shape)
    recovery_rate = np.broadcast_to(np.asarray(recovery_rate, dtype=float), residual.shape)
    caps = recovery_caps(residual, tolerance, recovery_rate, step_width)
    lower = (-caps - residual) / step_width
    upper = (caps - residual) / step_width
    kappa_over_h = -np.expm1(-recovery_rate * step_width) / step_width
    above, below = residual > tolerance, residual < -tolerance
    upper = np.where(above, -kappa_over_h * (residual - tolerance), upper)
    lower = np.where(below, kappa_over_h * (-residual - tolerance), lower)
    return lower, upper, caps


def minimum_metric_correction(jacobian, rhs, metric_diagonal):
    """Minimum M-norm u satisfying J u = rhs; reject inconsistent equations."""
    jacobian = np.asarray(jacobian, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    metric_diagonal = np.asarray(metric_diagonal, dtype=float)
    if np.any(metric_diagonal <= 0):
        raise ValueError("Metric must be positive definite")
    weighted_transpose = jacobian.T / metric_diagonal[:, None]
    gram = jacobian @ weighted_transpose
    correction = weighted_transpose @ np.linalg.pinv(gram, rcond=1e-12) @ rhs
    if np.linalg.norm(jacobian @ correction - rhs) > 1e-8 * max(1.0, np.linalg.norm(rhs)):
        raise ValueError("Inconsistent recovery equations")
    return correction


def solve_local_step(
    *,
    residual,
    jacobian,
    raw_increment,
    step_width,
    tolerance,
    recovery_rate,
    metric_diagonal=None,
    task_gradients=None,
    task_rate_floor=None,
    relation_curvature=None,
    task_curvature=None,
    correction_speed_limit=None,
    total_speed_limit=None,
):
    """Solve the scalar-tube controller without implicit slack or clipping.

    r_j and J_j are normalized, source-relative *measured* facts. They are
    not inferred or certified by this routine. Optional curvature constants
    bound second-order remainders in the fixed M norm. Task floors are rates,
    not per-step constants. Returned diagnostics concern the local surrogate;
    finite nonlinear and semantic acceptance remain the caller's responsibility.
    """
    r = np.atleast_1d(np.asarray(residual, dtype=float))
    j = np.asarray(jacobian, dtype=float)
    d0 = np.atleast_1d(np.asarray(raw_increment, dtype=float))
    n = len(d0)
    if step_width <= 0 or j.shape != (len(r), n):
        raise ValueError("Invalid step width or Jacobian shape")
    metric = np.ones(n) if metric_diagonal is None else np.asarray(metric_diagonal, dtype=float)
    if metric.shape != (n,) or np.any(metric <= 0):
        raise ValueError("A positive diagonal metric is required")
    relation_l = np.zeros(len(r)) if relation_curvature is None else np.broadcast_to(np.asarray(relation_curvature, dtype=float), r.shape)
    if np.any(relation_l < 0):
        raise ValueError("Curvature bounds must be nonnegative")
    if correction_speed_limit is not None and correction_speed_limit < 0:
        raise ValueError("Correction speed limit must be nonnegative")
    if total_speed_limit is not None and total_speed_limit < 0:
        raise ValueError("Total speed limit must be nonnegative")

    lower, upper, caps = scalar_rate_bounds(r, tolerance, recovery_rate, step_width)
    base_velocity = d0 / step_width
    velocity = cp.Variable(n)
    metric_sqrt = np.sqrt(metric)
    squared_speed = cp.sum_squares(cp.multiply(metric_sqrt, velocity))
    remainder_rate = 0.5 * step_width * cp.multiply(relation_l, squared_speed)
    constraints = [j @ velocity + remainder_rate <= upper,
                   -j @ velocity + remainder_rate <= -lower]
    task_g = None
    if task_gradients is not None:
        task_g = np.atleast_2d(np.asarray(task_gradients, dtype=float))
        if task_g.shape[1] != n or task_rate_floor is None:
            raise ValueError("Task gradients require matching rate floors")
        task_floor = np.broadcast_to(np.asarray(task_rate_floor, dtype=float), (len(task_g),))
        task_l = np.zeros(len(task_g)) if task_curvature is None else np.broadcast_to(np.asarray(task_curvature, dtype=float), (len(task_g),))
        if np.any(task_l < 0):
            raise ValueError("Task curvature bounds must be nonnegative")
        constraints.append(task_g @ velocity - 0.5 * step_width * cp.multiply(task_l, squared_speed) >= task_floor)
    if correction_speed_limit is not None:
        constraints.append(cp.norm(cp.multiply(metric_sqrt, velocity - base_velocity)) <= correction_speed_limit)
    if total_speed_limit is not None:
        constraints.append(cp.norm(cp.multiply(metric_sqrt, velocity)) <= total_speed_limit)

    objective = 0.5 * cp.sum_squares(cp.multiply(metric_sqrt, velocity - base_velocity))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    # A feasible nominal velocity has exactly zero correction cost. Avoid an
    # unnecessary conic solve and boundary-degenerate numerical corrections.
    velocity.value = base_velocity.copy()
    nominal_feasible = all(np.max(constraint.violation()) <= 0.0 for constraint in constraints)
    if nominal_feasible:
        status, objective_value, solver_name = cp.OPTIMAL, 0.0, "analytic_nominal_feasibility"
    else:
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10,
                      tol_gap_rel=1e-10, max_iter=250)
        status, objective_value, solver_name = problem.status, problem.value, "CLARABEL"
    metadata = {"solver": solver_name, "status": status,
                "nominal_feasible": nominal_feasible,
                "step_width": float(step_width), "residual_caps": caps.tolist(),
                "finite_nonlinear_acceptance_performed": False,
                "semantic_validity_certified": False}
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or velocity.value is None:
        return StepResult(status, None, None, metadata)
    v = np.asarray(velocity.value).reshape(-1)
    d = d0.copy() if nominal_feasible else step_width * v
    predicted = r + j @ d
    margin = 0.5 * relation_l * float(d @ (metric * d))
    violation = np.abs(predicted) + margin - caps
    metadata.update(objective=float(objective_value),
                    maximum_relation_constraint_violation=float(np.max(violation)),
                    correction_metric_norm=float(np.linalg.norm(metric_sqrt * (d - d0))),
                    total_metric_norm=float(np.linalg.norm(metric_sqrt * d)))
    if task_g is not None:
        gap = task_g @ v - 0.5 * task_l * step_width * float(v @ (metric * v)) - task_floor
        metadata["minimum_task_rate_margin"] = float(np.min(gap))
    return StepResult(status, d, v, metadata)
