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


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _broadcast_vector(value, length, name, nonnegative=False):
    array = np.asarray(value, dtype=float)
    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or vector")
    try:
        array = np.broadcast_to(array, (length,))
    except ValueError as error:
        raise ValueError(f"{name} does not match the required dimension") from error
    if not np.all(np.isfinite(array)) or (nonnegative and np.any(array < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return array


def _scalar(value, name, positive=False):
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array) or (array <= 0 if positive else array < 0):
        raise ValueError(f"{name} must be a finite " + ("positive" if positive else "nonnegative") + " scalar")
    return float(array)


def recovery_caps(residual, tolerance, recovery_rate, step_width):
    residual = _vector(residual, "residual")
    step_width = _scalar(step_width, "step_width", positive=True)
    tolerance = _broadcast_vector(tolerance, len(residual), "tolerance", nonnegative=True)
    recovery_rate = _broadcast_vector(recovery_rate, len(residual), "recovery_rate", nonnegative=True)
    violation = np.maximum(np.abs(residual) - tolerance, 0.0)
    return tolerance + np.exp(-recovery_rate * step_width) * violation


def scalar_rate_bounds(residual, tolerance, recovery_rate, step_width):
    """Stable bounds on J v, avoiding cancellation outside a scalar tube."""
    residual = _vector(residual, "residual")
    step_width = _scalar(step_width, "step_width", positive=True)
    tolerance = _broadcast_vector(tolerance, len(residual), "tolerance", nonnegative=True)
    recovery_rate = _broadcast_vector(recovery_rate, len(residual), "recovery_rate", nonnegative=True)
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
    rhs = _vector(rhs, "rhs")
    metric_diagonal = _vector(metric_diagonal, "metric_diagonal")
    if jacobian.shape != (len(rhs), len(metric_diagonal)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("Invalid finite Jacobian shape")
    if np.any(metric_diagonal <= 0):
        raise ValueError("Metric must be positive definite")
    # Solve the whitened least-norm system directly. Forming J M^-1 J^T
    # squares its condition number and can discard feasible weak directions.
    metric_sqrt = np.sqrt(metric_diagonal)
    whitened = jacobian / metric_sqrt[None, :]
    whitened_correction = np.linalg.lstsq(whitened, rhs, rcond=1e-12)[0]
    correction = whitened_correction / metric_sqrt
    if np.linalg.norm(jacobian @ correction - rhs) > 1e-8 * max(1.0, np.linalg.norm(rhs)):
        raise ValueError("Recovery equations are inconsistent at the selected numerical rank/tolerance")
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
    feasibility_atol=1e-8,
    feasibility_rtol=1e-7,
):
    """Solve the scalar-tube controller without implicit slack or clipping.

    r_j and J_j are normalized, source-relative *measured* facts. They are
    not inferred or certified by this routine. Optional curvature constants
    bound second-order remainders in the fixed M norm. Task floors are rates,
    not per-step constants. Returned diagnostics concern the local surrogate;
    finite nonlinear and semantic acceptance remain the caller's responsibility.
    """
    r = _vector(residual, "residual")
    j = np.asarray(jacobian, dtype=float)
    d0 = _vector(raw_increment, "raw_increment")
    step_width = _scalar(step_width, "step_width", positive=True)
    feasibility_atol = _scalar(feasibility_atol, "feasibility_atol")
    feasibility_rtol = _scalar(feasibility_rtol, "feasibility_rtol")
    n = len(d0)
    if j.shape != (len(r), n) or not np.all(np.isfinite(j)):
        raise ValueError("Invalid finite Jacobian shape")
    metric = np.ones(n) if metric_diagonal is None else _vector(metric_diagonal, "metric_diagonal")
    if metric.shape != (n,) or np.any(metric <= 0):
        raise ValueError("A positive diagonal metric is required")
    relation_l = np.zeros(len(r)) if relation_curvature is None else _broadcast_vector(relation_curvature, len(r), "relation_curvature", nonnegative=True)
    if correction_speed_limit is not None:
        correction_speed_limit = _scalar(correction_speed_limit, "correction_speed_limit")
    if total_speed_limit is not None:
        total_speed_limit = _scalar(total_speed_limit, "total_speed_limit")
    if task_gradients is None and (task_rate_floor is not None or task_curvature is not None):
        raise ValueError("Task rate floors/curvatures require task_gradients")

    lower, upper, caps = scalar_rate_bounds(r, tolerance, recovery_rate, step_width)
    base_velocity = d0 / step_width
    if not all(np.all(np.isfinite(value)) for value in (lower, upper, caps, base_velocity)):
        raise ValueError("Step width produces nonfinite rate bounds or base velocity")
    velocity = cp.Variable(n)
    metric_sqrt = np.sqrt(metric)
    squared_speed = cp.sum_squares(cp.multiply(metric_sqrt, velocity))
    remainder_rate = 0.5 * step_width * cp.multiply(relation_l, squared_speed)
    constraints = [j @ velocity + remainder_rate <= upper,
                   -j @ velocity + remainder_rate <= -lower]
    task_g = None
    if task_gradients is not None:
        task_g = np.atleast_2d(np.asarray(task_gradients, dtype=float))
        if task_g.ndim != 2 or task_g.shape[0] == 0 or task_g.shape[1] != n or not np.all(np.isfinite(task_g)) or task_rate_floor is None:
            raise ValueError("Task gradients require matching rate floors")
        task_floor = _broadcast_vector(task_rate_floor, len(task_g), "task_rate_floor")
        task_l = np.zeros(len(task_g)) if task_curvature is None else _broadcast_vector(task_curvature, len(task_g), "task_curvature", nonnegative=True)
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
                "solver_status": status,
                "nominal_feasible": nominal_feasible,
                "task_constraints_active": task_g is not None,
                "feasibility_atol": feasibility_atol, "feasibility_rtol": feasibility_rtol,
                "step_width": float(step_width), "residual_caps": caps.tolist(),
                "finite_nonlinear_acceptance_performed": False,
                "semantic_validity_certified": False}
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or velocity.value is None:
        metadata["local_constraints_satisfied"] = False
        return StepResult(status, None, None, metadata)
    v = np.asarray(velocity.value).reshape(-1)
    d = d0.copy() if nominal_feasible else step_width * v
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(d)):
        metadata.update(status="numerical_nonfinite_candidate", local_constraints_satisfied=False)
        return StepResult("numerical_nonfinite_candidate", None, None, metadata)
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

    # Audit every numerical constraint independently of the optimizer status.
    # This certifies only the submitted local model, within stated tolerances.
    audits = []

    def audit(name, lhs, rhs):
        lhs, rhs = np.broadcast_arrays(np.asarray(lhs, dtype=float), np.asarray(rhs, dtype=float))
        finite = bool(np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs)))
        error = lhs - rhs
        allowed = feasibility_atol + feasibility_rtol * np.maximum(1.0, np.maximum(np.abs(lhs), np.abs(rhs)))
        passed = finite and bool(np.all(error <= allowed))
        audits.append({"name": name, "passed": passed,
                       "maximum_violation": float(np.max(error)) if finite else None,
                       "maximum_tolerance_excess": float(np.max(error - allowed)) if finite else None})

    speed_squared = float(v @ (metric * v))
    relation_rate_remainder = 0.5 * step_width * relation_l * speed_squared
    audit("relation_rate_upper", j @ v + relation_rate_remainder, upper)
    audit("relation_rate_lower", -j @ v + relation_rate_remainder, -lower)
    if task_g is not None:
        audit("task_rate", task_floor, task_g @ v - 0.5 * step_width * task_l * speed_squared)
    if correction_speed_limit is not None:
        audit("correction_speed", np.linalg.norm(metric_sqrt * (v - base_velocity)), correction_speed_limit)
    if total_speed_limit is not None:
        audit("total_speed", np.linalg.norm(metric_sqrt * v), total_speed_limit)
    passed = all(row["passed"] for row in audits)
    metadata.update(constraint_audit=audits, local_constraints_satisfied=passed)
    if not passed:
        metadata["status"] = "numerical_constraint_violation"
        return StepResult("numerical_constraint_violation", None, None, metadata)
    return StepResult(status, d, v, metadata)
