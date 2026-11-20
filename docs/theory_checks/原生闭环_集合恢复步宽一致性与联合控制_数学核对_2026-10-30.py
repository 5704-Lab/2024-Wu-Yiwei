"""Constructed checks; no diffusion models, images, or semantic truth supplied."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/native_feedback_theory_20261030"))
from controller_reference import (  # noqa: E402
    minimum_metric_correction,
    recovery_caps,
    scalar_rate_bounds,
    solve_local_step,
)


def run():
    rng = np.random.default_rng(20261030)
    rows = []

    def check(name, condition, **metrics):
        rows.append({"name": name, "passed": bool(condition), **metrics})
        if not condition:
            raise AssertionError((name, metrics))

    def solve(r, j, d0, h=0.1, beta=0.1, eta=2.0, **kwargs):
        result = solve_local_step(residual=np.atleast_1d(r), jacobian=np.atleast_2d(j),
                                  raw_increment=np.atleast_1d(d0), step_width=h,
                                  tolerance=beta, recovery_rate=eta, **kwargs)
        return result

    result = solve(0.02, [1.0], 0.01)
    delta = float(abs(result.increment[0] - 0.01))
    check("valid_state_and_valid_raw_action_receive_no_extra_pull", delta < 1e-7,
          correction=delta)
    check("feasible_nominal_action_bypasses_solver_exactly",
          result.diagnostics["solver"] == "analytic_nominal_feasibility"
          and result.diagnostics["objective"] == 0.0
          and np.array_equal(result.increment, np.array([0.01])))

    result = solve(0.08, [1.0], 0.05)
    check("inside_tube_boundary_filter_allows_nonzero_motion",
          abs(result.increment[0] - 0.02) < 1e-7 and result.increment[0] > 0,
          accepted_increment=float(result.increment[0]))

    result = solve(0.5, [1.0], 0.03)
    expected = float(recovery_caps(np.array([0.5]), 0.1, 2.0, 0.1)[0])
    actual = 0.5 + result.increment[0]
    check("positive_violation_contracts_to_nonzero_tolerance", abs(actual - expected) < 1e-7,
          next_residual=float(actual), target=float(expected))

    result = solve(-0.5, [1.0], -0.03)
    check("negative_violation_contracts_symmetrically",
          abs(-0.5 + result.increment[0] + expected) < 1e-7)

    result = solve(0.0, [0.0, 1.0], [0.1, 0.0],
                   task_gradients=[[1.0, 0.0]], task_rate_floor=[0.9])
    check("legal_morphology_direction_is_not_frozen",
          np.linalg.norm(result.increment - [0.1, 0.0]) < 1e-7)

    result = solve(0.4, [0.0, 1.0], [0.1, 0.05],
                   task_gradients=[[1.0, 0.0]], task_rate_floor=[0.9])
    check("repair_preserves_separate_edit_progress",
          abs(result.increment[0] - 0.1) < 1e-7 and result.increment[1] < -0.04,
          task_increment=float(result.increment[0]), repair_increment=float(result.increment[1]))

    j = rng.normal(size=(3, 8))
    metric = rng.uniform(0.5, 2.5, size=8)
    rhs = rng.normal(size=3)
    u = minimum_metric_correction(j, rhs, metric)
    check("minimum_constraint_equations", np.linalg.norm(j @ u - rhs) < 1e-10)
    _, _, vh = np.linalg.svd(j, full_matrices=True)
    w = vh[3:].T @ rng.normal(size=5)
    work = float(w @ (metric * u))
    check("virtual_work_zero_on_compatible_directions", abs(work) < 1e-10,
          metric_virtual_work=work)
    cost_difference = float((u + w) @ (metric * (u + w)) - u @ (metric * u))
    null_cost = float(w @ (metric * w))
    check("metric_pythagorean_minimum_control_energy", abs(cost_difference - null_cost) < 1e-10)

    incompatible_detected = False
    try:
        minimum_metric_correction(np.array([[1.0], [1.0]]), np.array([1.0, -1.0]), np.ones(1))
    except ValueError:
        incompatible_detected = True
    check("inconsistent_equalities_not_silently_projected", incompatible_detected)

    step_sets = [[0.2] * 5, [0.1] * 10, [0.03, 0.17, 0.11, 0.19, 0.2, 0.3]]
    endpoints = []
    for steps in step_sets:
        residual = 0.5
        for h in steps:
            local = solve(residual, [1.0], 0.0, h=h)
            residual += float(local.increment[0])
        endpoints.append(residual)
    target = 0.1 + 0.4 * np.exp(-2.0)
    check("recovery_consistent_under_time_partition", max(abs(x - target) for x in endpoints) < 3e-7,
          endpoints=endpoints, expected=float(target))

    h_small = 1e-10
    _, upper, _ = scalar_rate_bounds(np.array([0.5]), 0.1, 2.0, h_small)
    check("small_step_rate_bound_uses_stable_exponential", abs(upper[0] + 0.8) < 1e-8,
          limiting_rate=float(upper[0]))

    old_action = -0.5 * 0.5
    old_velocities = [abs(old_action / h) for h in [0.1, 0.01, 0.001]]
    new_velocities = [float(-np.expm1(-2.0 * h) * 0.5 / h) for h in [0.1, 0.01, 0.001]]
    check("fixed_per_step_source_penalty_has_no_fixed_rate_limit",
          old_velocities[-1] > 90 * old_velocities[0] and max(new_velocities) <= 1.0,
          old_speeds=old_velocities, corrected_speeds=new_velocities)

    conflict = solve(0.5, [1.0], 0.03, task_gradients=[[1.0]], task_rate_floor=[1.0])
    check("task_recovery_conflict_reported_infeasible", conflict.increment is None and "infeasible" in conflict.status,
          status=conflict.status)

    limited = solve(0.5, [1.0], 0.03, correction_speed_limit=0.01)
    check("insufficient_action_budget_reported_infeasible", limited.increment is None and "infeasible" in limited.status)

    total = solve(0.0, [0.0, 1.0], [0.1, 1.0], beta=0.2,
                  task_gradients=[[1.0, 0.0]], task_rate_floor=[0.9], total_speed_limit=2.0)
    naive = np.array([0.1, 1.0]) * 0.2 / np.linalg.norm([0.1, 1.0])
    check("joint_trust_constraint_avoids_post_projection_clipping",
          total.increment[0] >= 0.09 - 1e-7 and np.linalg.norm(total.increment) <= 0.2 + 1e-7 and naive[0] < 0.09,
          accepted=total.increment.tolist(), naively_clipped=naive.tolist())

    no_margin = solve(0.0, [1.0], 0.1)
    with_margin = solve(0.0, [1.0], 0.1, relation_curvature=[2.0])
    true_readout = lambda d: float(d[0] + d[0] ** 2)
    check("relation_curvature_margin_controls_constructed_nonlinearity",
          true_readout(no_margin.increment) > 0.109 and true_readout(with_margin.increment) <= 0.1 + 1e-7,
          without_margin=true_readout(no_margin.increment), with_margin=true_readout(with_margin.increment))

    task_margin = solve(0.0, [0.0], 0.09, beta=1.0,
                        task_gradients=[[1.0]], task_rate_floor=[0.9], task_curvature=[2.0], total_speed_limit=2.0)
    true_task_gain = float(task_margin.increment[0] - task_margin.increment[0] ** 2)
    check("task_curvature_margin_protects_constructed_progress", true_task_gain >= 0.09 - 1e-7,
          true_task_gain=true_task_gain)

    vector = np.array([2.0, 1.0])
    direction = lambda z: z / np.linalg.norm(z)
    rotated = np.array([-vector[1], vector[0]])
    check("kinematic_director_separates_axis_stretch_and_rotation",
          np.linalg.norm(direction(vector) - direction(1.7 * vector)) < 1e-12
          and np.linalg.norm(direction(vector) - direction(rotated)) > 1.0)

    points = np.array([[-1.0, -0.2], [1.0, -0.2], [0.0, 0.4]])
    angle = 0.6
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    other = points @ rotation.T
    aligned = other @ rotation
    check("shape_alignment_can_remove_protected_rotation",
          np.linalg.norm(points - aligned) < 1e-12 and np.linalg.norm(points - other) > 0.1)

    legal = np.array([[1.0], [0.0]])
    protect = np.array([[0.0], [1.0]])
    projector = np.eye(2) - legal @ np.linalg.pinv(legal)
    check("conditional_identifiability_distinguishes_confounded_factor",
          np.linalg.norm(projector @ protect) > 0.9 and np.linalg.norm(projector @ legal) < 1e-12)

    fixed_state = 0.6
    apparent_drift = (0.9 * fixed_state - 1.0 * fixed_state) / 0.1
    check("observer_change_causes_residual_change_without_state_motion", abs(apparent_drift + 0.6) < 1e-12)

    observed_residual, semantic_bias = 0.0, 0.2
    check("zero_proxy_error_does_not_remove_semantic_bias", abs(observed_residual + semantic_bias) > 0.19)

    eta, coefficient, initial = 2.0, 0.4, 0.7
    observed_floors = []
    for count in [5, 10, 20, 40]:
        h, current = 1.0 / count, initial
        for _ in range(count):
            current = np.exp(-eta * h) * current + coefficient * h * h
        bound = np.exp(-eta) * initial + coefficient * h * np.exp(eta * h) * (1 - np.exp(-eta)) / eta
        check(f"finite_step_remainder_bound_n{count}", current <= bound + 1e-12,
              actual=float(current), bound=float(bound))
        observed_floors.append(current - np.exp(-eta) * initial)
    check("finite_step_error_floor_decreases_with_step_width", all(a > b for a, b in zip(observed_floors, observed_floors[1:])),
          floors=[float(v) for v in observed_floors])

    invalid_width_rejected = False
    try:
        solve(0.0, [1.0], 0.0, h=0.0)
    except ValueError:
        invalid_width_rejected = True
    check("zero_width_is_not_a_recovery_step", invalid_width_rejected)

    return {"date": "2026-10-30", "scope": "constructed algebra and low-dimensional convex programs; no SD3 or image validation",
            "passed": len(rows), "total": len(rows), "checks": rows}


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "total": result["total"], "output": str(output)}, ensure_ascii=False))
