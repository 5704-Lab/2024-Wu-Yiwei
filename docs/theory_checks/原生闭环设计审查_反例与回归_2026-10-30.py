"""Design audit: numerical regressions and explicit modeling counterexamples.

All cases are constructed arrays. No SD3 weights, images, or real semantic
validation are used. Modeling counterexamples remain design limitations.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from unittest.mock import patch

import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "experiments/native_feedback_theory_20261030"
sys.path.insert(0, str(DIRECTORY))
import controller_reference as controller


def run():
    checks = []

    def record(name, condition, **metrics):
        checks.append({"name": name, "passed": bool(condition), **metrics})
        if not condition:
            raise AssertionError((name, metrics))

    def rejected(name, function):
        try:
            function()
        except ValueError as error:
            record(name, True, reason=str(error))
        else:
            record(name, False)

    def solve(**changes):
        inputs = dict(residual=[0.0], jacobian=[[1.0]], raw_increment=[0.0],
                      step_width=0.1, tolerance=1.0, recovery_rate=1.0)
        inputs.update(changes)
        return controller.solve_local_step(**inputs)

    matrix = np.diag([1.0, 1e-7])
    rhs = np.array([0.0, 1e-7])
    result = controller.minimum_metric_correction(matrix, rhs, np.ones(2))
    record("feasible_weak_direction_is_retained", np.linalg.norm(result - [0, 1]) < 1e-10,
           correction=result.tolist(), equality_error=float(np.linalg.norm(matrix @ result - rhs)))

    rejected("orphan_task_floor_is_not_silently_ignored", lambda: solve(task_rate_floor=[1.0]))
    rejected("orphan_task_curvature_is_not_silently_ignored", lambda: solve(task_curvature=[1.0]))
    rejected("nonfinite_step_width_is_rejected", lambda: solve(step_width=float("nan")))
    rejected("nonfinite_jacobian_is_rejected", lambda: solve(jacobian=[[float("nan")]]))
    rejected("matrix_shaped_residual_is_rejected", lambda: solve(residual=[[0.0, 0.1]]))
    rejected("empty_observer_requires_explicit_caller_handling", lambda: solve(residual=[], jacobian=np.zeros((0, 1))))
    rejected("nonfinite_metric_is_rejected", lambda: solve(metric_diagonal=[float("nan")]))

    def fake_solver(value, status):
        def run_fake(problem, **kwargs):
            problem.variables()[0].value = np.array([value])
            problem._status = status
            problem._value = 0.0
        return run_fake

    # Fault injection tests a missing numerical audit path; these are not
    # claims that CLARABEL produced these incorrect candidates in normal use.
    with patch.object(cp.Problem, "solve", fake_solver(1.0, cp.OPTIMAL_INACCURATE)):
        result = solve(residual=[0.5], raw_increment=[0.03], tolerance=0.1, recovery_rate=2.0)
    record("inaccurate_status_does_not_bypass_relation_audit",
           result.increment is None and result.status == "numerical_constraint_violation"
           and result.diagnostics["solver_status"] == cp.OPTIMAL_INACCURATE)

    with patch.object(cp.Problem, "solve", fake_solver(0.0, cp.OPTIMAL)):
        result = solve(jacobian=[[0.0]], task_gradients=[[1.0]], task_rate_floor=[1.0])
    record("optimal_status_does_not_bypass_task_audit",
           result.increment is None and any(row["name"] == "task_rate" and not row["passed"]
                                           for row in result.diagnostics["constraint_audit"]))

    with patch.object(cp.Problem, "solve", fake_solver(10.0, cp.OPTIMAL)):
        result = solve(jacobian=[[0.0]], raw_increment=[1.0], total_speed_limit=1.0)
    record("total_speed_is_audited_after_solver",
           result.increment is None and any(row["name"] == "total_speed" and not row["passed"]
                                           for row in result.diagnostics["constraint_audit"]))

    with patch.object(cp.Problem, "solve", fake_solver(12.0, cp.OPTIMAL)):
        result = solve(jacobian=[[0.0]], raw_increment=[1.0], correction_speed_limit=1.0,
                       task_gradients=[[1.0]], task_rate_floor=[11.0])
    record("correction_speed_is_audited_after_solver",
           result.increment is None and any(row["name"] == "correction_speed" and not row["passed"]
                                           for row in result.diagnostics["constraint_audit"]))

    result = solve(residual=[0.5], raw_increment=[0.03], tolerance=0.1, recovery_rate=2.0,
                   total_speed_limit=2.0, correction_speed_limit=2.0)
    record("actual_conic_solution_passes_all_local_audits",
           result.increment is not None and result.diagnostics["local_constraints_satisfied"]
           and all(row["passed"] for row in result.diagnostics["constraint_audit"]))

    angle = 0.15
    residual = np.array([np.cos(angle) - 1, np.sin(angle)])
    rotation = np.array([[2 ** -0.5, -2 ** -0.5], [2 ** -0.5, 2 ** -0.5]])
    rotated = rotation @ residual
    record("independent_direction_boxes_are_coordinate_dependent",
           not np.all(np.abs(residual) <= 0.12) and np.all(np.abs(rotated) <= 0.12),
           original=residual.tolist(), rotated=rotated.tolist())
    record("group_l2_direction_error_preserves_coordinate_rotation",
           abs(np.linalg.norm(residual) - np.linalg.norm(rotated)) < 1e-12)

    steps = np.array([0.1, 0.01, 0.001])
    increments = -(-np.expm1(-steps)) * (steps ** 2 - 1) / (2 * steps)
    record("bounded_jacobian_alone_does_not_bound_inverse_feedback",
           increments[-1] > 0.49 and increments[-1] / steps[-1] > 400,
           step_widths=steps.tolist(), increments=increments.tolist())

    result = solve(jacobian=[[0.0]], raw_increment=[0.1],
                   task_gradients=[[0.0]], task_rate_floor=[0.01])
    actual_nonlinear_gain = 0.1 ** 2  # task score s(y)=y^2 at y=0
    record("linearized_infeasibility_does_not_prove_nonlinear_unreachability",
           result.increment is None and "infeasible" in result.status and actual_nonlinear_gain >= 0.1 * 0.01,
           nonlinear_gain=actual_nonlinear_gain, required_finite_gain=0.001)

    old = ROOT / "docs/theory_checks/原生闭环_集合恢复步宽一致性与联合控制_数学核对_2026-10-30.py"
    spec = importlib.util.spec_from_file_location("prior_closure_regression", old)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = module.run()
    record("prior_30_checks_still_pass", previous["passed"] == previous["total"] == 30)

    directory = DIRECTORY / "audit_20261030"
    directory.mkdir(exist_ok=True)
    (directory / "prior_30_regression.json").write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n")
    result = {"date": "2026-10-30", "scope": "constructed numerical audit and modeling counterexamples; no image/model validation",
              "controller_sha256_after": hashlib.sha256((DIRECTORY / "controller_reference.py").read_bytes()).hexdigest(),
              "audit_checks_passed": len(checks), "audit_checks_total": len(checks),
              "prior_regression_passed": previous["passed"], "prior_regression_total": previous["total"],
              "checks": checks}
    (directory / "after.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: result[k] for k in ["audit_checks_passed", "audit_checks_total", "prior_regression_passed", "prior_regression_total"]}))
