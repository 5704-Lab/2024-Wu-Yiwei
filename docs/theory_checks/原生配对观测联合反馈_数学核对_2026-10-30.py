"""Finite-dimensional checks for the paired-observation research design.

These checks use constructed arrays, not SD3 or image-editing experiments.
Run from any directory; the JSON result is written beside this script.
"""

from pathlib import Path
import json

import numpy as np


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()


def run():
    rng = np.random.default_rng(20261030)
    checks = []

    def record(name, passed, **metrics):
        checks.append({"name": name, "passed": bool(passed), **metrics})
        if not passed:
            raise AssertionError(name)

    x, delta, noise = rng.normal(size=(3, 9))
    tau = 0.37
    qx = (1 - tau) * x + tau * noise
    qy = (1 - tau) * (x + delta) + tau * noise
    err = np.linalg.norm(qy - qx - (1 - tau) * delta)
    record("paired_input_difference", err < 1e-12, error=float(err))

    matrix = rng.normal(size=(7, 9))
    psi = lambda q: np.tanh(matrix @ q)
    identity = np.linalg.norm(psi(qx) - psi(qx.copy()))
    record("nonlinear_identity_residual_is_zero", identity == 0, error=float(identity))

    full_noise_difference = np.linalg.norm(psi(noise) - psi(noise.copy()))
    record("full_noise_probe_erases_even_different_clean_images", full_noise_difference == 0 and np.linalg.norm(delta) > 0,
           residual_at_tau_one=float(full_noise_difference), clean_image_difference=float(np.linalg.norm(delta)))

    tiny = 1e-5
    derivative = (1 - np.tanh(matrix @ qx) ** 2) * (matrix @ delta) * (1 - tau)
    fd = (psi(qx + tiny * (1 - tau) * delta) - psi(qx - tiny * (1 - tau) * delta)) / (2 * tiny)
    err = np.linalg.norm(fd - derivative) / np.linalg.norm(derivative)
    record("paired_local_derivative", err < 1e-7, relative_error=float(err))

    eps_a = rng.normal(size=(20000, 9))
    eps_b = rng.normal(size=(20000, 9))
    common = (((1 - tau) * (x + delta) + tau * eps_a) @ matrix.T
              - ((1 - tau) * x + tau * eps_a) @ matrix.T)
    independent = (((1 - tau) * (x + delta) + tau * eps_a) @ matrix.T
                   - ((1 - tau) * x + tau * eps_b) @ matrix.T)
    v_common, v_independent = np.var(common, axis=0).sum(), np.var(independent, axis=0).sum()
    record("positive_covariance_common_noise_example", v_common < 1e-20 and v_independent > 0.1,
           paired_variance=float(v_common), independent_variance=float(v_independent))

    eps_a, eps_b = rng.normal(size=(2, 60000))
    # psi(q)=sin(q), with clean inputs separated by pi/(1-tau).
    paired = np.sin(np.pi + tau * eps_a) - np.sin(tau * eps_a)
    independent = np.sin(np.pi + tau * eps_a) - np.sin(tau * eps_b)
    variance_ratio = np.var(paired) / np.var(independent)
    record("common_noise_not_universal_variance_reduction", variance_ratio > 1.8,
           paired_to_independent_variance=float(variance_ratio))

    query, keys, values = rng.normal(size=4), rng.normal(size=(12, 4)), rng.normal(size=(12, 5))
    order = rng.permutation(12)
    weights = softmax(query @ keys.T / 2)
    permuted_weights = softmax(query @ keys[order].T / 2)
    original_output, permuted_output = weights @ values, permuted_weights @ values[order]
    error = np.linalg.norm(original_output - permuted_output)
    record("pooled_attention_permutation_blindness", error < 1e-12, output_error=float(error))
    coordinates = rng.normal(size=(12, 2))
    displacement = np.linalg.norm(weights @ coordinates - permuted_weights @ coordinates)
    record("coordinates_expose_this_permutation", displacement > 1e-3,
           centroid_difference=float(displacement))

    probabilities = softmax(np.array([100.0, 0.0, 0.0]))
    image_mass = probabilities[1:].sum()
    conditional_mass = (probabilities[1:] / image_mass).sum()
    record("conditional_normalization_can_hide_no_image_support", image_mass < 1e-30 and conditional_mass > 0.99,
           image_attention_mass=float(image_mass), normalized_image_mass=float(conditional_mass))

    vector = np.array([2.0, 1.0])
    direction = lambda z: z / np.linalg.norm(z)
    stretch_error = np.linalg.norm(direction(vector) - direction(1.7 * vector))
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotation_difference = np.linalg.norm(direction(vector) - direction(rotation @ vector))
    record("direction_allows_positive_length_change_detects_rotation", stretch_error < 1e-12 and rotation_difference > 1.0,
           stretch_error=float(stretch_error), rotation_difference=float(rotation_difference))

    n, m = 9, 4
    raw = rng.normal(size=(n, n))
    metric = raw.T @ raw + np.eye(n)
    jacobian = rng.normal(size=(m, n))
    weights = np.diag(rng.uniform(0.5, 2, size=m))
    residual, d0 = rng.normal(size=m), rng.normal(size=n)
    lam = 2.3
    hessian = metric + lam * jacobian.T @ weights @ jacobian
    linear = metric @ d0 - lam * jacobian.T @ weights @ residual
    dbar = np.linalg.solve(hessian, linear)
    stationarity = np.linalg.norm(metric @ (dbar - d0) + lam * jacobian.T @ weights @ (residual + jacobian @ dbar))
    record("restorative_quadratic_unique_solution", stationarity < 1e-10 and np.linalg.eigvalsh(hessian).min() > 0,
           stationarity_error=float(stationarity))

    task_gradient = rng.normal(size=n)
    target_margin = task_gradient @ dbar + 0.4
    inv_g = np.linalg.solve(hessian, task_gradient)
    multiplier = (target_margin - task_gradient @ dbar) / (task_gradient @ inv_g)
    constrained = dbar + multiplier * inv_g
    kkt = np.linalg.norm(hessian @ constrained - linear - multiplier * task_gradient)
    boundary = abs(task_gradient @ constrained - target_margin)
    record("task_halfspace_kkt_solution", kkt < 1e-10 and boundary < 1e-12 and multiplier > 0,
           stationarity_error=float(kkt), task_boundary_error=float(boundary))

    proposed = np.array([1.0, 10.0])
    radius, margin = 2.0, 0.9
    scaled = proposed * radius / np.linalg.norm(proposed)
    feasible_alternative = np.array([margin, 0.0])
    record("scaling_a_qp_solution_can_break_task_constraint",
           scaled[0] < margin and np.linalg.norm(feasible_alternative) <= radius,
           task_after_scaling=float(scaled[0]), required_task=float(margin))

    initial = np.array([0.8, -0.4])
    restorative_action = -2 / 3 * initial
    remaining = np.linalg.norm(initial + restorative_action) / np.linalg.norm(initial)
    record("current_residual_needed_to_repair_existing_error", abs(remaining - 1 / 3) < 1e-12,
           restorative_residual_ratio=float(remaining), derivative_only_residual_ratio=1.0)

    # Compare the Schur expression with an explicit nullspace projector in whitened coordinates.
    task_jacobian = rng.normal(size=(3, n))
    lower = np.linalg.cholesky(metric)
    inverse_transpose = np.linalg.solve(lower.T, np.eye(n))
    task_white = task_jacobian @ inverse_transpose
    keep_white = jacobian @ inverse_transpose
    _, _, vh = np.linalg.svd(task_white, full_matrices=True)
    basis = vh[3:].T
    explicit = keep_white @ basis @ basis.T @ keep_white.T
    inverse = np.linalg.inv(metric)
    conditional = (jacobian @ inverse @ jacobian.T
                   - jacobian @ inverse @ task_jacobian.T
                   @ np.linalg.pinv(task_jacobian @ inverse @ task_jacobian.T)
                   @ task_jacobian @ inverse @ jacobian.T)
    error = np.linalg.norm(conditional - explicit)
    record("conditional_controllability_gramian", error < 1e-10 and np.linalg.eigvalsh(conditional).min() > -1e-10,
           projector_error=float(error), minimum_eigenvalue=float(np.linalg.eigvalsh(conditional).min()))

    legal = np.array([[1.0], [0.0]])
    projector = np.eye(2) - legal @ legal.T
    independent_error = np.array([[0.0], [1.0]])
    confounded_error = legal.copy()
    record("legal_projection_requires_conditional_identifiability",
           np.linalg.norm(projector @ independent_error) == 1.0 and np.linalg.norm(projector @ confounded_error) == 0.0)

    complex_matrix = rng.normal(size=(5, 9)) + 1j * rng.normal(size=(5, 9))
    state, step = rng.normal(size=(2, 9))
    response = complex_matrix @ state
    response_step = complex_matrix @ step
    phase_rate = np.imag(np.conj(response) * response_step) / np.abs(response) ** 2
    h = 1e-6
    phase_difference = np.angle((complex_matrix @ (state + h * step)) / (complex_matrix @ (state - h * step))) / (2 * h)
    error = np.linalg.norm(phase_rate - phase_difference)
    record("nonzero_complex_response_phase_differential", error < 1e-7, finite_difference_error=float(error))

    return {"date": "2026-10-30", "scope": "constructed finite-dimensional algebra only; no image/model validation",
            "seed": 20261030, "passed": len(checks), "total": len(checks), "checks": checks}


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "total": result["total"], "output": str(output)}, ensure_ascii=False))
