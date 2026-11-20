"""Executable checks and counterexamples for the accompanying proofs.

These use analytic mixture densities and explicit low-dimensional dynamics.
No image model, pretrained weights, semantic evaluator or GPU is used.
"""

import json
from pathlib import Path

import numpy as np
import torch

from reference import (clean_field, defect_certificate, entropy_noise_weights,
                       finite_accept, guarded_step, propagated_edit_bound,
                       radial_potential, value_gradient)

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent


class GaussianMixture:
    def __init__(self, means, covariances, weights):
        self.means = torch.tensor(means)
        self.covariances = torch.tensor(covariances)
        self.log_weights = torch.tensor(weights).log()

    def log_density(self, y, sigma):
        var = self.covariances + sigma**2 * torch.eye(y.numel())
        difference = y - self.means
        quadratic = torch.einsum("ki,kij,kj->k", difference, torch.linalg.inv(var), difference)
        return torch.logsumexp(self.log_weights - 0.5 * (quadratic + torch.linalg.slogdet(var)[1]
                               + y.numel() * np.log(2 * np.pi)), dim=0)

    def velocity(self, q, time):
        alpha = 1 - time
        var = alpha**2 * self.covariances + time**2 * torch.eye(q.numel())
        difference = q - alpha * self.means
        inv = torch.linalg.inv(var)
        quadratic = torch.einsum("ki,kij,kj->k", difference, inv, difference)
        probs = torch.softmax(self.log_weights - 0.5 * (quadratic + torch.linalg.slogdet(var)[1]), dim=0)
        gains = alpha * self.covariances @ inv
        means = self.means + torch.einsum("kij,kj->ki", gains, difference)
        posterior = (probs[:, None] * means).sum(dim=0)
        return (q - posterior) / time


def run():
    rng = np.random.default_rng(20261106)
    checks = []

    def record(name, passed, **metrics):
        checks.append(dict(name=name, passed=bool(passed), **metrics))
        if not passed:
            raise AssertionError(dict(name=name, **metrics))

    src = GaussianMixture([[-0.8, 0.2], [0.9, -0.4]],
                          [[[0.7, 0.15], [0.15, 0.5]], [[0.5, -0.1], [-0.1, 1.0]]], [0.6, 0.4])
    tar = GaussianMixture([[0.2, 0.3], [1.3, -0.1]],
                          [[[0.9, -0.2], [-0.2, 0.6]], [[0.6, 0.2], [0.2, 0.8]]], [0.35, 0.65])
    max_error = 0.0
    for sigma in [0.03, 0.1, 0.4, 1.0]:
        for state in rng.normal(size=(8, 2)):
            y = torch.tensor(state, requires_grad=True)
            log_ratio = tar.log_density(y, sigma) - src.log_density(y, sigma)
            exact, = torch.autograd.grad(log_ratio, y)
            actual = clean_field(y, sigma, src.velocity, tar.velocity)
            max_error = max(max_error, float((exact - actual).norm().detach()))
    record("T1_deterministic_heat_score_identity_multimodal", max_error < 1e-10, max_error=max_error)

    sigma, y = 0.3, torch.tensor([0.2, -0.5], requires_grad=True)
    t, alpha = sigma / (1 + sigma), 1 / (1 + sigma)
    ds = alpha * y - t * src.velocity(alpha * y, t)
    dt = alpha * y - t * tar.velocity(alpha * y, t)
    error = float(((dt - ds) / sigma**2 - clean_field(y, sigma, src.velocity, tar.velocity)).norm().detach())
    record("T1_posterior_mean_difference_representation", error < 1e-12, error=error)

    identical = clean_field(y, sigma, src.velocity, src.velocity)
    record("T1_identical_condition_field_is_zero", float(identical.norm()) == 0)

    # Heat semigroup on a Gaussian component: covariance increments add.
    p, s1, s2 = np.array([0.4, -0.3]), 0.2, 0.5
    base = np.array([[1.0, 0.3], [0.3, 0.8]])
    sequential = base + s1**2 * np.eye(2) + s2**2 * np.eye(2)
    direct = base + (s1**2 + s2**2) * np.eye(2)
    record("T1_heat_semigroup_variance_parameter", np.allclose(sequential, direct))

    exact0 = np.linalg.solve(base, p)
    bias = [np.linalg.norm(np.linalg.solve(base + sig**2 * np.eye(2), p) - exact0) for sig in [0.02, 0.01]]
    record("T2_smoothing_bias_second_order_example", 3.9 < bias[0] / bias[1] < 4.1, bias_ratio=bias[0] / bias[1])

    perturb = torch.tensor([1e-4, -2e-4])
    perturbed_target = lambda q, time: tar.velocity(q, time) + perturb
    error = float((clean_field(y, sigma, src.velocity, perturbed_target)
                   - clean_field(y, sigma, src.velocity, tar.velocity)).norm().detach())
    expected = float(perturb.norm()) / (sigma * (1 + sigma))
    record("T2_network_error_amplification", abs(error - expected) < 1e-12,
           measured=error, predicted=expected)

    # A bias/noise trade-off minimum, not a universal noise-level recommendation.
    B, epsilon = 1.2, 0.004
    sigma_star = (epsilon / (2 * B)) ** (1 / 3)
    risk = lambda sig: B * sig**2 + epsilon / sig
    record("T2_small_scale_bias_error_balance", risk(sigma_star) < min(risk(sigma_star / 2), risk(sigma_star * 2)),
           constructed_optimal_sigma=sigma_star)

    source = torch.tensor([-0.3, 0.2])
    state = np.array([0.4, -0.6])
    field = lambda z: clean_field(z, sigma, src.velocity, tar.velocity)
    exact_difference = float((tar.log_density(torch.tensor(state), sigma) - src.log_density(torch.tensor(state), sigma)
                             - tar.log_density(source, sigma) + src.log_density(source, sigma)).detach())
    errors = []
    for nodes in [16, 32, 64]:
        approx = float(radial_potential(torch.tensor(state), source, field, nodes).detach())
        errors.append(abs(approx - exact_difference))
    record("T3_radial_potential_recovers_ideal_log_ratio", errors[-1] < 1e-4 and errors[0] > errors[-1], errors=errors)
    record("T3_midpoint_quadrature_second_order", 3.8 < errors[0] / errors[1] < 4.2, ratio=errors[0] / errors[1])

    A = torch.tensor([[0.3, 1.2], [-0.8, -0.5]])
    c = torch.tensor([0.4, -0.2])
    linear_field = lambda z: A @ z + c
    scalar = lambda z: radial_potential(z, source, linear_field, 4)
    _, gradient = value_gradient(state, scalar)
    d = torch.tensor(state) - source
    expected = (A @ source + c + 0.5 * (A + A.T) @ d).numpy()
    record("T3_nonconservative_field_actual_potential_gradient", np.linalg.norm(gradient - expected) < 1e-12)
    defect = gradient - linear_field(torch.tensor(state)).numpy()
    predicted = (0.5 * (A.T - A) @ d).numpy()
    bound = 0.5 * np.linalg.norm((A.T - A).numpy(), 2) * float(d.norm())
    record("T3_curl_correction_identity_and_bound", np.linalg.norm(defect - predicted) < 1e-12
           and np.linalg.norm(defect) <= bound + 1e-12, correction_norm=float(np.linalg.norm(defect)), bound=bound)

    skew = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotational = lambda z: skew @ z
    origin = torch.zeros(2)
    _, grad_rot = value_gradient(state, lambda z: radial_potential(z, origin, rotational, 4))
    record("T3_rotation_has_no_radial_potential_progress", np.linalg.norm(grad_rot) < 1e-12
           and float(rotational(torch.tensor(state)).norm()) > 0.1)

    def nonlinear_field(z):
        return torch.stack([torch.sin(z[0]) + 0.3 * z[1]**2, torch.cos(z[1]) + 0.5 * z[0]])

    scalar = lambda z: radial_potential(z, source, nonlinear_field, 8)
    _, actual_grad = value_gradient(state, scalar)
    fd = np.array([(float(scalar(torch.tensor(state + 1e-5 * e)))
                    - float(scalar(torch.tensor(state - 1e-5 * e)))) / 2e-5 for e in np.eye(2)])
    record("T3_actual_quadrature_gradient_finite_difference", np.linalg.norm(actual_grad - fd) < 1e-8,
           error=float(np.linalg.norm(actual_grad - fd)))

    wrong = lambda z: sum(torch.sum(nonlinear_field(source + ((j + 0.5) / 8) * (z - source)).detach()
                                   * (z - source)) for j in range(8)) / 8
    _, detached_grad = value_gradient(state, wrong)
    record("T3_detach_breaks_claimed_gradient", np.linalg.norm(detached_grad - fd) > 0.05,
           wrong_gradient_error=float(np.linalg.norm(detached_grad - fd)))

    # Uniform field perturbation changes radial potential by <= epsilon*radius.
    delta_field = torch.tensor([0.02, -0.01])
    actual_pot = float(radial_potential(torch.tensor(state), source, lambda z: nonlinear_field(z) + delta_field, 8))
    base_pot = float(radial_potential(torch.tensor(state), source, nonlinear_field, 8))
    error_bound = float(delta_field.norm() * d.norm())
    record("T4_model_error_to_potential_value_bound", abs(actual_pot - base_pot) <= error_bound + 1e-12,
           error=abs(actual_pot - base_pot), bound=error_bound)

    # A changing objective need not share a Lyapunov function.
    before, after = np.array([0.0, 0.0]), np.array([0.0, 1.0])
    fixed_score = lambda z: z[0] - z[1]
    new_score = lambda z: z[1]
    record("T4_adaptive_scale_objective_needs_fixed_acceptance_reference",
           new_score(after) > new_score(before) and fixed_score(after) < fixed_score(before))

    # T5 residual certificate: q'=ell*q; quadratic source path approximation.
    ell, x, end = 0.7, 1.3, 0.6
    times = np.linspace(0, end, 13)
    qhat = lambda z: x * (1 + ell * z + ell**2 * z**2 / 2)
    residual = lambda z: -x * ell**3 * z**2 / 2
    cert = defect_certificate(times, np.abs(residual(times)), ell, abs(x * ell**3 * end))
    dense_times = np.linspace(0, end, 1001)
    delta = cert["conditional_residual_bound"]
    radius = delta * np.expm1(ell * dense_times) / ell
    actual_error = abs(qhat(dense_times) - x * np.exp(ell * dense_times))
    record("T5_source_defect_certificate", np.all(actual_error <= radius + 1e-12),
           terminal_error=float(actual_error[-1]), terminal_bound=float(radius[-1]))

    action = x**2 * ell**6 * end**5 / 40
    action_bound = np.sqrt(np.expm1(2 * ell * end) / (2 * ell)) * np.sqrt(2 * action)
    record("T5_action_to_path_error_bound", actual_error[-1] <= action_bound,
           terminal_error=float(actual_error[-1]), action_bound=float(action_bound))

    spike_times = np.linspace(0, 1, 11)
    sample_residuals = abs(np.sin(20 * np.pi * spike_times))
    spike_cert = defect_certificate(spike_times, sample_residuals, 0, 20 * np.pi)
    record("T5_sampled_zero_defects_need_between_sample_bound", sample_residuals.max() < 1e-12
           and spike_cert["conditional_residual_bound"] >= 1,
           sampled_max=float(sample_residuals.max()), true_max=1.0,
           certified_envelope=spike_cert["conditional_residual_bound"])

    # T6 same Euler scheme, perturbed source queries.
    hs = np.full(10, 0.04)
    queries = np.linspace(1, 0.7, 10)
    errors = np.linspace(0.01, 0.025, 10)
    ls, lt = 0.6, 0.8
    ref, modified = 1.0, 1.0
    observed = [0.0]
    for h, q, e in zip(hs, queries, errors):
        ref += h * (ls * q - (lt * (q + ref - 1) + 0.2))
        modified += h * (ls * (q + e) - (lt * (q + e + modified - 1) + 0.2))
        observed.append(abs(ref - modified))
    propagated = propagated_edit_bound(hs, errors, ls, lt)
    record("T6_source_error_propagation_to_edit_path", np.all(np.array(observed) <= propagated + 1e-12),
           terminal_error=observed[-1], terminal_bound=float(propagated[-1]))

    # T7 Gibbs variational identity, verified for arbitrary comparison measures.
    costs, prior, temperature = np.array([0.3, 1.2, -0.4]), np.array([0.2, 0.5, 0.3]), 0.7
    wstar = entropy_noise_weights(costs, prior, temperature)
    kl = lambda p, q: float(np.sum(p * np.log(p / q)))
    objective = lambda p: float(p @ costs + temperature * kl(p, prior))
    max_identity_error = 0.0
    for _ in range(30):
        w = rng.dirichlet(np.ones(3))
        max_identity_error = max(max_identity_error, abs(objective(w) - objective(wstar) - temperature * kl(w, wstar)))
    record("T7_gibbs_optimality_and_uniqueness_identity", max_identity_error < 1e-12,
           max_error=max_identity_error, selected_weights=wstar.tolist())
    record("T7_tilting_is_not_unbiased_original_sampling", np.linalg.norm(wstar - prior) > 0.1)

    # A convex combination cannot edit the task if every probe is nuisance-only.
    bank = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]])
    task_gradient = np.array([1.0, 0.0])
    record("T7_noise_selector_convex_hull_obstruction", np.max(task_gradient @ bank) == 0)

    # T8 complete finite-step loop with an explicit protected nuisance coordinate.
    objective_torch = lambda z: 1 - 0.5 * (z[0] - 1)**2 - 0.3 * (z[1] - 0.8)**2
    objective_np = lambda z: float(objective_torch(torch.tensor(z)))
    protected = lambda z: np.array([z[1] - 0.05, -z[1] - 0.05, np.dot(z, z) - 4])
    protected_jacobian = lambda z: np.array([[0.0, 1.0], [0.0, -1.0], [2 * z[0], 2 * z[1]]])
    y = np.zeros(2)
    states, scores, gains, certificates = [y.copy()], [objective_np(y)], [], []
    for _ in range(30):
        _, g = value_gradient(y, objective_torch)
        step = guarded_step(g, protected(y), protected_jacobian(y), [0, 0, 2], eta=0.4, radius=0.3)
        if step.delta is None:
            break
        accepted = finite_accept(y, step.delta, g, objective_np, protected)
        if not accepted["accepted"]:
            break
        bound = np.dot(step.delta, step.delta) / (2 * 0.4)
        gains.append(accepted["measured_gain"])
        certificates.append(bound)
        y = y + step.delta
        states.append(y.copy())
        scores.append(objective_np(y))
    states, scores = np.array(states), np.array(scores)
    record("T8_finite_loop_preserves_all_registered_constraints", len(states) > 5
           and max(float(protected(z).max()) for z in states) <= 0,
           accepted_steps=len(states) - 1, max_protected_change=float(abs(states[:, 1]).max()))
    record("T8_scalar_objective_is_monotone", np.all(np.diff(scores) > 0),
           initial=float(scores[0]), final=float(scores[-1]))
    record("T8_ascent_lower_bound", np.all(np.array(gains) + 1e-8 >= np.array(certificates)),
           minimum_margin=float(np.min(np.array(gains) - np.array(certificates))))
    record("T8_explicit_proxy_goal_reached", scores[-1] >= 0.82,
           threshold=0.82, final_state=states[-1].tolist())

    # Counterexample to infinitesimal safety without a finite check.
    circle_state, tangent = np.array([1.0, 0.0]), np.array([0.0, 0.1])
    check = finite_accept(circle_state, tangent, np.array([0.0, 1.0]), lambda z: z[1],
                          lambda z: np.array([np.dot(z, z) - 1]))
    record("T8_actual_gate_rejects_tangent_euler_leakage", not check["accepted"]
           and circle_state @ tangent == 0, finite_violation=check["constraint_values"][0])

    # Conservative margins account for uncertain scalar evaluations.
    gate = finite_accept(np.zeros(1), np.array([0.01]), np.ones(1), lambda z: z[0],
                         lambda z: np.array([-1.0]), objective_error_bound=0.01)
    record("T8_evaluation_uncertainty_requires_gain_margin", not gate["accepted"])

    stalled = guarded_step([0, 1], [0, 0], [[0, 1], [0, -1]], [0, 0], 0.3, 0.2)
    record("T8_no_feasible_progress_is_not_success", stalled.delta is None,
           status=stalled.status)

    # A feasible but saturated source-distance budget cannot guarantee task completion.
    required_task, maximum_allowed = 1.0, 0.2
    record("T9_preservation_and_task_can_be_incompatible", maximum_allowed < required_task)

    # Spectral/physical caution: density smoothing is not spatial image smoothing.
    image = np.array([1.0, -1.0, 1.0, -1.0])
    queried = 0.8 * image
    record("T1_query_scaling_does_not_spatially_blur_image", np.allclose(queried / 0.8, image)
           and np.sign(queried).tolist() == np.sign(image).tolist())

    # High-dimensional mean shifts can have large KL despite small coordinates.
    dimension, amplitude = 262144, 0.01
    kl_value = 0.5 * dimension * amplitude**2
    record("T7_bounded_coordinates_do_not_imply_small_KL", kl_value > 10, kl=kl_value)

    # Explicitly test validation failures instead of silently dropping constraints.
    invalid_cases = [lambda: clean_field(torch.ones(2), 0, src.velocity, tar.velocity),
                     lambda: defect_certificate([0, 1], [0, 0], 1, -1),
                     lambda: guarded_step([1, 0], [0], [[1, 0]], [-1], 0.2, 0.3)]
    raised = 0
    for case in invalid_cases:
        try:
            case()
        except ValueError:
            raised += 1
    record("invalid_assumptions_fail_explicitly", raised == len(invalid_cases), rejected_inputs=raised)

    output = HERE / "outputs"
    output.mkdir(exist_ok=True)
    result = dict(date="2026-11-06", seed=20261106,
                  scope="Analytic mixtures, explicit ODEs, and low-dimensional constrained optimization; no FLUX/SD3 inference",
                  passed=sum(c["passed"] for c in checks), total=len(checks), checks=checks,
                  example_trajectory=states.tolist(), example_scores=scores.tolist())
    (output / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_results(output / "analytic_examples.png", dense_times, actual_error, radius, states, scores)
    print(json.dumps(dict(passed=result["passed"], total=result["total"], final_state=states[-1].tolist(),
                          result=str(output / "checks.json")), ensure_ascii=False))


def plot_results(path, times, errors, radii, states, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uncontrolled = [np.zeros(2)]
    for _ in range(len(states) - 1):
        z = uncontrolled[-1]
        uncontrolled.append(z + 0.4 * np.array([1 - z[0], 0.6 * (0.8 - z[1])]))
    uncontrolled = np.array(uncontrolled)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(times, errors, label="Actual path error")
    axes[0].plot(times, radii, label="Defect bound (known L, M)")
    axes[0].set(xlabel="Noise time", ylabel="Absolute source-path error", title="Source coupling: conditional certificate")
    axes[0].legend(fontsize=8)
    axes[1].axhspan(-0.05, 0.05, color="#72a38c", alpha=0.3, label="Registered preservation tube")
    axes[1].plot(uncontrolled[:, 0], uncontrolled[:, 1], ".-", label="Unconstrained ascent")
    axes[1].plot(states[:, 0], states[:, 1], ".-", label="Finite guarded steps")
    axes[1].set(xlabel="Task coordinate", ylabel="Protected nuisance coordinate", title="Clean-state constrained editing (toy)")
    axes[1].legend(fontsize=8)
    axes[2].plot(scores, ".-", color="#286a48")
    axes[2].axhline(0.82, color="#ae7137", ls="--", label="Specified proxy threshold")
    axes[2].set(xlabel="Accepted step", ylabel="Explicit scalar objective", title="Verified finite objective progress")
    axes[2].legend(fontsize=8)
    fig.suptitle("Analytic verification only — no pretrained image-model result", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    run()
