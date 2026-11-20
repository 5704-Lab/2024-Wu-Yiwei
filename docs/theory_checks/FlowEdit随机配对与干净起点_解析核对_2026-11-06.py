"""Analytic Gaussian counterexamples; no pretrained model or image inference.

Run: .venv/bin/python docs/theory_checks/FlowEdit随机配对与干净起点_解析核对_2026-11-06.py
The JSON and PNG beside this file are reproducible mathematical diagnostics.
"""

import json
from pathlib import Path

import numpy as np


def covariance(t, cov):
    return (1 - t) ** 2 * cov + t**2 * np.eye(len(cov))


def score(q, t, mean, cov):
    return -np.linalg.solve(covariance(t, cov), (q - (1 - t) * mean).T).T


def denoise(q, t, mean, cov):
    gain = (1 - t) * cov @ np.linalg.inv(covariance(t, cov))
    return mean + (q - (1 - t) * mean) @ gain.T


def velocity(q, t, mean, cov):
    if t == 0:
        return -np.asarray(q)
    return (q - denoise(q, t, mean, cov)) / t


def log_density(q, t, mean, cov):
    var = covariance(t, cov)
    d = q - (1 - t) * mean
    return -0.5 * (d @ np.linalg.solve(var, d) + np.linalg.slogdet(var)[1])


def run():
    rng = np.random.default_rng(20261106)
    checks = []

    def record(name, condition, **metrics):
        checks.append(dict(name=name, passed=bool(condition), **metrics))
        if not condition:
            raise AssertionError(name)

    mean = np.array([0.3, -0.2])
    cov = np.array([[1.3, 0.4], [0.4, 0.8]])
    x, eps = np.array([1.2, -0.8]), np.array([-0.7, 1.1])
    t = 0.43
    q = (1 - t) * x + t * eps
    v = velocity(q, t, mean, cov)
    s = score(q, t, mean, cov)
    error = float(np.linalg.norm(s + (q + (1 - t) * v) / t))
    record("exact_score_velocity_identity", error < 1e-12, error=error)

    r = v - (eps - x)
    error = float(np.linalg.norm(r - (x - denoise(q, t, mean, cov)) / t))
    record("chord_residual_is_posterior_reconstruction_residual", error < 1e-12,
           identity_error=error, residual_norm=float(np.linalg.norm(r)))

    # At a fixed noisy q, the regression irreducible variance is Cov(X|q)/t^2.
    posterior_cov = cov - (1 - t) ** 2 * cov @ np.linalg.solve(covariance(t, cov), cov)
    irreducible_mse = float(np.trace(posterior_cov) / t**2)
    record("perfect_velocity_still_has_nonzero_chord_residual", irreducible_mse > 0,
           conditional_expected_squared_residual=irreducible_mse)

    # Isotropic Gaussian: exact ODE map q(t)=alpha*mu+sqrt(V)/sigma*(x-mu).
    sigma, scalar_x, scalar_t = 1.0, 2.0, 0.5
    var = (1 - scalar_t) ** 2 * sigma**2 + scalar_t**2
    inv_q = np.sqrt(var) / sigma * scalar_x
    base_mean, base_cov = np.array([0.0]), np.eye(1)
    inv_d = float(denoise(np.array([inv_q]), scalar_t, base_mean, base_cov)[0])
    proxy_q = var * scalar_x / ((1 - scalar_t) * sigma**2)
    proxy_d = float(denoise(np.array([proxy_q]), scalar_t, base_mean, base_cov)[0])
    proxy_actual_return = sigma * proxy_q / np.sqrt(var)
    record("posterior_mean_inverse_is_not_ode_inverse", abs(proxy_d - scalar_x) < 1e-12
           and abs(proxy_actual_return - scalar_x) > 0.8,
           x=scalar_x, time=scalar_t, posterior_consistent_q=proxy_q,
           posterior_prediction=proxy_d, actual_ode_return=proxy_actual_return)
    record("true_ode_inverse_need_not_have_zero_denoiser_residual", abs(inv_d - scalar_x) > 0.5,
           true_inverse_q=float(inv_q), posterior_prediction=inv_d,
           actual_ode_return=float(sigma * inv_q / np.sqrt(var)))

    samples, noises = rng.normal(size=(2, 100000))
    random_queries = (1 - scalar_t) * samples + scalar_t * noises
    ode_queries = np.sqrt(var) * samples
    errors = [abs(float(np.var(a)) - var) for a in [random_queries, ode_queries]]
    record("same_marginals_do_not_imply_same_sample_coupling", max(errors) < 0.012
           and np.mean((random_queries - ode_queries) ** 2) > 0.2,
           theoretical_variance=var, empirical_variance_errors=errors,
           paired_query_mse=float(np.mean((random_queries - ode_queries) ** 2)))

    other_mean = np.array([1.0, 0.0])
    zero_mean = np.zeros(2)
    endpoint_diff = velocity(x, 0, zero_mean, cov) - velocity(x, 0, other_mean, cov)
    record("clean_endpoint_condition_difference_degenerates", np.linalg.norm(endpoint_diff) == 0,
           velocity_difference_norm=float(np.linalg.norm(endpoint_diff)))
    clean_score_diff = score(x, 0, other_mean, cov) - score(x, 0, zero_mean, cov)
    tiny = 1e-4
    tiny_q = (1 - tiny) * x
    b = velocity(tiny_q, tiny, zero_mean, cov) - velocity(tiny_q, tiny, other_mean, cov)
    scaled = (1 - tiny) ** 2 / tiny * b
    error = float(np.linalg.norm(scaled - clean_score_diff))
    record("normalized_small_noise_signal_survives", error < 1e-6,
           raw_signal_norm=float(np.linalg.norm(b)), normalized_limit_error=error)

    ts = [0.02, 0.01]
    lift_errors = [abs(np.sqrt((1 - z) ** 2 + z**2) * scalar_x - (1 - z) * scalar_x) for z in ts]
    ratio = lift_errors[0] / lift_errors[1]
    record("deterministic_first_order_lift_has_second_order_local_error", 3.9 < ratio < 4.2,
           t_values=ts, errors=lift_errors, error_ratio=ratio)
    # For a smooth Gaussian, q=alpha*x-t^2/2*score_0(x)+O(t^3).
    second_order_errors = [abs(np.sqrt((1 - z) ** 2 + z**2) * scalar_x
                               - ((1 - z) * scalar_x + z**2 * scalar_x / 2)) for z in ts]
    record("second_order_lift_expansion", 7.8 < second_order_errors[0] / second_order_errors[1] < 8.3,
           errors=second_order_errors)

    d = np.array([0.3, -0.4])
    qs, qt_fe = q, q + d
    qt_shared = (1 - t) * (x + d) + t * eps
    error = float(np.linalg.norm(qt_fe - qt_shared - t * d))
    record("equal_displacement_vs_shared_noising", error < 1e-12,
           error=error, query_difference_norm=float(np.linalg.norm(qt_fe - qt_shared)))

    same_prompt_b = velocity(qs, t, mean, cov) - velocity(qs, t, mean, cov)
    naive_b = eps - x - velocity(qs, t, mean, cov)
    record("replacing_only_source_velocity_breaks_no_edit_identity",
           np.linalg.norm(same_prompt_b) == 0 and np.linalg.norm(naive_b) > 0.1,
           original_norm=float(np.linalg.norm(same_prompt_b)), naive_norm=float(np.linalg.norm(naive_b)))

    # Coordinate transform on smooth ODE paths, not a claim about resampled queries.
    a, adot = 1 - t, -1.0
    vs, vt = np.array([0.4, -0.2]), np.array([-0.6, 0.8])
    d_dot = (vt - vs - adot * d) / a
    error = float(np.linalg.norm(vs + adot * d + a * d_dot - vt))
    record("time_dependent_query_scaling_requires_chain_rule", error < 1e-12,
           chain_rule_error=error, naive_derivative_error=float(np.linalg.norm(vs + adot * d + a * (vt - vs) - vt)))

    # Exact Gaussian Gibbs tilt: J(e)=(e-b)^2/2, temperature lambda.
    desired, lam = 2.0, 1.0
    tilted_mean, tilted_var = desired / (1 + lam), lam / (1 + lam)
    kl = 0.5 * (tilted_mean**2 + tilted_var - 1 - np.log(tilted_var))
    record("adaptive_noise_changes_measure", tilted_mean != 0 and kl > 0,
           original_mean=0.0, selected_mean=tilted_mean, selected_variance=tilted_var, kl=float(kl))

    # Even Gaussian marginals at every timestep do not remove adaptive correlation.
    e1, independent = rng.normal(size=(2, 100000))
    rho, h = 0.8, 0.1
    e2 = rho * e1 + np.sqrt(1 - rho**2) * independent
    estimated_bias = float(np.mean((h * e1) * e2))
    record("correlated_noise_changes_adaptive_dynamics", abs(estimated_bias - h * rho) < 0.003,
           independent_mean_field_prediction=0.0, measured_bias=estimated_bias, exact_bias=h * rho)

    corr_cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    ratio_gradient = score(np.zeros(2), 0, other_mean, corr_cov) - score(np.zeros(2), 0, zero_mean, corr_cov)
    record("conditional_prior_changes_nuisance_even_with_equal_nuisance_marginals",
           abs(ratio_gradient[1]) > 1,
           task_direction=float(ratio_gradient[0]), nuisance_direction=float(ratio_gradient[1]))

    # Gradient of smoothed log density ratio at fixed Gaussian probes.
    probes = rng.normal(size=(16, 2))
    y = np.array([0.6, -0.3])
    target_cov = np.array([[0.9, -0.1], [-0.1, 1.1]])

    def objective(z):
        return np.mean([log_density((1 - t) * z + t * e, t, other_mean, target_cov)
                        - log_density((1 - t) * z + t * e, t, mean, cov) for e in probes])

    queries = (1 - t) * y + t * probes
    analytic = (1 - t) ** 2 / t * np.mean(velocity(queries, t, mean, cov)
                                           - velocity(queries, t, other_mean, target_cov), axis=0)
    fd = np.array([(objective(y + 1e-5 * e) - objective(y - 1e-5 * e)) / 2e-5 for e in np.eye(2)])
    error = float(np.linalg.norm(analytic - fd))
    record("clean_state_likelihood_ratio_gradient", error < 1e-8, finite_difference_error=error)

    # Both maps are bijections with exact cycle consistency and the same target law.
    # Source (A,N)~N(0,I), target law N((1,0),I); second map reverses N.
    p = np.array([0.3, 1.4])
    edited = np.array([p[0] + 1, -p[1]])
    returned = np.array([edited[0] - 1, -edited[1]])
    record("exact_cycle_and_marginals_do_not_identify_preserved_content",
           np.allclose(returned, p) and edited[1] != p[1],
           cycle_error=float(np.linalg.norm(returned - p)), nuisance_change=float(edited[1] - p[1]))

    # Orthogonal to a reconstruction velocity does not preserve a different feature.
    reconstruction_velocity, edit = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    keep_jacobian = np.array([[0.0, 1.0]])
    record("velocity_orthogonality_is_not_feature_preservation",
           reconstruction_velocity @ edit == 0 and (keep_jacobian @ edit)[0] != 0,
           orthogonality=float(reconstruction_velocity @ edit), feature_change=float((keep_jacobian @ edit)[0]))

    # Differential feature preservation alone allows finite-step drift on a circle.
    p, tangent, step = np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.2
    feature_error = float(np.dot(p + step * tangent, p + step * tangent) - 1)
    record("finite_step_requires_actual_feature_check", p @ tangent == 0 and feature_error > 0.03,
           finite_step_feature_error=feature_error)

    out = dict(date="2026-11-06", scope="Analytic models and seeded synthetic arrays only; no FLUX/SD3 inference",
               seed=20261106, passed=sum(c["passed"] for c in checks), total=len(checks), checks=checks)
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_gaussian_example(Path(__file__).with_suffix(".png"))
    print(json.dumps(dict(passed=out["passed"], total=out["total"], result=str(path)), ensure_ascii=False))


def plot_gaussian_example(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, 0.8, 301)
    a, x = 1 - t, 2.0
    var = a**2 + t**2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(t, a * x - t, a * x + t, color="#a6c8e4", alpha=0.5,
                    label="Random noising: conditional mean +/- 1 SD")
    ax.plot(t, np.sqrt(var) * x, color="#176c44", label="Exact source ODE trajectory")
    ax.plot(t, a * x, "--", color="#296299", label="Deterministic first-order lift")
    ax.set(title="Same marginal law, different source coupling", xlabel="Noise time t", ylabel="Query q(t)")
    ax.legend(fontsize=7)
    ax = axes[1]
    labels = ["Exact ODE\ninverse", "Posterior mean\nconsistent query"]
    ax.bar(np.arange(2) - 0.18, [2 - np.sqrt(2), 0], width=0.36,
           label="Absolute posterior prediction error", color="#677fa4")
    ax.bar(np.arange(2) + 0.18, [0, 2 * np.sqrt(2) - 2], width=0.36,
           label="Absolute ODE return error", color="#b66d46")
    ax.set(xticks=np.arange(2), xticklabels=labels, ylabel="Absolute error", title="D(q)=x is not exact inversion (t=0.5)")
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(t, t / var, label="Raw same-query velocity difference", color="#b66d46")
    ax.plot(t, a**2 / var, label="Rescaled clean-state gradient", color="#176c44")
    ax.set(title="Text signal degenerates at t=0", xlabel="Noise time t", ylabel="Signal (source/target means: 0 and 1)")
    ax.legend(fontsize=7)
    fig.suptitle("Analytic Gaussian examples — not pretrained-model or image-editing results", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    run()
