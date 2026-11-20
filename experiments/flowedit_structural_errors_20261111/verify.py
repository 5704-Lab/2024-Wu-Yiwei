"""CPU algebra and analytic counterexamples for three structural deviations.

Gaussian integrals use deterministic quadrature, not empirical image evidence.
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def gaussian_rule(order=24):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return np.sqrt(2)*nodes, weights/np.sqrt(np.pi)


def interval_rule(order=24):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes+1)/2, weights/2


def covariance(values, weights):
    mean = weights @ values
    centered = values-mean
    return (centered.T*weights) @ centered


def gaussian_velocity_matrix(cov, t):
    eye = np.eye(len(cov))
    alpha = 1-t
    return (t*eye-alpha*cov) @ np.linalg.inv(alpha*alpha*cov+t*t*eye)


def main():
    checks = []
    def record(name, ok, **evidence):
        if not bool(ok):
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True, evidence=evidence))

    eps, w = gaussian_rule()
    r, rw = interval_rule()
    x, t = 1., .5
    m, z = (1-t)*x, np.sqrt((1-t)**2+t*t)*x
    q = m+t*eps
    conditional_w2 = (m-z)**2+t*t
    record("S1_conditional_coupling_distance", np.isclose(w @ (q-z)**2, conditional_w2),
           w2_squared=conditional_w2, conditional_noise_variance=t*t)

    g = lambda u, d: 1.2*(u+d)
    source_bias = float(w @ g(q, .3)-g(z, .3))
    record("S2_ideal_affine_mean_bias", np.isclose(source_bias, 1.2*(m-z)) and abs(source_bias)>.24,
           bias=source_bias, curvature_bias=0.)

    # Independent 2D quadrature for the Bayes residual, source variance one.
    xx, ee = np.meshgrid(eps, eps, indexing="ij")
    ww = w[:, None]*w[None, :]
    qq = .5*(xx+ee)
    predicted_velocity = np.zeros_like(qq)
    residual_mse = float(np.sum(ww*(predicted_velocity-(ee-xx))**2))
    record("S4_posterior_variance_residual", np.isclose(residual_mse, .5/.25),
           residual_mse=residual_mse, posterior_variance=.5,
           note="perfect model velocity zero at t=.5; residual is not model error")

    # Heat identity for a cubic difference field (Gaussian moments exact here).
    m, t, d = .7, .4, .3
    cubic_g = lambda u: u**3+2*u-(.5*(u+d)**3-(u+d))
    cubic_laplacian = lambda u: 3*(u-d)
    smoothing = float(w @ cubic_g(m+t*eps)-cubic_g(m))
    heat_integral = .5*t*t*sum(weight*float(w @ cubic_laplacian(m+np.sqrt(t*t*s)*eps))
                             for s, weight in zip(r, rw))
    record("S3_gaussian_heat_identity", np.isclose(smoothing, heat_integral),
           exact_smoothing=smoothing, heat_integral=heat_integral)

    # Same-covariance Gaussian translation: integrate log density ratio.
    alpha = 1-t
    d = .8
    target_normal_mean = 1.2
    target_flow_mean = target_normal_mean+t*d
    sample = target_flow_mean+t*eps
    log_ratio = ((sample-target_normal_mean)**2-(sample-target_flow_mean)**2)/(2*t*t)
    kl = float(w @ log_ratio)
    record("Q1_gaussian_query_translation", np.isclose(kl, .5*d*d) and
           np.isclose(target_flow_mean-target_normal_mean, t*d),
           w2=t*abs(d), kl=kl)

    va = lambda u: np.sin(u)+u*u
    jva = lambda u: np.cos(u)+2*u
    query, t, d = .31, .4, .8
    alpha = 1-t
    query_difference = va(query+alpha*d)-va(query+d)
    jacobian_integral = -t*d*float(rw @ jva(query+alpha*d+r*t*d))
    record("Q2_target_jacobian_integral", np.isclose(query_difference, jacobian_integral),
           exact=query_difference, integral=jacobian_integral)
    record("Q2_ideal_affine_query_bias", np.isclose(.5*1.2*1., .6),
           source_variance=1., target_variance=4., t=.5, d=1., field_bias=.6)

    sigma = np.array([[4., 1.], [1., 1.]])
    t, alpha = .5, .5
    aa = gaussian_velocity_matrix(sigma, t)
    ass = gaussian_velocity_matrix(np.eye(2), t)
    posterior_a = np.linalg.inv(np.linalg.inv(sigma)+alpha*alpha/(t*t)*np.eye(2))
    posterior_s = t*t/(alpha*alpha+t*t)*np.eye(2)
    posterior_gain = alpha*sigma @ np.linalg.inv(alpha*alpha*sigma+t*t*np.eye(2))
    record("V2_posterior_jacobian_identity", np.allclose(posterior_gain, alpha/(t*t)*posterior_a) and
           np.allclose(aa, np.eye(2)/t-alpha/(t**3)*posterior_a),
           target_velocity_jacobian=aa.tolist(), posterior_covariance=posterior_a.tolist())

    m, disp = np.array([1., 2.]), np.array([1., 0.])
    condition_only = (ass-aa) @ m
    mean_full = ass @ m-aa @ (m+disp)
    mean_normal = ass @ m-aa @ (m+alpha*disp)
    transport = -aa @ disp
    record("V1_V4_ideal_nonedit_cross_coupling", np.allclose(mean_full, condition_only+transport) and
           np.isclose(condition_only[1], 0) and np.isclose(mean_full[1], 4/9) and
           np.isclose(mean_normal[1], 2/9),
           same_point_response=condition_only.tolist(), full_mean=mean_full.tolist(),
           normal_noising_mean=mean_normal.tolist(), source=[2, 4], current_state=[3, 4])
    preserving = np.array([[np.sqrt(3), 1.], [0., 1.]])
    record("V5_target_is_preserving_feasible", np.allclose(preserving @ preserving.T, sigma) and
           np.array_equal(preserving[1], [0, 1]), target_covariance=sigma.tolist(),
           preserving_transport=preserving.tolist())

    ex, ey = np.meshgrid(eps, eps, indexing="ij")
    noise = np.stack([ex.ravel(), ey.ravel()], axis=1)
    weights = (w[:, None]*w[None, :]).ravel()
    queries = m+t*noise
    sample_g = queries @ ass.T-(queries+disp) @ aa.T
    actual_cov = covariance(sample_g, weights)
    kk = ass-aa
    predicted_cov = t*t*kk @ kk.T
    record("V3_ideal_covariance_difference_and_variance", np.allclose(kk, alpha/t**3*(posterior_a-posterior_s)) and
           np.allclose(actual_cov, predicted_cov),
           exact_covariance=actual_cov.tolist(), analytic_covariance=predicted_cov.tolist())

    m1, t1 = .7, .4
    q1 = m1+t1*eps
    nonlinear = q1*q1
    actual_var = float(w @ (nonlinear-w @ nonlinear)**2)
    lower = t1*t1*float(w @ (2*q1))**2
    upper = t1*t1*float(w @ (4*q1*q1))
    record("V3_nonlinear_gaussian_variance_bounds", lower <= actual_var <= upper,
           lower=lower, variance=actual_var, upper=upper)

    # This adverse common-noise example uses two exact Gaussian RF fields.
    ap = gaussian_velocity_matrix(np.array([[1/3]]), .5)[0, 0]
    an = gaussian_velocity_matrix(np.array([[3.]]), .5)[0, 0]
    common_var = (.5*(ap-an))**2
    independent_var = .25*(ap*ap+an*an)
    record("V3_shared_noise_not_universally_better", np.isclose(ap, 1) and np.isclose(an, -1) and
           common_var > independent_var, shared_variance=common_var,
           independent_variance=independent_var)

    # J1/J2: exact Gaussian conditional moments; source-zero conditioning does
    # not imply zero target conditioning. Use a possible fixed-time state map.
    dd = np.diag([.1, 0.])
    ar = alpha*np.eye(2)+dd
    mu_matrix = alpha*ass-aa @ ar
    xs, es = np.array([.3, -.8]), np.array([1.1, .2])
    qs, target = alpha*xs+t*es, ar @ xs+t*es
    sample_b = ass @ qs-aa @ target
    mean_b = mu_matrix @ xs
    rs = es-xs-ass @ qs
    rprime = es-xs-mean_b
    ra = rprime-aa @ target
    source_residual_cross_q = -(np.eye(2)+alpha*ass)*alpha+(np.eye(2)-t*ass)*t
    source_residual_cross_r = -(np.eye(2)+alpha*ass) @ ar.T+(np.eye(2)-t*ass)*t
    xi_cross_r = t*t*(ass-aa)
    defect_cross = source_residual_cross_r+xi_cross_r
    cr = ar @ ar.T+t*t*np.eye(2)
    ar_prime = -mu_matrix-np.eye(2)
    cr_prime = ar_prime @ ar.T+ar @ ar_prime.T+2*t*np.eye(2)
    density_covariance_defect = cr_prime-aa @ cr-cr @ aa.T
    record("J1_J2_target_path_and_conditioning_defect", np.allclose(ra, rs+sample_b-mean_b) and
           np.allclose(source_residual_cross_q, 0) and not np.allclose(defect_cross, 0) and
           np.allclose(density_covariance_defect, defect_cross+defect_cross.T),
           source_residual_cross_source_query=source_residual_cross_q.tolist(),
           target_defect_cross_target_query=defect_cross.tolist(),
           target_covariance_equation_defect=density_covariance_defect.tolist())

    # Nonlinear interaction survives even when the mean anchor equals z.
    t, d, z = .4, .7, 0.
    alpha = 1-t
    scalar_g = lambda q, shift: .3*q-np.cos(q+shift)
    q = t*eps
    avg_f = float(w @ scalar_g(q, d))
    avg_n = float(w @ scalar_g(q, alpha*d))
    base_f, base_n = scalar_g(z, d), scalar_g(z, alpha*d)
    interaction = (avg_f-avg_n)-(base_f-base_n)
    exact_interaction = (np.exp(-t*t/2)-1)*(np.cos(alpha*d)-np.cos(d))
    bound = t*abs(d)*np.sqrt(t*t)  # Hessian cos bounded by one; p=1, b=0.
    record("J4_mixed_interaction_and_bound", np.isclose(interaction, exact_interaction) and
           0 < abs(interaction) <= bound,
           interaction=interaction, analytic=exact_interaction, bound=bound)
    telescoped = base_n+(avg_n-base_n)+(avg_f-avg_n)
    incorrectly_added = base_n+(avg_f-base_f)+(avg_f-avg_n)
    record("J3_telescoping_avoids_double_count", np.isclose(telescoped, avg_f) and
           np.isclose(incorrectly_added-avg_f, interaction) and not np.isclose(interaction, 0),
           true_mean=avg_f, telescoped=telescoped, double_count_error=incorrectly_added-avg_f)

    # Full-time ideal mean FlowEdit. Numerical size is not the nonzero proof;
    # strict monotonicity in E2 supplies that proof in the document.
    lam, basis = np.linalg.eigh(sigma)
    def endpoint_integral(order):
        tt, ww = interval_rule(order)
        d1 = (1-tt)**2+tt*tt
        return np.array([np.sum(ww*np.sqrt(a)*tt/(d1*np.sqrt(a*(1-tt)**2+tt*tt))) for a in lam])
    ff = endpoint_integral(96)
    ff_error = float(np.max(np.abs(ff-endpoint_integral(64))))
    def rhs(time, state):
        al = (time-(1-time)*lam)/((1-time)**2*lam+time*time)
        a1 = (2*time-1)/((1-time)**2+time*time)
        return al*state-time*al-(1-time)*a1
    state = np.ones(2)
    h = -1/4096
    for k in range(4096):
        time = 1+k*h
        k1 = rhs(time, state)
        k2 = rhs(time+h/2, state+h*k1/2)
        k3 = rhs(time+h/2, state+h*k2/2)
        k4 = rhs(time+h, state+h*k3)
        state += h*(k1+2*k2+2*k3+k4)/6
    mm = (basis*ff) @ basis.T
    spectral_bounds = all(min(1., np.sqrt(a)) < f < max(1., np.sqrt(a)) for a, f in zip(lam, ff))
    record("E1_E3_complete_trajectory_and_target_mismatch", ff_error < 1e-12 and
           np.max(np.abs(ff-state)) < 1e-10 and mm[1, 0] > .18 and spectral_bounds,
           endpoint_matrix=mm.tolist(), spectral_gains=ff.tolist(),
           target_standard_deviations=np.sqrt(lam).tolist(), integral_order_error=ff_error,
           independent_ode_error=float(np.max(np.abs(ff-state))))

    # A control showing that query displacement is not a universal field error.
    source_a = gaussian_velocity_matrix(np.array([[2.]]), .4)[0, 0]
    same_cov_g = lambda u: source_a*u-source_a*(u+.8)
    same_cov_bias = float(w @ same_cov_g(.6+.4*eps)-same_cov_g(1.3))
    record("zero_source_error_control", abs(same_cov_bias) < 1e-12 and
           float(w @ (same_cov_g(.6+.4*eps)-w @ same_cov_g(.6+.4*eps))**2) < 1e-25,
           source_query_changed=True, mean_field_bias=same_cov_bias)

    out = dict(date="2026-11-11", scope="CPU analytic identities and counterexamples only",
               new_image_model_forwards=0, checks=len(checks),
               passed=sum(c["passed"] for c in checks), results=checks)
    path = HERE / "verification.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(checks=out["checks"], passed=out["passed"], output=str(path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
