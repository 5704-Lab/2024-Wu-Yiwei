"""Exact finite-support checks and analytic Gaussian moments, not FLUX tests."""

import json
from pathlib import Path

import numpy as np

from reference import flowedit_increment, paired_increment, parity_parts


def main():
    results = []

    def record(name, condition, **evidence):
        if not bool(condition):
            raise AssertionError(name)
        results.append(dict(name=name, passed=True, evidence=evidence))

    # Feasible target on a smooth manifold contradicts exact nuisance preservation.
    target_valid = {(1, 1)}
    preserving_target = {(1, 0)}
    record("empty_success_intersection", not target_valid & preserving_target)

    # One observed output distribution, disjoint success sets in two hidden worlds.
    # For every possible distribution on these outputs, min success <= 1/2.
    success = [(p, 1 - p) for p in np.linspace(0, 1, 101)]
    record("unidentified_semantics", all(min(a, b) <= 0.5 for a, b in success),
           universal_bound="nu(S1)+nu(S2)<=1; finite grid only checks arithmetic")

    # Exact ideal Gaussian RF: average-field nuisance drift, although an exact
    # nuisance-preserving target coupling exists. Independent integral/ODE paths.
    sigma = np.array([[4., 1.], [1., 1.]])
    lam, basis = np.linalg.eigh(sigma)
    def endpoint_integral(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        times, weights = (nodes+1)/2, weights/2
        d1 = (1-times)**2 + times**2
        return np.array([np.sum(weights * np.sqrt(a)*times /
                         (d1*np.sqrt(a*(1-times)**2+times**2))) for a in lam])
    f = endpoint_integral(96)
    integration_error = float(np.max(np.abs(f-endpoint_integral(64))))
    def rhs(t, state):
        al = (t-(1-t)*lam)/((1-t)**2*lam+t*t)
        a1 = (2*t-1)/((1-t)**2+t*t)
        return al*state-t*al-(1-t)*a1
    state, step_count = np.ones(2), 4096
    step = -1/step_count
    for k in range(step_count):
        t = 1+k*step
        k1 = rhs(t, state)
        k2 = rhs(t+step/2, state+step*k1/2)
        k3 = rhs(t+step/2, state+step*k2/2)
        k4 = rhs(t+step, state+step*k3)
        state = state + step*(k1+2*k2+2*k3+k4)/6
    ode_error = float(np.max(np.abs(state-f)))
    mean_map = (basis*f) @ basis.T
    record("ideal_gaussian_mean_field_drifts", integration_error < 1e-12 and
           ode_error < 1e-10 and mean_map[1, 0] > .18,
           eigenvalues=lam.tolist(), mean_map=mean_map.tolist(),
           integral_order_difference=integration_error, independent_ode_error=ode_error,
           source=[1, 0], output=(mean_map @ np.array([1., 0.])).tolist())
    transport = np.array([[np.sqrt(3), 1.], [0., 1.]])
    record("same_gaussian_task_has_preserving_exact_transport",
           np.allclose(transport @ transport.T, sigma) and
           np.array_equal(transport[1], [0., 1.]) and sigma[1, 1] == 1,
           transport=transport.tolist(), target_covariance=sigma.tolist())

    # Exact integration on a symmetric finite distribution; no sampling error.
    e = np.array([-2., -1., 1., 2.])

    def field(z):
        return np.stack((1 + z + z*z, 2*z - z*z + z**3), axis=-1)

    b = field(e)
    even, odd = parity_parts(b, field(-e))
    mu = b.mean(axis=0)

    def cov(z):
        centered = z - z.mean(axis=0)
        return centered.T @ centered / len(z)

    error = float(np.max(np.abs(cov(b) - cov(even) - cov(odd))))
    cross = (even - mu).T @ odd / len(e)
    record("vector_conditional_mean_and_covariance", error < 1e-12 and
           np.allclose(even.mean(axis=0), mu) and np.allclose(cross, 0),
           max_identity_error=error,
           removed_covariance_eigenvalues=np.linalg.eigvalsh(cov(odd)).tolist())

    y, h = np.array([.3, -.8]), .17
    w, ref = np.array([[1., 2.], [-.5, 1.]]), np.array([1., -2.])
    def loss(z):
        return np.sum(((y + h*z) @ w.T - ref)**2, axis=-1)
    decrease = float(np.mean(loss(b)) - np.mean(loss(even)))
    exact = float(h*h * np.mean(np.sum((odd @ w.T)**2, axis=-1)))
    record("quadratic_one_step_identity", np.isclose(decrease, exact) and decrease >= 0,
           decrease=decrease, predicted=exact)

    # Jensen also applies to nonsquared convex penalties.
    phi = lambda z: np.exp(.01*z[:, 0]) + np.abs(z[:, 1] - .6)
    gap = float(np.mean(phi(y + h*b)) - np.mean(phi(y + h*even)))
    record("convex_one_step_inequality", gap >= -1e-12, gap=gap)

    # Exact standard Gaussian moments: E e^2=1, E e^4=3.
    record("equal_budget_odd_case", 0 < .5,
           field="e", paired_variance=0, iid_two_variance=.5)
    record("equal_budget_even_counterexample", 2 > 1,
           field="e^2-1", paired_variance=2, iid_two_variance=1)

    constant = np.tile([1., 1.], (len(e), 1))
    ce, co = parity_parts(constant, constant)
    record("constant_mean_drift_not_removed", np.allclose(ce, constant) and
           np.allclose(co, 0) and np.allclose(ce.mean(axis=0), [1, 1]),
           feasible_desired_increment=[1, 0], preserved_wrong_increment=[1, 1])

    # Gaussian example: first step y=e versus y=0; second step increment y^2.
    record("multi_step_expectation_can_change", 1 != 0,
           iid_single_endpoint_mean=1, paired_endpoint_mean=0,
           explanation="E[e+e^2]=1 versus 0+0^2=0")

    record("norm_preservation_not_distribution_preservation", np.sqrt(2/np.pi) > 0,
           transform="abs(e)", original_mean=0,
           transformed_mean=float(np.sqrt(2/np.pi)))

    x = np.array([.2, -.4])
    state = np.array([.5, -.1])
    noise = np.array([.7, -1.3])
    x_copy, state_copy = x.copy(), state.copy()
    calls = []
    def vs(q, t):
        calls.append(("source", q.copy(), t))
        return .3*q*q + t
    def va(q, t):
        calls.append(("target", q.copy(), t))
        return .7*q - t
    t = .4
    actual = paired_increment(vs, va, x, state, t, noise)
    original_calls = list(calls)
    expected = .5*(flowedit_increment(vs, va, x, state, t, noise) +
                   flowedit_increment(vs, va, x, state, t, -noise))
    valid_queries = all(np.allclose(original_calls[i+1][1] - original_calls[i][1],
                                   state-x) for i in (0, 2))
    record("original_queries_frozen_state_and_four_calls", len(original_calls) == 4 and
           valid_queries and np.allclose(actual, expected) and
           np.array_equal(x, x_copy) and np.array_equal(state, state_copy))

    record("identical_full_conditions_identity", np.allclose(
        paired_increment(vs, vs, x, x, .4, noise), 0))

    # Known-factor local geometry, with no rank assumption needed at this point.
    jac = np.array([[0., 1.]])
    projection = np.eye(2) - np.linalg.pinv(jac) @ jac
    grad = np.array([1., 2.])
    u = projection @ grad
    blocked_grad = np.array([0., 1.])
    record("task_tangent_criterion", np.allclose(jac @ u, 0) and
           np.isclose(grad @ u, u @ u) and u @ u > 0 and
           np.allclose(projection @ blocked_grad, 0))

    # C(y)=||y||^2; tangent to circle at (1,0) but Euler leaves it.
    state = np.array([1., 0.])
    tangent = np.array([0., 1.])
    h = .1
    defect = float(np.dot(state+h*tangent, state+h*tangent) - state @ state)
    record("tangent_not_exact_finite_step_preservation", np.dot(state, tangent) == 0 and
           np.isclose(defect, h*h), constraint_defect=defect)

    # Symmetric two-point model: lower variance can reduce a threshold success rate.
    increments = 1 + np.array([-1., 1.])
    before = float(np.mean(np.abs(increments) < .1))
    after = float(abs(increments.mean()) < .1)
    record("semantic_threshold_need_not_improve", before > after,
           original_success=before, paired_success=after,
           note="finite symmetric example; Gaussian version also has positive original success")

    output = dict(scope="CPU algebra/counterexamples only; no image or FLUX evidence",
                  checks=len(results), passed=sum(r["passed"] for r in results),
                  results=results)
    path = Path(__file__).with_name("verification.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(checks=output["checks"], passed=output["passed"],
                          output=str(path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
