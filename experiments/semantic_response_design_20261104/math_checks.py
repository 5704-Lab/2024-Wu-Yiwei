"""Constructed checks for semantic-response experiment design; no image-model evidence.

Run: python experiments/semantic_response_design_20261104/math_checks.py
Only writes outputs/math_checks_v1.json; old differing output is never overwritten.
"""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(20261104)
    rows = []

    def check(name, ok, **evidence):
        rows.append(dict(name=name, passed=bool(ok), evidence=evidence))
        assert ok, name

    # Arbitrary scorer temperature changes an alleged semantic energy share.
    share = lambda scale: (.5 * scale)**2 / (1 + (.5 * scale)**2)
    check('raw_gradient_energy_share_depends_on_score_scale',
          share(1) != share(10), original=share(1), rescaled=share(10))
    d = np.array([1., 0.])
    unit = d / np.linalg.norm(d)
    overlap = sum(float(np.dot(unit, d)**2) for _ in range(2)) / np.dot(d, d)
    check('overlapping_projections_double_count', overlap == 2., summed_share=overlap)
    # A class label alone may change the data distribution of pose.
    mu_s, mu_t = np.zeros(2), np.array([1., .8])
    log_ratio_gradient = mu_t - mu_s
    check('class_condition_can_have_unrequested_pose_score_gradient',
          log_ratio_gradient[1] != 0, class_gradient=float(log_ratio_gradient[0]),
          pose_gradient=float(log_ratio_gradient[1]))

    # The symmetric CFG decomposition is exact, but not a license partition.
    ts, tt, ut, ss, us = rng.normal(size=(5, 13))
    gs, gt = 3.5, 13.5
    guided = lambda v, u, scale: u + scale * (v - u)
    original = guided(tt, ut, gt) - guided(ss, us, gs)
    text = .5 * (gt + gs) * (tt - ts)
    asym = (gt - gs) * (.5 * (tt + ts) - ut)
    state = guided(ts, ut, gs) - guided(ss, us, gs)
    err = float(np.max(np.abs(original - text - asym - state)))
    check('text_cfg_state_identity', err < 1e-12, max_abs_error=err)
    same = rng.normal(size=13)
    drift = guided(same, ut, gt) - guided(same, ut, gs)
    check('same_prompt_asymmetric_cfg_nonzero', np.linalg.norm(drift) > 0,
          norm=float(np.linalg.norm(drift)))
    check('same_prompt_equal_cfg_noop',
          np.array_equal(guided(same, ut, gs)-guided(same, ut, gs), np.zeros(13)))
    pieces = [text, asym, state]
    cross = sum(2 * np.dot(pieces[i], pieces[j]) for i in range(3) for j in range(i+1, 3))
    check('branch_norms_need_cross_terms',
          np.isclose(np.dot(original, original), sum(np.dot(b, b) for b in pieces)+cross),
          cross_terms=float(cross))

    # Exact Gaussian RF: z=(1-t)y+t*epsilon; clean pullback includes (1-t).
    tau = .35
    a = 1-tau
    variance_x = .7
    var_z = a*a*variance_x + tau*tau
    y, eps = rng.normal(size=(2, 5))
    ms, mt = rng.normal(size=(2, 5))
    z = a*y + tau*eps

    def gaussian_v(mean):
        ex = mean + a*variance_x/var_z*(z-a*mean)
        return (z-ex)/tau

    delta_v = gaussian_v(mt)-gaussian_v(ms)
    clean_gradient = a*a*(mt-ms)/var_z
    pullback = -a*a/tau*delta_v
    check('gaussian_rf_clean_logratio_pullback',
          np.allclose(clean_gradient, pullback),
          max_abs_error=float(np.max(np.abs(clean_gradient-pullback))))

    # Denoising evidence depends on y through both the query and target epsilon-y.
    A = rng.normal(size=(5, 5))*.2
    b = rng.normal(size=5)
    def loss(point):
        residual = A@(a*point+tau*eps)+b-eps+point
        return float(np.dot(residual, residual))
    residual = A@z+b-eps+y
    analytical = 2*(a*A+np.eye(5)).T@residual
    h = 1e-5
    finite = np.array([(loss(y+h*np.eye(5)[i])-loss(y-h*np.eye(5)[i]))/(2*h) for i in range(5)])
    loss_error = float(np.max(np.abs(analytical-finite)))
    check('denoising_loss_full_chain_derivative', loss_error < 1e-7,
          max_abs_error=loss_error)
    missing_identity = 2*(a*A).T@residual
    check('denoising_target_derivative_cannot_be_dropped',
          np.linalg.norm(analytical-missing_identity) > .1,
          missing_term_error=float(np.linalg.norm(analytical-missing_identity)))

    # Locally additive semantic responses, unlike overlapping energy fractions.
    J = rng.normal(size=(4, 13))
    check('directional_response_adds_for_exact_velocity_pieces',
          np.allclose(J@original, J@text+J@asym+J@state))
    B = rng.normal(size=(13, 3))
    check('response_matrix_is_jacobian_times_actions',
          np.allclose(J@B, np.stack([J@B[:, i] for i in range(3)], axis=1)))
    chart=np.diag(np.linspace(.5, 2., 13))
    check('covector_velocity_pairing_invariant_under_linear_coordinates',
          np.allclose((J@np.linalg.inv(chart))@(chart@original), J@original))
    score_delta, calibration_delta, temperature = .4, .8, 7.
    check('factor_calibrated_finite_response_invariant_to_score_rescaling',
          np.isclose(score_delta/calibration_delta,
                     (temperature*score_delta)/(temperature*calibration_delta)))
    # A reader with identical class/pose rows cannot identify the two factors.
    mixed = np.array([[1., 1.], [1., 1.]])
    check('reader_aliasing_makes_distinct_factors_unidentifiable',
          np.array_equal(mixed@np.array([1., 0.]), mixed@np.array([0., 1.])),
          rank=int(np.linalg.matrix_rank(mixed)))
    check('small_gradient_does_not_bound_finite_semantic_change',
          2*0. == 0. and (1.**2-0.**2) == 1.,
          endpoint_gradient=0., finite_change=1.)

    # Whole-step path integration explains the score difference, not necessity.
    def score(q): return np.sin(q[0])+q[0]*q[1]+q[1]**3
    def grad(q): return np.array([np.cos(q[0])+q[1], q[0]+3*q[1]**2])
    start, step = np.array([.2, -.4]), np.array([.7, .8])
    nodes, weights = np.polynomial.legendre.leggauss(12)
    integral = sum(w*np.dot(grad(start+(.5*n+.5)*step), step)*.5 for n, w in zip(nodes, weights))
    change = score(start+step)-score(start)
    check('integrated_directional_gradient_matches_finite_score_change',
          np.isclose(integral, change, atol=1e-12),
          integral=float(integral), finite=float(change))
    # Integral of positive derivatives double-counts excursions on a path.
    path = np.linspace(0., 1., 10001)
    positive_work = float(np.trapezoid(np.maximum(2*np.pi*np.cos(2*np.pi*path), 0.), path))
    endpoint = float(np.sin(2*np.pi)-np.sin(0.))
    check('positive_local_work_is_not_endpoint_violation_growth',
          positive_work>1.9 and abs(endpoint)<1e-12,
          positive_variation=positive_work, endpoint_change=endpoint)

    violation = lambda pose: max(abs(pose)-.1, 0.)
    check('large_pose_restoration_can_reduce_protected_violation',
          violation(.2)<violation(.8) and abs(.2-.8)>.1,
          pose_motion=.6, violation_before=violation(.8), violation_after=violation(.2))
    check('small_per_step_leak_ratio_does_not_bound_cumulative_drift',
          .01/1. < .05 and 100*.01 > .1,
          per_step_ratio=.01, accumulated_drift=1., allowed_drift=.1)
    def ratio(task_gain, leak): return None if task_gain<=0 else leak/task_gain
    check('zero_task_progress_ratio_is_undefined', ratio(0., .1) is None)
    # Same norm does not imply equal task progress; retain both controls.
    e = np.array([1., 0.])
    u, w = np.array([1., 0.]), np.array([.6, .8])
    check('equal_norm_is_not_equal_task_completion',
          np.isclose(np.linalg.norm(u), np.linalg.norm(w)) and np.dot(e, u)!=np.dot(e, w))

    # Binary prompt main effects and all interactions reconstruct every cell.
    cells = {bits:rng.normal(size=7) for bits in itertools.product([0, 1], repeat=3)}
    effects = {}
    for bits in cells:
        effects[bits] = sum((-1)**(sum(bits)-sum(sub))*value for sub, value in cells.items()
                            if all(s<=b for s, b in zip(sub, bits)))
    factorial_error = max(float(np.max(np.abs(value-sum(effect for sub, effect in effects.items()
                                if all(s<=b for s, b in zip(sub, bits)))))) for bits, value in cells.items())
    check('three_factor_mobius_reconstruction_includes_interactions', factorial_error<1e-12,
          max_abs_error=factorial_error)
    # Orthogonality to a reconstruction direction is not pose invariance.
    rec, edit, pose_row = np.array([1., 0.]), np.array([0., 1.]), np.array([0., 1.])
    check('reconstruction_orthogonality_does_not_preserve_pose',
          np.dot(rec, edit)==0 and np.dot(pose_row, edit)==1.)
    # A zero feedback step must remain available when finite constraints hold.
    safe_y = np.array([.8, 0.])
    d0 = np.array([.1, 0.])
    h_keep = lambda q: .1**2-q[1]**2
    check('nominal_feasible_step_requires_no_correction',
          h_keep(safe_y+d0)>=0 and (safe_y+d0)[0]>=safe_y[0])
    # Independently validated keep Jacobian, not an assumed legal text subspace.
    target=np.array([1., 1.]);keep=np.array([[0., 1.]])
    projector=np.eye(2)-np.linalg.pinv(keep)@keep
    feasible=projector@target
    check('target_progress_exists_in_validated_keep_nullspace',
          np.allclose(keep@feasible, 0.) and np.dot(target, feasible)>0,
          normalized_max_progress=float(np.linalg.norm(feasible)))
    check('fully_aliased_task_and_keep_have_no_first_order_nullspace_progress',
          np.allclose(projector@keep[0],0.))
    restricted=np.array([[1.], [1.]])
    check('restricted_action_bank_can_remove_full_space_feasibility',
          np.linalg.matrix_rank(keep@restricted)==1 and np.linalg.norm(feasible)>0.)
    actions=np.array([[1., 2.],[0., 1.]])
    coefficients=np.array([.5, .8])
    latent_cost=float(np.dot(actions@coefficients, actions@coefficients))
    gram_cost=float(coefficients@(actions.T@actions)@coefficients)
    check('action_coefficient_cost_requires_latent_gram_matrix',
          np.isclose(latent_cost,gram_cost) and not np.isclose(latent_cost,np.dot(coefficients,coefficients)),
          actual_latent_cost=latent_cost, coefficient_cost=float(np.dot(coefficients,coefficients)))
    payload = dict(scope='constructed_math_and_counterexamples_only', image_model_runs=0,
                   seed=20261104, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   total=len(rows), passed=sum(r['passed'] for r in rows), checks=rows)
    dest = HERE/'outputs/math_checks_v1.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    serialized=json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if dest.exists() and dest.read_text()!=serialized:
        raise RuntimeError('Existing output differs; use a fresh version before writing.')
    dest.write_text(serialized)
    print(json.dumps(dict(total=payload['total'], passed=payload['passed'],
                         scope=payload['scope'], output=str(dest)), ensure_ascii=False))


if __name__=='__main__':
    main()
