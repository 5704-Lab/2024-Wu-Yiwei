"""Finite mathematics audit, without pretrained models or real-image results.

The tasks, nuisance coordinates, and interventions here are declared toy facts.
They do not implement free-text binding or establish semantic editing quality.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp

rng = np.random.default_rng(20261025)
rows = []


def check(name, passed, **metrics):
    rows.append(dict(name=name, passed=bool(passed), metrics=metrics))
    if not passed:
        raise AssertionError((name, metrics))


# Six fixed queries: state={source,current}, text={empty,source,target}.
v = rng.normal(size=(2, 3, 9))
g = v[:, 1:] - v[:, :1]
gamma = 4.2
b_empty = v[0, 0] - v[1, 0]
b_edit = -gamma * (g[0, 1] - g[0, 0])
b_source_state = -gamma * (g[1, 0] - g[0, 0])
b_interaction = -gamma * ((g[1, 1] - g[1, 0]) - (g[0, 1] - g[0, 0]))
guided = lambda i, j, scale: v[i, 0] + scale * (v[i, j] - v[i, 0])
b = guided(0, 1, gamma) - guided(1, 2, gamma)
check('six_query_decomposition', np.allclose(b, b_empty+b_edit+b_source_state+b_interaction))
check('same_state_empty_cancels_from_target_source_contrast',
      np.allclose(g[1, 1]-g[1, 0], v[1, 2]-v[1, 1]))
gs, gt = 3.5, 13.5
asymmetric = guided(0, 1, gs) - guided(1, 2, gt)
extra_cfg = (gs-gt)*g[0, 0]
check('asymmetric_cfg_exact_residual', np.allclose(asymmetric,
      (guided(0, 1, gt)-guided(1, 2, gt))+extra_cfg))
same_text_initial = guided(0, 1, gs)-guided(0, 1, gt)
check('same_text_asymmetric_cfg_is_not_no_edit',
      np.allclose(same_text_initial, (gs-gt)*g[0, 0]) and np.linalg.norm(same_text_initial)>1)
check('same_text_matched_cfg_initial_identity', np.array_equal(
      guided(0, 1, gs)-guided(0, 1, gs), np.zeros(9)))

x, y, noise = rng.normal(size=(3, 9)); t = .61; h = .03
qs = (1-t)*x+t*noise
qy = qs+y-x
check('flowedit_query_is_not_forward_noise_of_current_y',
      np.allclose(qy-((1-t)*y+t*noise), t*(y-x)) and np.linalg.norm(t*(y-x))>.1)
effective_clean = x+(y-x)/(1-t)
check('query_effective_clean_coordinate', np.allclose(qy, (1-t)*effective_clean+t*noise))
mu_source = qs-t*v[0, 1]
mu_target = qy-t*v[1, 2]
anchor_target = x+mu_target-mu_source
b_raw = v[0, 1]-v[1, 2]
check('source_calibrated_prediction_is_exactly_flowedit',
      np.allclose(y+(h/t)*(anchor_target-y), y+h*b_raw))
mu_empty_s = qs-t*v[0, 0]; mu_empty_y = qy-t*v[1, 0]
anchor_empty = x+mu_empty_y-mu_empty_s
check('source_calibrated_empty_branch_feedback', np.allclose(anchor_empty-y, t*b_empty))
check('calibrated_target_minus_empty_equals_condition_increment',
      np.allclose(anchor_target-anchor_empty, t*(b_raw-b_empty)))
check('source_calibration_zero_state_zero_condition_identity',
      np.allclose(x+mu_source-mu_source, x))
constant_bias = rng.normal(size=9)
check('paired_calibration_cancels_common_bias', np.allclose(
      x+(mu_target+constant_bias)-(mu_source+constant_bias), anchor_target))
varying_bias = .3*(y-x)
check('paired_calibration_does_not_cancel_state_dependent_bias',
      np.allclose(x+(mu_target+varying_bias)-mu_source-anchor_target, varying_bias)
      and np.linalg.norm(varying_bias)>.1)
empty_trajectory = x.copy()
for time in np.linspace(.8, .1, 8):
    query_s = (1-time)*x+time*noise
    query_y = query_s+(empty_trajectory-x)
    empty_trajectory += .01*(np.sin(query_s)-np.sin(query_y))
check('both_ends_empty_full_trajectory_is_trivially_identity',
      np.array_equal(empty_trajectory, x))
source_text_offset = np.array([.2, -.3])
initial_query = np.array([.4, -.2])
source_velocity = initial_query+source_text_offset
empty_velocity = initial_query
check('empty_target_with_nonempty_source_is_not_a_no_edit_control',
      np.allclose(source_velocity-empty_velocity, source_text_offset)
      and np.linalg.norm(source_velocity-empty_velocity)>0.)

# Independently derive scores by finite differences of exact mixture densities.
atoms = rng.normal(size=(5, 3))*1.7
prior = np.array([.35, .15, .10, .10, .30])
conditional = np.array([.10, .20, .20, .40, .10])
t = .43; q = rng.normal(size=3)


def posterior(qval, weights):
    residual = qval-(1-t)*atoms
    logits = np.log(weights)-np.sum(residual**2, axis=1)/(2*t*t)
    normalizer = logsumexp(logits)
    probs = np.exp(logits-normalizer)
    mean = probs@atoms
    score = -(probs[:, None]*residual).sum(axis=0)/(t*t)
    velocity = (qval-mean)/t
    return mean, score, velocity, float(normalizer)


mean, score, velocity, _ = posterior(q, prior)
mc, sc, vc, _ = posterior(q, conditional)
eps = 1e-6; eye = np.eye(3)
score_fd = np.array([(posterior(q+eps*e, prior)[3]-posterior(q-eps*e, prior)[3])/(2*eps) for e in eye])
ratio_fd = np.array([((posterior(q+eps*e, conditional)[3]-posterior(q+eps*e, prior)[3])
                    -(posterior(q-eps*e, conditional)[3]-posterior(q-eps*e, prior)[3]))/(2*eps) for e in eye])
check('exact_gaussian_mixture_score_matches_density_derivative',
      np.max(np.abs(score_fd-score))<2e-9, max_error=float(np.max(np.abs(score_fd-score))))
check('rectified_velocity_score_identity', np.allclose(score, -q/t-(1-t)*velocity/t))
check('conditional_empty_velocity_is_density_ratio_gradient',
      np.max(np.abs(vc-velocity+t/(1-t)*ratio_fd))<2e-9,
      max_error=float(np.max(np.abs(vc-velocity+t/(1-t)*ratio_fd))))
check('denoiser_difference_is_scaled_velocity_difference', np.allclose(mc-mean, -t*(vc-velocity)))
log_weights = np.log(prior)-np.sum((q-(1-t)*atoms)**2, axis=1)/(2*t*t)
probabilities = np.exp(log_weights-logsumexp(log_weights))
centered = atoms-mean
covariance = (probabilities[:, None]*centered).T@centered
mean_jacobian_fd = np.column_stack([(posterior(q+eps*e, prior)[0]-posterior(q-eps*e, prior)[0])/(2*eps) for e in eye])
velocity_jacobian_fd = np.column_stack([(posterior(q+eps*e, prior)[2]-posterior(q-eps*e, prior)[2])/(2*eps) for e in eye])
check('posterior_covariance_equals_mean_susceptibility',
      np.max(np.abs(mean_jacobian_fd-(1-t)/(t*t)*covariance))<2e-8,
      max_error=float(np.max(np.abs(mean_jacobian_fd-(1-t)/(t*t)*covariance))))
check('posterior_fluctuations_determine_empty_velocity_jacobian',
      np.max(np.abs(velocity_jacobian_fd-(np.eye(3)/t-(1-t)/t**3*covariance)))<2e-8,
      max_error=float(np.max(np.abs(velocity_jacobian_fd-(np.eye(3)/t-(1-t)/t**3*covariance)))))
q_start = q-.3*rng.normal(size=3); segment = q-q_start
velocity_work = quad(lambda s: float((posterior(q_start+s*segment, conditional)[2]
                                     -posterior(q_start+s*segment, prior)[2])@segment), 0., 1., epsabs=1e-12)[0]
ratio_change = ((posterior(q, conditional)[3]-posterior(q, prior)[3])
                -(posterior(q_start, conditional)[3]-posterior(q_start, prior)[3]))
check('conditional_velocity_line_integral_recovers_fixed_time_log_ratio_change',
      abs(-(1-t)/t*velocity_work-ratio_change)<2e-12,
      error=float(abs(-(1-t)/t*velocity_work-ratio_change)))
triangle = [q_start, q, q+.2*rng.normal(size=3), q_start]
circulation = 0.
for start, end in zip(triangle[:-1], triangle[1:]):
    tangent = end-start
    circulation += quad(lambda s: float((posterior(start+s*tangent, conditional)[2]
                                        -posterior(start+s*tangent, prior)[2])@tangent), 0., 1., epsabs=1e-12)[0]
check('exact_conditional_contrast_has_zero_fixed_time_circulation',
      abs(circulation)<1e-12, circulation=float(circulation))
# A learned field with rotational error need not be a gradient field.
rotational_loop = quad(lambda angle: float(np.array([-np.sin(angle), np.cos(angle)])
                                          @np.array([-np.sin(angle), np.cos(angle)])), 0., 2*np.pi)[0]
check('generic_learned_field_need_not_define_path_independent_semantic_energy',
      abs(rotational_loop-2*np.pi)<1e-12, circulation=float(rotational_loop))
source_atom = 1.; sampled_noise = -1.; symmetric_atoms = np.array([-1., 1.])
source_q = .5*source_atom+.5*sampled_noise
likelihood = np.exp(-(source_q-.5*symmetric_atoms)**2/(2*.5**2))
source_mean = float((likelihood/likelihood.sum())@symmetric_atoms)
check('even_exact_empty_posterior_does_not_reconstruct_each_source',
      abs(source_mean-source_atom)==1., reconstruction_error=float(abs(source_mean-source_atom)))

# An exact bimodal unconditional model can repel a source-calibrated state.
t = .5; atom_radius = 2.
v_empty = lambda z: (z-atom_radius*np.tanh((1-t)*atom_radius*z/(t*t)))/t
derivative_fd = (v_empty(1e-6)-v_empty(-1e-6))/(2e-6)
derivative_exact = (1-(1-t)*atom_radius**2/(t*t))/t
check('exact_empty_feedback_can_be_locally_repulsive',
      abs(derivative_fd-derivative_exact)<1e-8 and derivative_exact<0,
      velocity_jacobian=float(derivative_exact))
delta = .001; step = .01
new_delta = delta+step*(v_empty(0.)-v_empty(delta))
check('source_calibration_fixed_point_does_not_imply_stability',
      abs(new_delta)>abs(delta), amplification=float(new_delta/delta))

# Text response can be fully entangled with the protected coordinate.
branches = np.array([[1., 2., -1.], [1., 2., -1.]])
lp = linprog(np.zeros(3), A_eq=branches, b_eq=np.array([1., 0.]),
             bounds=[(None, None)]*3, method='highs')
check('branch_reweighting_cannot_create_missing_task_keep_direction',
      lp.status==2, solver_status=int(lp.status))
expanded = np.column_stack([branches, np.array([0., -1.])])
lp_repair = linprog(np.zeros(4), A_eq=expanded, b_eq=np.array([1., 0.]),
                    bounds=[(None, None)]*4, method='highs')
check('independent_source_restoration_direction_restores_feasibility',
      lp_repair.success and np.linalg.norm(expanded@lp_repair.x-np.array([1., 0.]))<1e-10)
task_readout = np.array([1., 0.]); pose_readout = np.array([0., 1.])
raw_text_change = branches[:, 0]; repaired_change = np.array([1., 0.])
check('large_text_response_does_not_certify_necessity_of_pose',
      task_readout@raw_text_change==task_readout@repaired_change==1.
      and pose_readout@raw_text_change==1. and pose_readout@repaired_change==0.)

# A strictly convex task/keep QP checked against an independent constrained solve.
n = 7; temp = rng.normal(size=(n, n)); G = temp.T@temp+.5*np.eye(n)
K = rng.normal(size=(2, n)); C = rng.normal(size=(1, n))
M = np.vstack([K, C]); desired = np.array([-.1, .2, .7])
GiMt = np.linalg.solve(G, M.T)
u = GiMt@np.linalg.solve(M@GiMt, desired)
numeric = minimize(lambda z:.5*z@G@z, np.zeros(n), jac=lambda z:G@z,
                   constraints={'type':'eq', 'fun':lambda z:M@z-desired, 'jac':lambda z:M},
                   method='SLSQP', options={'ftol':1e-12, 'maxiter':300})
check('joint_minimum_change_qp_matches_independent_solver',
      numeric.success and np.linalg.norm(u-numeric.x)<2e-6,
      solution_error=float(np.linalg.norm(u-numeric.x)))
check('joint_qp_includes_restoring_already_nonzero_source_residual',
      np.linalg.norm(K@u-desired[:2])<1e-12)
Gi = np.linalg.inv(G)
S = Gi-Gi@K.T@np.linalg.solve(K@Gi@K.T, K@Gi)
restore = Gi@K.T@np.linalg.solve(K@Gi@K.T, desired[:2])
g_task = C[0]; kappa = float(g_task@S@g_task)
radius = float(np.sqrt(restore@G@restore)+.8)
remaining_radius = float(np.sqrt(radius**2-restore@G@restore))
maximum_progress = float(g_task@restore+remaining_radius*np.sqrt(kappa))
maximizer = restore+remaining_radius/np.sqrt(kappa)*(S@g_task)
max_numeric = minimize(lambda z:-g_task@z, restore, jac=lambda z:-g_task,
                       constraints=[{'type':'eq', 'fun':lambda z:K@z-desired[:2], 'jac':lambda z:K},
                                    {'type':'ineq', 'fun':lambda z:radius**2-z@G@z,
                                     'jac':lambda z:-2*G@z}],
                       method='SLSQP', options={'ftol':1e-12, 'maxiter':300})
check('task_keep_capacity_bound_matches_independent_optimization',
      max_numeric.success and abs(g_task@max_numeric.x-maximum_progress)<2e-8
      and np.linalg.norm(K@maximizer-desired[:2])<1e-12,
      progress_error=float(abs(g_task@max_numeric.x-maximum_progress)))
check('task_gradient_fully_in_keep_span_has_zero_compatible_capacity',
      abs(K[0]@S@K[0])<1e-12)

# Necessity depends on the joint completion set, not individual attribution.
grid = np.linspace(0., 1., 1001)
joint = np.column_stack([grid, 1-grid])
cost = np.sum(joint**2, axis=1)
optimum = float(cost.min())
check('completion_minimum_change_uses_joint_not_coordinatewise_decisions',
      abs(optimum-.5)<1e-12 and np.allclose(joint[np.argmin(cost)], [.5, .5]))
check('zero_excluded_from_both_phase_modes_at_strict_minimum_budget',
      np.allclose(joint[cost<=optimum+1e-12].min(axis=0), [.5, .5]))
check('each_mode_can_be_ablated_with_a_more_expensive_alternative',
      abs(cost[0]-1)<1e-12 and abs(cost[-1]-1)<1e-12)
check('combining_individually_possible_deletions_can_destroy_goal',
      joint[:, 0].min()==0 and joint[:, 1].min()==0 and np.sum([0., 0.])<1)
check('zeroing_a_branch_coefficient_is_not_zeroing_its_actual_effect',
      np.array_equal(0*np.array([1., 0.])+1*np.array([1., 0.]), np.array([1., 0.])))

# Finite synergy: both task derivatives vanish initially, but both changes matter.
points = np.array(list(itertools.product([0., .5, 1.], repeat=2)))
task = np.prod(points, axis=1)
feasible = points[task>=1]
synergy = lambda state: state[0]*state[1]
synergy_gradient = np.array([(synergy(eps*e)-synergy(-eps*e))/(2*eps) for e in np.eye(2)])
check('first_order_task_progress_can_miss_required_joint_change',
      np.linalg.norm(synergy_gradient)==0 and np.array_equal(feasible, [[1., 1.]]))
check('finite_group_ablation_detects_joint_necessity',
      not np.any(task[points[:, 0]==0]>=1) and not np.any(task[points[:, 1]==0]>=1))
# Preparation followed by consummation: no first-step task increase is possible.
prepare = lambda state, control: np.array([state[0], state[1]+control])
consume = lambda state, control: np.array([state[0]+state[1]*control, 0.])
s0 = np.array([0., 0.]); s1 = prepare(s0, 1.); s2 = consume(s1, 1.)
check('short_horizon_can_retain_a_task_neutral_preparatory_change',
      s1[0]==s0[0] and s2[0]==1 and s2[1]==0)
check('removing_preparation_prevents_goal_in_same_horizon', consume(prepare(s0, 0.), 1.)[0]<1)

# Shared noise is a paired design; variance reduction requires positive covariance.
var_s = 101.; var_a = 101.; cov = 100.
check('paired_noise_difference_variance_identity', var_s+var_a-2*cov==2.)
check('common_random_numbers_need_not_reduce_variance',
      1.+1.-2*(-1.) > 1.+1.)

# Phase velocities add, but finite phase increments of separate candidates do not.
z = 1.+0j; dz1 = 1j; dz2 = 1j
phase_rate = lambda dz: float(np.imag(np.conj(z)*dz)/abs(z)**2)
check('instantaneous_phase_response_respects_branch_sum',
      phase_rate(dz1+dz2)==phase_rate(dz1)+phase_rate(dz2))
finite_joint = float(np.angle(z+dz1+dz2)-np.angle(z))
finite_sum = float(np.angle(z+dz1)+np.angle(z+dz2)-2*np.angle(z))
check('finite_phase_changes_cannot_be_added_branchwise',
      abs(finite_joint-finite_sum)>.4, nonadditivity=float(abs(finite_joint-finite_sum)))

path = Path(__file__)
result = dict(scope='Exact identities, independent density derivatives and convex solves, and finite counterexamples only; no pretrained image experiments or natural-language binding validation.',
              seed=20261025, passed=sum(r['passed'] for r in rows), total=len(rows),
              source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checks=rows)
path.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed', 'total', 'source_sha256']}, ensure_ascii=False))
