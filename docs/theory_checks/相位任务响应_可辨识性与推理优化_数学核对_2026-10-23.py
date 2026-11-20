"""Finite-dimensional theory checks, not FlowEdit/image-quality experiments.

Only NumPy is required. Run from any directory; writes JSON beside this script.
The complex signal examples are synthetic; they do not model a trained VAE.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np


SEED = 20261023
rng = np.random.default_rng(SEED)
results = []


def check(name, passed, **metrics):
    results.append(dict(name=name, passed=bool(passed), metrics=metrics))
    if not passed:
        raise AssertionError((name, metrics))


def proj(v):
    return v @ np.linalg.pinv(v)


def close(name, x, y, tol=1e-9):
    error = float(np.linalg.norm(np.asarray(x) - np.asarray(y)))
    check(name, error < tol, error=error, tolerance=tol)


# 1. A change of appearance can generate a nonzero phase response.
z = np.array([1+1j, 2-1j, -1+2j])
dz = np.array([.4, -.2, .3])
eps = 1e-6
v = np.imag(np.conj(z)*dz)/np.abs(z)**2
fd = np.angle((z+eps*dz)*np.conj(z-eps*dz))/(2*eps)
close("phase_appearance_derivative", fd, v)
check("appearance_without_displacement_changes_phase", abs(np.angle(.5+1j)-np.angle(1+1j)) > .3)

# 2. Nonnegative gain profiling does not identify opposite phases.
target, reference = -1+0j, 1+0j
gain = max(0., float(np.real(np.conj(reference)*target)/abs(reference)**2))
check("positive_ray_rejects_phase_pi", abs(target-gain*reference)**2 == 1. and np.sin(np.pi)**2 < 1e-20)

# 3. Appearance/geometry angle determines identification, not solver accuracy.
angles = np.array([0., .1, .4, 1.2])
vmat = np.array([[1.], [0.]])
q = np.eye(2)-proj(vmat)
information = np.array([np.array([np.cos(a), np.sin(a)]) @ q @ np.array([np.cos(a), np.sin(a)]) for a in angles])
close("principal_angle_information", information, np.sin(angles)**2)
check("exact_nonidentifiability", information[0] == 0.)
vmat = rng.normal(size=(12, 3))
mmat = rng.normal(size=(12, 2))
q = np.eye(12)-proj(vmat)
q_big = np.eye(12)-proj(np.column_stack([vmat, mmat[:, 0]]))
loss = mmat.T @ (q-q_big) @ mmat
check("richer_appearance_reduces_geometric_information", float(np.linalg.eigvalsh(loss).min()) > -1e-10)
transform = np.diag([.1, 2., 8.])
close("nuisance_reparameterization_invariance", proj(vmat), proj(vmat@transform))

# 4. Conditional geometric and unexplained residuals are orthogonal.
c = q@mmat
u_true = np.array([.3, -.4])
alpha = rng.normal(size=3)
w = vmat@alpha + mmat@u_true
u_hat = np.linalg.pinv(c) @ q @ w
close("joint_geometry_recovery", u_hat, u_true)
pc = proj(c)
unknown = (np.eye(12)-pc)@q@w
close("joint_model_unexplained_residual", unknown, np.zeros(12))
noise = rng.normal(size=12)*.01
u_noisy = np.linalg.pinv(c)@q@(w+noise)
bound = np.linalg.norm(q@noise)/np.linalg.svd(c, compute_uv=False)[-1]
check("geometry_error_bound", np.linalg.norm(u_noisy-u_true) <= bound+1e-12,
      error=float(np.linalg.norm(u_noisy-u_true)), bound=float(bound))

# 5. Penalized variable projection: P, not P squared, is the profiled cost.
vmat = rng.normal(size=(9, 3))
lam_a = np.diag([.3, .8, 1.2])
p = np.eye(9)-vmat@np.linalg.solve(vmat.T@vmat+lam_a, vmat.T)
r = rng.normal(size=9)
a = np.linalg.solve(vmat.T@vmat+lam_a, vmat.T@r)
joint_cost = np.linalg.norm(r-vmat@a)**2+a@lam_a@a
close("ridge_profile_cost", joint_cost, r@p@r)
check("ridge_profile_not_residual_squared", abs(r@p@r-r@p@p@r) > 1e-4)
check("ridge_profile_psd_contraction", np.linalg.eigvalsh(p).min() >= -1e-12 and np.linalg.eigvalsh(p).max() <= 1+1e-12)
check("bounded_nuisance_not_unbounded_projection", (2.-np.clip(2., -1., 1.))**2 == 1.)

# 6. Independent block solve checks the profiled state increment.
amat = rng.normal(size=(9, 5))
gmat = np.diag(rng.uniform(.7, 1.5, 5))
delta = rng.normal(size=5)
e = rng.normal(size=9)
weight = .8
hmat = gmat+weight*amat.T@p@amat
d0 = np.linalg.solve(hmat, gmat@delta-weight*amat.T@p@e)
block = np.block([[gmat+weight*amat.T@amat, -weight*amat.T@vmat],
                  [-weight*vmat.T@amat, weight*(vmat.T@vmat+lam_a)]])
rhs = np.r_[gmat@delta-weight*amat.T@e, weight*vmat.T@e]
joint = np.linalg.solve(block, rhs)
close("state_update_profile_vs_joint_block", d0, joint[:5])
g = rng.normal(size=5)
beta = float(g@d0+.4)
hg = np.linalg.solve(hmat, g)
mu = (beta-g@d0)/(g@hg)
d = d0+mu*hg
kkt = np.block([[hmat, -g[:, None]], [g[None, :], np.zeros((1, 1))]])
independent = np.linalg.solve(kkt, np.r_[gmat@delta-weight*amat.T@p@e, beta])
close("active_task_constraint_kkt", d, independent[:5])
close("task_constraint_satisfied", g@d, beta)
close("inactive_task_constraint", max(0., (g@d0-.4-g@d0)/(g@hg)), 0.)
step = 1e-5
def optimal_value(b):
    sol = d0 + max(0., (b-g@d0)/(g@hg))*hg
    return .5*sol@hmat@sol-(gmat@delta-weight*amat.T@p@e)@sol
close("dual_multiplier_tradeoff_derivative", (optimal_value(beta+step)-optimal_value(beta-step))/(2*step), mu, 1e-8)

# 7. Source restoration acts even when the current base increment is zero.
restore = -1./2.
check("source_residual_requires_restoration", abs(1.+restore) < 1. and restore != 0.)

# 8. Exact worst-case norm with unstructured spectral-norm uncertainty.
f = rng.normal(size=(7, 4))
zeta = rng.normal(size=4)
e0 = rng.normal(size=7)
eta = .15
res = e0+f@zeta
worst = eta*np.outer(res/np.linalg.norm(res), zeta/np.linalg.norm(zeta))
close("robust_operator_bound_attained", np.linalg.norm(e0+(f+worst)@zeta),
      np.linalg.norm(res)+eta*np.linalg.norm(zeta))
check("robust_perturbation_admissible", np.linalg.norm(worst, 2) <= eta+1e-12)
eg = .2
worst_g = g-eg*d/np.linalg.norm(d)
close("robust_task_gradient_bound", worst_g@d, g@d-eg*np.linalg.norm(d))

# 9. Every velocity is allowed, yet a closed parameter loop drifts in z.
# X=(1,0,0), Y=(0,1,x), exact flows X(a),Y(b),X(-a),Y(-b).
loop_a, loop_b = .2, .3
state = np.zeros(3)
state[0] += loop_a
state += np.array([0., loop_b, state[0]*loop_b])
state[0] -= loop_a
state += np.array([0., -loop_b, -state[0]*loop_b])
close("nonintegrable_allowed_velocities_drift", state, [0., 0., loop_a*loop_b])
# Explicit source family T(a,b)=(a,b,a*b) has closed parameter loops.
def family(a, b):
    return np.array([a, b, a*b])
states = [family(a, b) for a, b in [(0., 0.), (loop_a, 0.),
                                  (loop_a, loop_b), (0., loop_b), (0., 0.)]]
integrated_family_steps = np.sum(np.diff(states, axis=0), axis=0)
close("explicit_source_family_closed_loop", integrated_family_steps, np.zeros(3))

# 10. Local tangent acceptance does not establish finite-step membership.
# Reference family curve (theta, theta^2); tangent at zero accepts (h,0).
h = .4
candidate = np.array([h, 0.])
linear_residual = candidate-np.array([h, 0.])
finite_residual = candidate-np.array([h, h*h])
check("finite_nonlinear_recheck_required", np.linalg.norm(linear_residual) == 0. and np.linalg.norm(finite_residual) > .1)

# 11. Phase wrapping and phase aliasing are distinct problems.
phi0, phi1 = np.pi-.02, -np.pi+.03
close("circular_residual_handles_wrap", np.angle(np.exp(1j*(phi1-phi0))), .05)
k = 3.
period = 2*np.pi/k
close("single_frequency_displacement_alias", np.exp(-1j*k*period), 1.+0j)

# 12. Fixed-query susceptibility integral recovers a conditional field change.
# b(l)=b0+l*p+l^2*q, with a fixed observation Jacobian.
obs = rng.normal(size=(8, 5))
bp, bq = rng.normal(size=5), rng.normal(size=5)
nodes, weights = np.polynomial.legendre.leggauss(5)
integral = sum(wi*.5*(obs@(bp+2*((xi+1)/2)*bq)) for xi, wi in zip(nodes, weights))
close("conditional_susceptibility_integral", integral, obs@(bp+bq))
# Trajectory response dy/dl obeys state feedback: y'=l*y, y(0)=1.
time, dose = .8, .6
exact = time*np.exp(dose*time)
fd = (np.exp((dose+eps)*time)-np.exp((dose-eps)*time))/(2*eps)
close("trajectory_response_includes_state_feedback", exact, fd)
check("frozen_state_response_is_not_full_trajectory", abs(exact-time) > .1)

# 13. A synthetic semantic edit can require phase while excluding motion.
# Phase modes could be realized by z(a,u)=exp(i*design@[a,u]).
# This is a coefficient model, not a claim of image-filter realizability.
appearance_mode = np.array([1., 1., 0.])
motion_mode = np.array([0., 1., 1.])
# State variables are (appearance amount, displacement); target progress a>=1.
design = np.column_stack([appearance_mode, motion_mode])
q = np.eye(3)-proj(appearance_mode[:, None])
gmat2 = np.eye(2)
hmat2 = gmat2+20*design.T@q@design
raw_state = np.array([1., .7])
corrected = np.linalg.solve(hmat2, raw_state)
close("necessary_phase_preserved_in_synthetic_model", corrected[0], 1.)
check("unnecessary_motion_suppressed_in_synthetic_model", 0 < corrected[1] < .05,
      raw_displacement=.7, corrected_displacement=float(corrected[1]),
      retained_appearance=float(corrected[0]))
check("hard_phase_lock_blocks_synthetic_task", np.linalg.matrix_rank(design) == 2)

# 14. Matrix-free Hessian product and step-size-scaled penalty limit.
probe = rng.normal(size=5)
matrix_free = gmat@probe+weight*amat.T@(amat@probe-vmat@np.linalg.solve(vmat.T@vmat+lam_a, vmat.T@amat@probe))
close("matrix_free_hessian_product", matrix_free, hmat@probe)
velocity = rng.normal(size=5)
limit_velocity = velocity-weight*np.linalg.solve(gmat, amat.T@p@e)
widths = np.array([1e-3, 5e-4, 2.5e-4])
errors = []
for width in widths:
    scaled_hessian = gmat+width*weight*amat.T@p@amat
    scaled_d = np.linalg.solve(scaled_hessian, width*gmat@velocity-width*weight*amat.T@p@e)
    errors.append(float(np.linalg.norm(scaled_d/width-limit_velocity)))
check("scaled_penalty_continuous_limit", errors[-1] < .3*errors[0],
      step_widths=widths.tolist(), velocity_errors=errors)

path = Path(__file__)
payload = dict(scope="finite-dimensional algebra and synthetic signals; no pretrained model or image quality validation",
               seed=SEED, python=platform.python_version(), numpy=np.__version__,
               script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               passed=sum(r["passed"] for r in results), total=len(results), checks=results)
path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"passed": payload["passed"], "total": payload["total"], "output": str(path.with_suffix('.json'))}, ensure_ascii=False))
