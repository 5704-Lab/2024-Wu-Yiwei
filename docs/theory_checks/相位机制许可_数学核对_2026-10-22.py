"""Algebra, signal identities, and counterexamples only; no image-model run.

The checks do not establish semantic identifiability or image-editing quality.
Run from any directory. The JSON is written next to this file.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np


SEED = 20261022
rng = np.random.default_rng(SEED)
results = []


def check(name, passed, **metrics):
    results.append({"name": name, "passed": bool(passed), "metrics": metrics})
    if not passed:
        raise AssertionError((name, metrics))


def projector(design):
    return design @ np.linalg.pinv(design)


def unit(z):
    return z / np.abs(z)


# Exact instantaneous phase/amplitude derivatives; finite differences only
# approximate them. This complex linear map does not model a VAE decoder.
z = rng.uniform(0.8, 2.0, 20) * np.exp(1j * rng.uniform(-np.pi, np.pi, 20))
linear = rng.normal(size=(20, 9)) + 1j * rng.normal(size=(20, 9))
direction = rng.normal(size=9)
dz = linear @ direction
eps = 1e-7
phase_derivative = np.imag(np.conj(z) * dz) / np.abs(z) ** 2
phase_fd = np.angle((z + eps * dz) * np.conj(z - eps * dz)) / (2 * eps)
phase_error = np.max(np.abs(phase_fd - phase_derivative))
check("phase_velocity_identity", phase_error < 1e-7, max_error=float(phase_error))
amplitude_derivative = np.real(np.conj(z) * dz) / np.abs(z)
amplitude_fd = (np.abs(z + eps * dz) - np.abs(z - eps * dz)) / (2 * eps)
amplitude_error = np.max(np.abs(amplitude_fd - amplitude_derivative))
check("amplitude_velocity_identity", amplitude_error < 1e-7,
      max_error=float(amplitude_error))
radial = rng.normal(size=len(z)) * z
radial_phase = np.max(np.abs(np.imag(np.conj(z) * radial) / np.abs(z) ** 2))
check("radial_velocity_has_zero_phase_velocity", radial_phase < 1e-12,
      max_phase_velocity=float(radial_phase))

# Same spatial structures, different mixture strengths: necessary nonzero phase.
mixture_change = abs(np.angle((0.5 + 1j) / (1.0 + 1j)))
check("static_mixture_reweighting_can_change_phase", mixture_change > 0.3,
      radians=float(mixture_change), degrees=float(np.rad2deg(mixture_change)))

# Complex disk -> arcsin bound. Include tangent points to check tightness.
radius_ratio = rng.uniform(0.001, 0.95, 10000)
random_error = (radius_ratio * np.sqrt(rng.uniform(size=10000))
                * np.exp(1j * rng.uniform(-np.pi, np.pi, 10000)))
angle = np.abs(np.angle(1.0 + random_error))
bound = np.arcsin(radius_ratio)
disk_violation = np.max(angle - bound)
tangent = ((1 - radius_ratio ** 2)
           + 1j * radius_ratio * np.sqrt(1 - radius_ratio ** 2))
tangent_gap = np.max(np.abs(np.angle(tangent) - bound))
check("complex_disk_phase_bound", disk_violation < 1e-12 and tangent_gap < 1e-12,
      max_violation=float(disk_violation), tangent_gap=float(tangent_gap))

# Convex sector constraints, with an explicit positive projection floor.
phi0 = 0.8
tau = 0.45
floor = 0.08
candidate = rng.normal(size=10000) + 1j * rng.normal(size=10000)
rotated = candidate * np.exp(-1j * phi0)
sector = ((np.abs(rotated.imag) <= np.tan(tau) * rotated.real)
          & (rotated.real >= floor))
angular = ((np.abs(np.angle(rotated)) <= tau) & (rotated.real >= floor))
check("convex_sector_matches_angular_constraint", np.array_equal(sector, angular),
      number_checked=len(candidate), disagreements=int(np.sum(sector != angular)))

# A zero-DC Fourier example verifies affine invariance, not local-filter quality.
signal = rng.normal(size=64)
z0 = np.fft.fft(signal)[1:32]
zg = np.fft.fft(0.25 * signal + 3.0)[1:32]
gain_phase_error = np.max(np.abs(unit(z0) - unit(zg)))
check("positive_affine_change_preserves_non_dc_phase", gain_phase_error < 1e-12,
      max_error=float(gain_phase_error))
smooth_unit = lambda a: a / np.sqrt(np.abs(a) ** 2 + 0.2 ** 2)
smooth_gap = np.linalg.norm(smooth_unit(np.array([1.0 + 1j]))
                            - smooth_unit(np.array([0.2 + 0.2j])))
check("epsilon_normalization_loses_exact_gain_invariance", smooth_gap > 0.1,
      normalized_response_gap=float(smooth_gap))

# Weighted nuisance elimination, followed by conditional geometric information.
n, nuisance_dim, geom_dim = 18, 4, 2
v = rng.normal(size=(n, nuisance_dim))
m = rng.normal(size=(n, geom_dim))
ws = np.diag(np.sqrt(rng.uniform(0.3, 2.0, n)))
vw = ws @ v
mw = ws @ m
q = np.eye(n) - projector(vw)
omega = rng.normal(size=n)
alpha = np.linalg.lstsq(vw, ws @ omega, rcond=None)[0]
ls_energy = np.linalg.norm(ws @ (omega - v @ alpha)) ** 2
profile_energy = np.linalg.norm(q @ ws @ omega) ** 2
profile_gap = abs(ls_energy - profile_energy)
check("weighted_phase_nuisance_elimination", profile_gap < 1e-10,
      absolute_error=float(profile_gap))
information = mw.T @ q @ mw
eigenvalues = np.linalg.eigvalsh(information)
check("conditional_geometry_information_psd", eigenvalues.min() > -1e-10,
      eigenvalues=eigenvalues.tolist())
true_u = np.array([0.16, -0.23])
true_a = rng.normal(size=nuisance_dim)
observed = v @ true_a + m @ true_u
u_hat = np.linalg.solve(information, mw.T @ q @ ws @ observed)
u_error = np.linalg.norm(u_hat - true_u)
check("identifiable_geometry_recovery_after_nuisance", u_error < 1e-10,
      displacement_error=float(u_error))
conflicting_m = np.column_stack((v[:, 0], rng.normal(size=n)))
conflicting_info = (ws @ conflicting_m).T @ q @ (ws @ conflicting_m)
conflicting_eigenvalues = np.linalg.eigvalsh(conflicting_info)
null_residual = np.linalg.norm(q @ ws @ conflicting_m[:, 0])
check("appearance_geometry_overlap_is_unidentifiable",
      abs(conflicting_eigenvalues[0]) < 1e-10 and null_residual < 1e-10,
      minimum_eigenvalue=float(conflicting_eigenvalues[0]),
      unobservable_direction_residual=float(null_residual))

# Graph relations cannot add information that was removed by construction.
phases = rng.uniform(-np.pi, np.pi, 12)
u = np.exp(1j * phases)
edges = u[:, None] * np.conj(u[None, :])
cycle_error = 0.0
for i, j, k in rng.integers(0, len(u), size=(100, 3)):
    cycle_error = max(cycle_error, abs(edges[i, j] * edges[j, k] * edges[k, i] - 1))
check("node_derived_phase_cycles_are_tautologies", cycle_error < 1e-12,
      max_cycle_error=float(cycle_error))
gauged = np.exp(0.7j) * u
gauge_error = np.max(np.abs(edges - gauged[:, None] * np.conj(gauged[None, :])))
absolute_change = np.linalg.norm(gauged - u)
check("relative_phase_graph_hides_common_phase", gauge_error < 1e-12 and absolute_change > 1,
      relative_graph_error=float(gauge_error), absolute_change=float(absolute_change))
positions = np.linspace(0, 5, 15)
carrier = 1.7
shift = 0.3
plane = np.exp(1j * carrier * positions)
shifted = np.exp(1j * carrier * (positions - shift))
plane_graph_error = np.max(np.abs(
    plane[:, None] * np.conj(plane[None, :])
    - shifted[:, None] * np.conj(shifted[None, :])))
check("single_carrier_translation_is_a_graph_gauge", plane_graph_error < 1e-12,
      relative_graph_error=float(plane_graph_error),
      anchored_phase_shift=float(abs(np.angle(shifted[0] / plane[0]))))

# Transport-reaction decomposition is not identifiable without restrictions.
gradient = rng.normal(size=(10, 2))
image_velocity = rng.normal(size=10)
velocity1 = np.tile([0.1, -0.2], (10, 1))
velocity2 = np.tile([-0.3, 0.4], (10, 1))
reaction1 = image_velocity + np.sum(gradient * velocity1, axis=1)
reaction2 = image_velocity + np.sum(gradient * velocity2, axis=1)
reconstruction1 = reaction1 - np.sum(gradient * velocity1, axis=1)
reconstruction2 = reaction2 - np.sum(gradient * velocity2, axis=1)
reaction_error = max(np.linalg.norm(reconstruction1 - image_velocity),
                     np.linalg.norm(reconstruction2 - image_velocity))
check("unrestricted_reaction_absorbs_any_transport", reaction_error < 1e-12,
      reconstruction_error=float(reaction_error),
      reaction_difference=float(np.linalg.norm(reaction1 - reaction2)))
rigid_gradient = np.array([[0.0, -0.4], [0.4, 0.0]])
strain = (rigid_gradient + rigid_gradient.T) / 2
check("linear_strain_hides_infinitesimal_rigid_rotation", np.linalg.norm(strain) == 0,
      strain_norm=float(np.linalg.norm(strain)),
      rotation_gradient_norm=float(np.linalg.norm(rigid_gradient)))

# Minimum necessary phase plus a positive definite metric: independent KKT solve.
a = rng.normal(size=(3, 6))
g = rng.normal(size=6)
beta = 0.35
hessian = a.T @ a + 0.2 * np.eye(6)
inverse_g = np.linalg.solve(hessian, g)
closed = beta * inverse_g / (g @ inverse_g)
kkt = np.block([[hessian, -g[:, None]], [g[None, :], np.zeros((1, 1))]])
independent = np.linalg.solve(kkt, np.r_[np.zeros(6), beta])[:6]
kkt_gap = np.linalg.norm(closed - independent)
check("minimum_necessary_phase_closed_form", kkt_gap < 1e-10,
      kkt_gap=float(kkt_gap), task_progress=float(g @ closed))

# Trust-region feasibility in the exact zero-phase nullspace.
_, singular, vh = np.linalg.svd(a, full_matrices=True)
rank = np.sum(singular > 1e-12)
null_basis = vh[rank:].T
null_g = null_basis @ (null_basis.T @ g)
rho = 0.4
maximizer = rho * null_g / np.linalg.norm(null_g)
formula_progress = rho * np.linalg.norm(null_g)
check("zero_phase_task_feasibility_bound",
      np.linalg.norm(a @ maximizer) < 1e-10
      and abs(g @ maximizer - formula_progress) < 1e-10,
      maximum_progress=float(formula_progress),
      phase_residual=float(np.linalg.norm(a @ maximizer)))
conflicting_g = a.T @ rng.normal(size=a.shape[0])
conflicting_null_g = np.linalg.norm(null_basis.T @ conflicting_g)
check("pure_phase_task_has_no_zero_phase_progress", conflicting_null_g < 1e-10,
      nullspace_task_component=float(conflicting_null_g))

# Singular phase metric: task gradient lies in the range of that metric.
p = a.T @ a
pinv_p = np.linalg.pinv(p)
range_g = a.T @ rng.normal(size=a.shape[0])
minimizer = beta * pinv_p @ range_g / (range_g @ pinv_p @ range_g)
lower_bound = beta ** 2 / (range_g @ pinv_p @ range_g)
actual_cost = np.linalg.norm(a @ minimizer) ** 2
check("singular_metric_minimum_phase_lower_bound",
      abs(actual_cost - lower_bound) < 1e-10 and abs(range_g @ minimizer - beta) < 1e-10,
      actual_cost=float(actual_cost), lower_bound=float(lower_bound))

# At the source, squared-error gradient constraints are vacuous for a finite step.
origin_gradient = np.zeros(2)
bad_step = np.array([0.0, 0.3])
bad_operator = np.array([[0.0, 1.0]])
loss_after_step = 0.5 * np.linalg.norm(bad_operator @ bad_step) ** 2
check("zero_source_gradient_does_not_protect_a_finite_step",
      origin_gradient @ bad_step == 0 and loss_after_step > 0.04,
      linearized_loss_change=float(origin_gradient @ bad_step),
      actual_loss_change=float(loss_after_step))
tube_bound = 0.1
check("residual_tube_rejects_that_bad_step",
      np.linalg.norm(bad_operator @ bad_step) > tube_bound,
      residual_norm=float(np.linalg.norm(bad_operator @ bad_step)),
      allowed_norm=tube_bound)

# For arg(1+ix)=atan(x), |second derivative| <= 1 globally. A concrete nonlinear
# Taylor check, not a decoder-wide certificate.
x0 = rng.uniform(-1, 1, 10000)
step = rng.uniform(-0.5, 0.5, 10000)
phase_remainder = np.abs(np.arctan(x0 + step) - np.arctan(x0)
                         - step / (1 + x0 ** 2))
taylor_violation = np.max(phase_remainder - 0.5 * step ** 2)
check("smooth_phase_finite_step_remainder_example", taylor_violation < 1e-12,
      max_upper_bound_violation=float(taylor_violation))

# Proper complex noise: E|noise|^2=sigma_c^2; phase variance at high SNR.
amplitude = 1.3
sigma_complex = 0.012
noise = sigma_complex / np.sqrt(2) * (
    rng.normal(size=200000) + 1j * rng.normal(size=200000))
empirical_variance = np.var(np.angle(amplitude + noise))
predicted_variance = sigma_complex ** 2 / (2 * amplitude ** 2)
variance_ratio = empirical_variance / predicted_variance
check("high_snr_phase_variance_factor_two", abs(variance_ratio - 1) < 0.03,
      empirical_variance=float(empirical_variance),
      predicted_variance=float(predicted_variance), ratio=float(variance_ratio))

# Independent changes of redundant coefficients need not be realizable.
analysis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
consistent = analysis @ np.array([1.0, 2.0])
modified = consistent.astype(complex)
modified[0] *= np.exp(0.6j)
consistency_error = np.linalg.norm((np.eye(3) - projector(analysis)) @ modified)
check("independent_phase_edits_can_leave_analysis_range", consistency_error > 0.1,
      distance_from_consistent_range=float(consistency_error))

per_step = 0.01
steps = 50
check("small_increment_bound_does_not_bound_cumulative_drift",
      steps * per_step > 20 * per_step,
      one_step_phase=per_step, accumulated_phase=steps * per_step)

payload = {
    "date": "2026-10-22",
    "scope": "finite_dimensional_algebra_signal_identities_and_counterexamples_only",
    "no_model_experiments": True,
    "seed": SEED,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "check_groups": len(results),
    "all_passed": all(row["passed"] for row in results),
    "results": results,
    "limitations": [
        "No VAE, DiT, diffusion, flow, or real image is evaluated.",
        "Signal identities do not identify semantic necessity.",
        "Nuisance and motion recovery use a correctly specified synthetic model.",
        "The nonlinear Taylor example is not a neural-network Lipschitz certificate.",
        "No final image quality or runtime improvement is established.",
    ],
}
output = Path(__file__).with_suffix(".json")
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "check_groups": payload["check_groups"], "all_passed": payload["all_passed"],
    "phase_derivative_error": float(phase_error), "kkt_gap": float(kkt_gap),
    "output": str(output),
}, ensure_ascii=False))
