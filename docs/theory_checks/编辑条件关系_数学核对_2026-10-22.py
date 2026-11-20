"""Finite-dimensional checks; no image models, weights, or generated images.

These checks validate algebra and known blind spots. They do not validate
semantic disentanglement or the quality of the proposed image-editing method.
Run from any directory; the JSON is written beside this file.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np


rng = np.random.default_rng(20261022)
results = []


def check(name, passed, **metrics):
    results.append({"name": name, "passed": bool(passed), "metrics": metrics})
    if not passed:
        raise AssertionError((name, metrics))


def projection(design):
    return design @ np.linalg.pinv(design)


# Weighted variable projection, compared to an independent least-squares solve.
n = 17
x = rng.normal(size=(n, 3))
b = np.column_stack((np.ones(n), x))
w = np.diag(rng.uniform(0.5, 2.0, size=n))
ws = np.diag(np.sqrt(np.diag(w)))
p = projection(ws @ b)
q = np.eye(n) - p
l = ws @ q @ ws
y = rng.normal(size=(n, 3))
coef = np.linalg.lstsq(ws @ b, ws @ y, rcond=None)[0]
direct = np.linalg.norm(ws @ (y - b @ coef)) ** 2
profile = np.linalg.norm(q @ ws @ y) ** 2
check(
    "weighted_profile_equals_least_squares",
    abs(direct - profile) < 1e-10,
    absolute_error=float(abs(direct - profile)),
)
check(
    "source_conditioned_operator_psd",
    np.linalg.eigvalsh(l).min() > -1e-10,
    minimum_eigenvalue=float(np.linalg.eigvalsh(l).min()),
)
affine = b @ rng.normal(size=(4, 3))
err = np.linalg.norm(q @ ws @ affine)
check("affine_color_is_in_nullspace", err < 1e-10, residual=float(err))
constant = np.ones((n, 1)) @ np.array([[0.1, 0.2, 0.3]])
err = np.linalg.norm(q @ ws @ constant)
check(
    "constant_image_is_a_real_blind_spot",
    err < 1e-10,
    residual=float(err),
)

# A spatially shifted linear ramp is also a photometric bias.
coords = np.linspace(-1, 1, n)
ramp = np.column_stack((np.ones(n), coords))
residualizer = np.eye(n) - projection(ramp)
shifted_ramp = coords - 0.2
err = np.linalg.norm(residualizer @ shifted_ramp)
check("ramp_translation_blind_spot", err < 1e-10, residual=float(err))
tiny_design = np.column_stack((np.ones(3), rng.normal(size=(3, 3))))
err = np.linalg.norm(np.eye(3) - projection(tiny_design))
check("too_small_window_is_uninformative", err < 1e-10, residual=float(err))

# Fourier coefficients away from DC demonstrate the same affine phase identity
# as any zero-DC complex linear filter. This is NOT a local image benchmark.
signal = rng.normal(size=64)
z = np.fft.fft(signal)[1:32]
zp = np.fft.fft(0.3 * signal + 4.0)[1:32]
unit = lambda a: a / np.abs(a)
err = np.max(np.abs(unit(z) - unit(zp)))
check("positive_gain_offset_phase_invariance", err < 1e-10, max_error=float(err))
zt = np.fft.fft(np.roll(signal, 3))[1:32]
phase_change = np.mean(np.abs(unit(z) - unit(zt)) ** 2)
check(
    "anchored_phase_detects_this_translation",
    phase_change > 0.5,
    mean_squared_phase_change=float(phase_change),
)
periodic = np.cos(2 * np.pi * np.arange(64) / 16)
err = np.linalg.norm(periodic - np.roll(periodic, 16))
check(
    "periodic_texture_translation_ambiguity",
    err < 1e-10,
    signal_difference=float(err),
)
smooth_unit = lambda a: a / np.sqrt(np.abs(a) ** 2 + 1e-8)
flat_phase_loss = np.mean(np.abs(smooth_unit(z)) ** 2)
check(
    "fixed_source_phase_penalizes_signal_erasure",
    flat_phase_loss > 0.99,
    erased_signal_loss=float(flat_phase_loss),
)

# Residual covariance: OLS residualizer is idempotent, ridge is not.
m = 240
common = rng.normal(size=(m, 1))
common -= common.mean(axis=0)
hi = 8 * common + rng.normal(size=(m, 1))
hj = 8 * common + rng.normal(size=(m, 1))
hi -= hi.mean(axis=0)
hj -= hj.mean(axis=0)
r = np.eye(m) - projection(common)
raw_corr = float(np.corrcoef(hi[:, 0], hj[:, 0])[0, 1])
res_corr = float(np.corrcoef((r @ hi)[:, 0], (r @ hj)[:, 0])[0, 1])
check(
    "specified_common_factor_can_explain_raw_correlation",
    raw_corr > 0.95 and abs(res_corr) < 0.2,
    raw_correlation=raw_corr,
    residual_correlation=res_corr,
)
err = np.max(np.abs(hi.T @ r @ hj - (r @ hi).T @ (r @ hj)))
check("ols_residual_gram_identity", err < 1e-10, max_error=float(err))
ridge = np.eye(m) - common @ np.linalg.solve(
    common.T @ common + 20 * np.eye(1), common.T
)
correct = (ridge @ hi).T @ (ridge @ hj)
shortcut = hi.T @ ridge @ hj
gap = float(np.abs(correct - shortcut).item())
check("ridge_requires_r_transpose_r", gap > 1.0, shortcut_error=gap)

# Population Schur complement, with a positive-definite covariance.
g = rng.normal(size=(7, 7))
sigma = g @ g.T + np.eye(7)
schur = sigma[:4, :4] - sigma[:4, 4:] @ np.linalg.solve(
    sigma[4:, 4:], sigma[4:, :4]
)
check(
    "population_schur_complement_psd",
    np.linalg.eigvalsh(schur).min() > 0,
    minimum_eigenvalue=float(np.linalg.eigvalsh(schur).min()),
)

# Solve the correction using two independent constructions:
# a full KKT system and a nullspace-reduced SPD system.
worst_kkt_gap = 0.0
worst_task_error = 0.0
worst_bound_violation = -np.inf
for _ in range(12):
    dim = 11
    a = rng.normal(size=(6, dim))
    cmat = rng.normal(size=(3, dim))
    e = rng.normal(size=6)
    gg = rng.normal(size=(dim, dim))
    metric = gg @ gg.T + np.eye(dim)
    weight = np.diag(rng.uniform(0.2, 2.0, size=6))
    lam = float(rng.uniform(0.1, 3.0))
    hessian = metric + lam * a.T @ weight @ a
    kkt = np.block(
        [[hessian, cmat.T], [cmat, np.zeros((3, 3))]]
    )
    rhs = np.concatenate((-lam * a.T @ weight @ e, np.zeros(3)))
    c_kkt = np.linalg.solve(kkt, rhs)[:dim]
    _, _, vt = np.linalg.svd(cmat, full_matrices=True)
    z0 = vt[3:].T
    chol = np.linalg.cholesky(z0.T @ metric @ z0)
    z_basis = z0 @ np.linalg.inv(chol.T)
    az = a @ z_basis
    reduced = -lam * np.linalg.solve(
        np.eye(z_basis.shape[1]) + lam * az.T @ weight @ az,
        az.T @ weight @ e,
    )
    c_reduced = z_basis @ reduced
    old_loss = float(0.5 * e @ weight @ e)
    residual = e + a @ c_kkt
    new_loss = float(0.5 * residual @ weight @ residual)
    bound = old_loss - float(c_kkt @ metric @ c_kkt) / (2 * lam)
    worst_kkt_gap = max(worst_kkt_gap, float(np.linalg.norm(c_kkt - c_reduced)))
    worst_task_error = max(worst_task_error, float(np.linalg.norm(cmat @ c_kkt)))
    worst_bound_violation = max(worst_bound_violation, new_loss - bound)
check(
    "kkt_equals_nullspace_solution_12_cases",
    worst_kkt_gap < 1e-10,
    worst_difference=worst_kkt_gap,
)
check(
    "linear_task_preservation_12_cases",
    worst_task_error < 1e-10,
    worst_residual=worst_task_error,
)
check(
    "linear_observation_descent_bound_12_cases",
    worst_bound_violation < 1e-10,
    worst_bound_violation=float(worst_bound_violation),
)

# An exactly conflicting task and preservation observable admits no correction.
a = np.array([[0.0, 1.0]])
cmat = a.copy()
e = np.array([1.0])
hessian = np.eye(2) + a.T @ a
kkt = np.block([[hessian, cmat.T], [cmat, np.zeros((1, 1))]])
solution = np.linalg.solve(kkt, np.array([0.0, -1.0, 0.0]))[:2]
check(
    "full_conflict_can_give_zero_correction",
    np.linalg.norm(solution) < 1e-10,
    correction_norm=float(np.linalg.norm(solution)),
)

payload = {
    "date": "2026-10-22",
    "scope": "finite_dimensional_algebra_and_counterexamples_only",
    "no_model_experiments": True,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "seed": 20261022,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "check_groups": len(results),
    "all_passed": all(row["passed"] for row in results),
    "results": results,
    "limitations": [
        "No semantic identifiability or image-editing quality validation.",
        "Fourier phase checks do not validate local filters on decoded images.",
        "Synthetic common-factor removal uses a correctly specified nuisance.",
        "No nonlinear finite-step or final-trajectory guarantee is tested.",
    ],
}
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "check_groups": payload["check_groups"],
    "all_passed": payload["all_passed"],
    "kkt_gap": worst_kkt_gap,
    "task_residual": worst_task_error,
    "output": str(out),
}, ensure_ascii=False))
