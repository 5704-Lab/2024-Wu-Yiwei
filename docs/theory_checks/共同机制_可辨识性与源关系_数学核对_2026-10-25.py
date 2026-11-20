"""Finite-dimensional checks and counterexamples for the research note.

No image models, semantic labels, or editing quality validation are involved.
Operators are explicitly constructed; their identifiability is not evidence
that corresponding task/keep operators are available in frozen SD3.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


rng = np.random.default_rng(20261025)
rows = []


def check(name, condition, **metrics):
    row = dict(name=name, passed=bool(condition), metrics=metrics)
    rows.append(row)
    assert row['passed'], row


# A common mechanism distinguishes a direction that independent parameters hide.
B = np.ones((2, 1))
G = np.array([[1.0], [-1.0]])
P = np.eye(2) - B @ np.linalg.pinv(B)
check('shared_mechanism_accepts_its_legal_direction', np.linalg.norm(P @ B) < 1e-12)
check('shared_mechanism_detects_declared_geometric_direction',
      float((G.T @ P @ G).item()) > 1.9)
P_free = np.eye(2) - np.eye(2) @ np.linalg.pinv(np.eye(2))
check('independent_appearance_can_absorb_all_geometry',
      np.linalg.norm(P_free @ G) < 1e-12)
one_view = np.array([[1.0, 1.0]])
two_views = np.c_[B, G]
check('complementary_views_can_remove_a_declared_local_ambiguity',
      np.linalg.matrix_rank(one_view) == 1 and np.linalg.matrix_rank(two_views) == 2)
check('repeating_same_relation_does_not_add_identifiable_dimensions',
      np.linalg.matrix_rank(np.tile(one_view, (20, 1))) == 1)

# Conditional local error bound: the geometry coordinates are declared, not
# identified as human pose. The error bound is known only in this toy setup.
geometry = np.array([0.3])
noise = np.array([0.01, -0.015])
residual = B @ np.array([0.8]) + G @ geometry + noise
kappa = np.linalg.svd(P @ G, compute_uv=False).min()
tau = np.linalg.norm(P @ residual)
delta = np.linalg.norm(noise)
bound = (tau + delta) / kappa
check('local_geometry_bound_holds_under_declared_bounded_error',
      np.linalg.norm(geometry) <= bound + 1e-12,
      actual_geometry_norm=float(np.linalg.norm(geometry)), bound=float(bound))
unobservable_geometry = B * 100.0
check('unobservable_geometry_has_no_residual_based_magnitude_bound',
      np.linalg.norm(P @ unobservable_geometry) < 1e-10
      and np.linalg.norm(unobservable_geometry) > 100.0)

# Exact profiling of an unconstrained local quadratic; source prior is explicit.
m, n, p = 13, 5, 3
B = rng.normal(size=(m, p))
J = rng.normal(size=(m, n))
W = np.diag(rng.uniform(0.2, 2.0, size=m))
C = np.diag(rng.uniform(0.1, 0.7, size=p))
r = rng.normal(size=m)
d = rng.normal(size=n)
Q = B.T @ W @ B + C
Weff = W - W @ B @ np.linalg.solve(Q, B.T @ W)
e = r + J @ d
a = np.linalg.solve(Q, B.T @ W @ e)
joint = 0.5 * (e - B @ a) @ W @ (e - B @ a) + 0.5 * a @ C @ a
profile = 0.5 * e @ Weff @ e
check('schur_profile_equals_joint_quadratic_minimum',
      abs(joint - profile) < 1e-10, absolute_error=float(abs(joint - profile)))
check('profile_metric_is_symmetric_positive_semidefinite',
      np.allclose(Weff, Weff.T) and np.linalg.eigvalsh(Weff).min() > -1e-10)
gradient = J.T @ Weff @ e
step = 1e-6
profile_at = lambda v: 0.5 * (r + J @ v) @ Weff @ (r + J @ v)
fd = np.array([(profile_at(d + step * v) - profile_at(d - step * v)) / (2 * step)
               for v in np.eye(n)])
check('profile_gradient_matches_finite_difference',
      np.max(np.abs(fd - gradient)) < 1e-7,
      max_error=float(np.max(np.abs(fd - gradient))))
ridge_remaining = np.eye(m) - B @ np.linalg.solve(B.T @ B + C, B.T)
check('ridge_remaining_is_not_an_orthogonal_projector',
      np.linalg.norm(ridge_remaining @ ridge_remaining - ridge_remaining) > 1e-3)
check('ridge_penalizes_some_legal_mechanism_change',
      np.linalg.norm(ridge_remaining @ B) > 1e-3)
check('regularization_does_not_supply_missing_data_rank',
      np.linalg.matrix_rank(one_view) == 1
      and np.linalg.eigvalsh(one_view.T @ one_view + np.eye(2)).min() > 0.9)

# A bound invalidates unrestricted variable elimination.
unbounded_a = 2.0
bounded_a = np.clip(unbounded_a, -0.2, 0.2)
check('bounded_mechanism_requires_different_profile_solution',
      0.5 * (2.0 - bounded_a) ** 2 > 1.0)

# A positive scalar gauge is unidentifiable from a product image alone.
reflectance = np.array([0.2, 0.5, 0.8])
shading = np.array([0.7, 0.8, 0.6])
k = 1.1
check('intrinsic_reflectance_shading_gauge_leaves_image_unchanged',
      np.allclose(reflectance * shading, (k * reflectance) * (shading / k)))

# A declared exact periodic image family has both transport and appearance.
N = 256
q = 2 * np.pi * np.arange(N) / N
base = lambda x: 1.0 + 0.4 * np.cos(3 * x) + 0.2 * np.sin(7 * x) + 0.15 * np.cos(11 * x)
base_dx = lambda x: -1.2 * np.sin(3 * x) + 1.4 * np.cos(7 * x) - 1.65 * np.sin(11 * x)
reaction = 0.15 * np.cos(2 * q) + 0.07 * np.sin(5 * q)
warp_rate = 0.12
image_at = lambda s: np.exp(s * reaction) * base(q - s * warp_rate)
image_rate = reaction * base(q) - warp_rate * base_dx(q)
fd_image = (image_at(step) - image_at(-step)) / (2 * step)
check('restricted_metamorphosis_matches_declared_finite_image_family',
      np.max(np.abs(image_rate - fd_image)) < 1e-8,
      max_error=float(np.max(np.abs(image_rate - fd_image))))
frequencies = np.fft.fftfreq(N) * N
response_filter = np.exp(-0.5 * ((frequencies - 5.0) / 2.0) ** 2)
response_filter[0] = 0.0
filter_image = lambda image: np.fft.ifft(np.fft.fft(image) * response_filter)
z = filter_image(base(q))
dz = filter_image(image_rate)
zp, zm = filter_image(image_at(step)), filter_image(image_at(-step))
phase_rate = np.imag(np.conj(z) * dz) / np.abs(z) ** 2
phase_fd = np.angle(zp * np.conj(zm)) / (2 * step)
amp_rate = np.real(np.conj(z) * dz) / np.abs(z)
amp_fd = (np.abs(zp) - np.abs(zm)) / (2 * step)
valid = np.abs(z) > 1e-4
check('common_transport_reaction_predicts_phase_and_amplitude',
      np.max(np.abs(phase_rate[valid] - phase_fd[valid])) < 1e-6
      and np.max(np.abs(amp_rate[valid] - amp_fd[valid])) < 1e-8)
appearance_dz = filter_image(reaction * base(q))
appearance_phase_rate = np.imag(np.conj(z) * appearance_dz) / np.abs(z) ** 2
check('geometry_fixed_appearance_can_require_phase_change',
      np.max(np.abs(appearance_phase_rate[valid])) > 0.01)
z1 = rng.normal(size=20) + 1j * rng.normal(size=20)
z2 = rng.normal(size=20) + 1j * rng.normal(size=20)
decomposed = ((np.abs(z2) - np.abs(z1)) ** 2
              + 2 * np.abs(z1) * np.abs(z2) * (1 - np.cos(np.angle(z2 * np.conj(z1)))))
check('finite_joint_complex_residual_has_amplitude_phase_identity',
      np.max(np.abs(np.abs(z2 - z1) ** 2 - decomposed)) < 1e-12)
check('sine_only_finite_phase_penalty_misses_pi_error',
      abs(np.sin(np.pi)) < 1e-12 and abs((-1 + 0j) - (1 + 0j)) ** 2 == 4)

# Distance-only relations lose source orientation. Balanced matching is not truth.
points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
rotated = points @ rotation.T
vectors = points[:, None, :] - points[None, :, :]
rotated_vectors = rotated[:, None, :] - rotated[None, :, :]
check('pairwise_distances_alone_cannot_protect_orientation',
      np.allclose(np.linalg.norm(vectors, axis=-1), np.linalg.norm(rotated_vectors, axis=-1)))
check('oriented_source_relations_detect_same_rotation',
      np.linalg.norm(vectors - rotated_vectors) > 1.0)
ambiguous_coupling = 0.25 * np.ones((2, 2))
check('balanced_coupling_does_not_imply_unique_semantic_correspondence',
      np.allclose(ambiguous_coupling.sum(0), 0.5)
      and np.allclose(ambiguous_coupling.sum(1), 0.5)
      and np.count_nonzero(ambiguous_coupling[0]) == 2)

# Local image feasibility and finite source feasibility remain separate checks.
decoder_jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
physical_response = np.array([0.0, 0.0, 1.0])
fitted = decoder_jacobian @ np.linalg.lstsq(decoder_jacobian, physical_response, rcond=None)[0]
check('image_mechanism_can_be_outside_local_decoder_range',
      np.linalg.norm(fitted - physical_response) > 0.99)
check('zero_local_constraint_derivative_does_not_ensure_finite_feasibility',
      0.0 * 0.1 == 0 and 0.1 ** 2 > 0.005)

path = Path(__file__)
output = {
    'scope': 'Constructed mathematical checks only; no SD3 image quality or free-text binding validation.',
    'seed': 20261025,
    'script_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'passed': sum(row['passed'] for row in rows),
    'total': len(rows),
    'checks': rows,
}
path.with_suffix('.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: output[key] for key in ['scope', 'passed', 'total']}, ensure_ascii=False))
