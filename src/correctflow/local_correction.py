from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class QPResult:
    displacement: np.ndarray
    halfspace_multiplier: float
    ball_multiplier: float
    constraint_direction: np.ndarray
    iterations: int
    status: str


def solve_zero_threshold_qp(
    drift: np.ndarray,
    direction: np.ndarray,
    weight: np.ndarray,
    *,
    alpha: float = 0.3,
    rho: float = 1.0,
    radius: float,
    tolerance: float = 1e-12,
) -> QPResult:
    """Solve the V2 zero-threshold QP; multipliers use a unit halfspace normal.

    Reference implementation for small-dimensional mathematical audits.
    Real-image inference must use inverse actions in the low-rank basis.
    """
    drift = np.asarray(drift, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    n = len(drift)
    if drift.shape != (n,) or direction.shape != (n,) or weight.shape != (n, n):
        raise ValueError("invalid QP dimensions")
    if not all(np.isfinite(a).all() for a in (drift, direction, weight)):
        raise ValueError("non-finite QP input")
    if alpha <= 0 or rho <= 0 or radius < 0 or not np.isfinite(radius):
        raise ValueError("invalid QP coefficients")
    if not np.allclose(weight, weight.T, atol=1e-12):
        raise ValueError("weight is not symmetric")
    if np.linalg.eigvalsh(weight).min() < 1 - 1e-10:
        raise ValueError("weight must be at least identity")
    norm_a = np.linalg.norm(direction)
    if radius == 0 or norm_a < 1e-14:
        return QPResult(np.zeros(n), 0.0, 0.0, np.zeros(n), 0, "policy_fallback")
    unit = direction / norm_a
    hessian = alpha * np.eye(n) + rho * weight
    force = alpha * drift
    eigvals, eigvecs = np.linalg.eigh(hessian)

    def inverse(y: np.ndarray, mu: float) -> np.ndarray:
        return eigvecs @ ((eigvecs.T @ y) / (eigvals + mu))

    def at(mu: float) -> tuple[np.ndarray, float]:
        w0 = -inverse(force, mu)
        ha = inverse(unit, mu)
        multiplier = max(-unit @ w0, 0.0) / (unit @ ha)
        return w0 + multiplier * ha, float(multiplier)

    w, multiplier = at(0.0)
    if np.linalg.norm(w) <= radius:
        return QPResult(w, multiplier, 0.0, unit, 0, "optimal")
    lo, hi = 0.0, 1.0
    iterations = 0
    while np.linalg.norm(at(hi)[0]) > radius:
        hi *= 2
        iterations += 1
        if iterations > 1024:
            raise RuntimeError("failed to bracket ball multiplier")
    for _ in range(160):
        mid = (lo + hi) / 2
        w_mid, _ = at(mid)
        if np.linalg.norm(w_mid) > radius:
            lo = mid
        else:
            hi = mid
        iterations += 1
        if hi - lo <= tolerance * (1 + hi):
            break
    w, multiplier = at(hi)
    return QPResult(w, multiplier, hi, unit, iterations, "optimal")


def objective(drift, displacement, weight, alpha=0.3, rho=1.0) -> float:
    updated = drift + displacement
    return float(alpha / 2 * updated @ updated + rho / 2 * displacement @ weight @ displacement)


def kkt_residuals(drift, weight, result, radius, alpha=0.3, rho=1.0) -> dict:
    w, a = result.displacement, result.constraint_direction
    mu, lam = result.ball_multiplier, result.halfspace_multiplier
    hessian = alpha * np.eye(len(w)) + rho * weight
    force = alpha * drift
    stationarity = (hessian + mu * np.eye(len(w))) @ w + force - lam * a
    return {
        "stationarity": float(np.linalg.norm(stationarity) / (1 + np.linalg.norm(force) + np.linalg.norm(hessian, 2) * np.linalg.norm(w))),
        "primal": float(max(0.0, -a @ w, np.linalg.norm(w) - radius) / (1 + np.linalg.norm(drift) + radius)),
        "halfspace_complementarity": float(abs(lam * (a @ w)) / (1 + lam * np.linalg.norm(w))),
        "ball_complementarity": float(abs(mu * (w @ w - radius ** 2)) / (1 + mu * radius ** 2)),
    }


def low_rank_inverse_action(y, source, target, *, alpha, rho, lambda_s, lambda_u, mu=0.0):
    """Equation T20, including missing-direction handling from the V2 policy."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    basis_vectors = []
    qs = np.zeros((n, n))
    qa = np.zeros((n, n))
    for vec, label in ((source, "source"), (target, "target")):
        if vec is None or np.linalg.norm(vec) < 1e-14:
            if label == "target":
                lambda_u = 0.0
            continue
        unit = np.asarray(vec) / np.linalg.norm(vec)
        basis_vectors.append(unit)
        if label == "source":
            qs = np.outer(unit, unit)
        else:
            qa = np.outer(unit, unit)
    c = alpha + rho * (1 + lambda_u) + mu
    if not basis_vectors:
        return y / c
    # SVD is used only for this tiny reference basis, never on the model Jacobian.
    u, values, _ = np.linalg.svd(np.stack(basis_vectors, axis=1), full_matrices=False)
    u = u[:, values > 1e-12]
    k = rho * u.T @ (lambda_s * qs - lambda_u * qa) @ u
    return (y - u @ (u.T @ y)) / c + u @ np.linalg.solve(c * np.eye(u.shape[1]) + k, u.T @ y)
