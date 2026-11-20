"""Matrix-free zero-threshold quadratic correction in the source/target span."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Rank2Result:
    displacement: np.ndarray
    mu: float
    lam: float
    normal: np.ndarray
    iterations: int
    status: str


class Rank2Weight:
    def __init__(self, source, target, alpha=0.3, rho=1.0, lambda_s=1.0, lambda_u=1.0):
        self.alpha, self.rho = float(alpha), float(rho)
        self.ls, self.lu = float(lambda_s), float(lambda_u)
        if alpha <= 0 or rho <= 0 or lambda_s < 0 or lambda_u < 0:
            raise ValueError("invalid positive-definite weight parameters")
        self.units = {}
        for name, value in (("source", source), ("target", target)):
            value = np.asarray(value, dtype=np.float64).reshape(-1)
            norm = np.linalg.norm(value)
            if not np.isfinite(norm):
                raise ValueError("nonfinite response")
            self.units[name] = value / norm if norm > 1e-14 else None
        self.n = np.asarray(source).size
        if np.asarray(target).size != self.n:
            raise ValueError("latent dimension mismatch")
        if self.units["target"] is None:
            self.lu = 0.0
        vectors = [x for x in self.units.values() if x is not None]
        if vectors:
            u, values, _ = np.linalg.svd(np.stack(vectors, axis=1), full_matrices=False)
            self.u = u[:, values > 1e-12]
        else:
            self.u = np.empty((self.n, 0))
        self.k = np.zeros((self.u.shape[1], self.u.shape[1]))
        for name, strength in (("source", self.ls), ("target", -self.lu)):
            v = self.units[name]
            if v is not None:
                projected = self.u.T @ v
                self.k += self.rho * strength * np.outer(projected, projected)

    def weight_action(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        result = (1 + self.lu) * y
        for name, strength in (("source", self.ls), ("target", -self.lu)):
            v = self.units[name]
            if v is not None:
                result = result + strength * v * (v @ y)
        return result

    def hessian_action(self, y, mu=0.0):
        return (self.alpha + mu) * y + self.rho * self.weight_action(y)

    def inverse(self, y, mu=0.0):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        c = self.alpha + self.rho * (1 + self.lu) + mu
        coordinates = self.u.T @ y
        return (y - self.u @ coordinates) / c + self.u @ np.linalg.solve(c * np.eye(len(coordinates)) + self.k, coordinates)

    def minimum_weight_eigenvalue(self):
        c = 1 + self.lu
        eigen = np.linalg.eigvalsh(c * np.eye(self.k.shape[0]) + self.k / self.rho)
        candidates = list(eigen)
        if self.u.shape[1] < self.n:
            candidates.append(c)
        return float(min(candidates))

    def objective(self, drift, w):
        return float(self.alpha / 2 * np.dot(drift + w, drift + w) + self.rho / 2 * np.dot(w, self.weight_action(w)))

    def solve(self, drift, direction, radius, tol=1e-12):
        drift = np.asarray(drift, dtype=np.float64).reshape(-1)
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        if len(drift) != self.n or len(direction) != self.n:
            raise ValueError("dimension mismatch")
        if not np.isfinite(drift).all() or not np.isfinite(direction).all() or not np.isfinite(radius) or radius < 0:
            raise ValueError("nonfinite input or invalid budget")
        norm = np.linalg.norm(direction)
        if norm < 1e-14 or radius == 0:
            return Rank2Result(np.zeros(self.n), 0, 0, np.zeros(self.n), 0, "policy_fallback")
        normal = direction / norm
        force = self.alpha * drift

        def at(mu):
            w0 = -self.inverse(force, mu)
            a = self.inverse(normal, mu)
            lam = max(-normal @ w0, 0.0) / (normal @ a)
            return w0 + lam * a, float(lam)

        w, lam = at(0)
        if np.linalg.norm(w) <= radius:
            return Rank2Result(w, 0, lam, normal, 0, "optimal")
        lo, hi = 0.0, 1.0
        iterations = 0
        while np.linalg.norm(at(hi)[0]) > radius:
            hi *= 2
            iterations += 1
            if iterations > 1024:
                raise RuntimeError("ball multiplier could not be bracketed")
        for _ in range(160):
            mid = (lo + hi) / 2
            if np.linalg.norm(at(mid)[0]) > radius:
                lo = mid
            else:
                hi = mid
            iterations += 1
            if hi - lo <= tol * (1 + hi):
                break
        w, lam = at(hi)
        return Rank2Result(w, hi, lam, normal, iterations, "optimal")
