"""Mualem-van Genuchten reference coefficients of the water-content Richards equation.

With z positive downward the equation expands to
    theta_t = f1(theta) theta_z + f2(theta) theta_zz + f3(theta) theta_z^2,
where f1 = -dK/dtheta, f2 = D and f3 = dD/dtheta.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VanGenuchten:
    Ks: float
    theta_s: float
    theta_r: float
    alpha: float
    n: float
    l: float = 0.5

    @property
    def m(self) -> float:
        return 1 - 1 / self.n

    @property
    def span(self) -> float:
        return self.theta_s - self.theta_r

    def Se(self, theta):
        return (np.asarray(theta, dtype=float) - self.theta_r) / self.span

    def psi(self, theta):
        """Pressure head, negative in unsaturated soil."""
        return -((self.Se(theta) ** (-1 / self.m) - 1) ** (1 / self.n)) / self.alpha

    def K(self, theta):
        se = self.Se(theta)
        return self.Ks * se**self.l * (1 - (1 - se ** (1 / self.m)) ** self.m) ** 2

    def dK(self, theta):
        se, m = self.Se(theta), self.m
        a = 1 - se ** (1 / m)
        return (self.Ks / self.span) * (
            self.l * se ** (self.l - 1) * (1 - a**m) ** 2
            + 2 * se ** (self.l - 1 + 1 / m) * (1 - a**m) * a ** (m - 1)
        )

    def D(self, theta):
        se, m, n = self.Se(theta), self.m, self.n
        a = 1 - se ** (1 / m)
        factor = self.Ks / (self.span * self.alpha * n * m)
        return factor * se ** (self.l - 1 / m - 1) * (1 - a**m) ** 2 * (se ** (-1 / m) - 1) ** (1 / n - 1)

    def dD(self, theta):
        se, m, n = self.Se(theta), self.m, self.n
        a = 1 - se ** (1 / m)
        log_derivative = (
            (self.l - 1 / m - 1) / se
            + 2 * se ** (1 / m - 1) * a ** (m - 1) / (1 - a**m)
            + (1 / m) * (1 - 1 / n) * se ** (-1 / m - 1) / (se ** (-1 / m) - 1)
        )
        return self.D(theta) * log_derivative / self.span

    def coefficients(self, theta) -> dict[str, np.ndarray]:
        """Reference coefficient of each active term, keyed by library term name."""
        return {"theta_z": -self.dK(theta), "theta_zz": self.D(theta), "theta_z^2": self.dD(theta)}


@dataclass(frozen=True)
class HeterogeneousSoil:
    """Hydraulic parameters that vary with depth; scalars mean a parameter is uniform.

    With K(theta, z) and psi(theta, z), the water-content form of Richards' equation is
        theta_t = f0 + f1 theta_z + f2 theta_zz + f3 theta_z^2      (Song et al., 2025, Eqs. 4-8)
    Depth derivatives here are total derivatives at fixed theta, taken directly on K and psi, so
    terms that mix different parameters are included and any parameter may vary.
    """

    depths: np.ndarray
    Ks: np.ndarray
    alpha: np.ndarray
    n: np.ndarray
    theta_s: np.ndarray | float
    theta_r: np.ndarray | float

    def at(self, z) -> VanGenuchten:
        def value(parameter):
            return np.interp(z, self.depths, parameter) if np.ndim(parameter) else parameter

        return VanGenuchten(value(self.Ks), value(self.theta_s), value(self.theta_r), value(self.alpha), value(self.n))

    def coefficients(self, theta, z) -> dict[str, np.ndarray]:
        """Reference coefficient of each active term at matching theta and depth arrays."""
        h = float(self.depths[1] - self.depths[0])
        z = np.asarray(z, dtype=float)
        soil, above, below = self.at(z), self.at(z - h), self.at(z + h)
        K, D = soil.K(theta), soil.D(theta)
        dK = soil.dK(theta)
        K_z = (below.K(theta) - above.K(theta)) / (2 * h)
        psi_z = (below.psi(theta) - above.psi(theta)) / (2 * h)
        psi_zz = (below.psi(theta) - 2 * soil.psi(theta) + above.psi(theta)) / h**2
        psi_theta = D / K
        psi_theta_z = (below.D(theta) / below.K(theta) - above.D(theta) / above.K(theta)) / (2 * h)
        return {
            "1": K_z * psi_z + K * psi_zz - K_z,
            "theta_z": dK * psi_z + psi_theta * K_z + 2 * K * psi_theta_z - dK,
            "theta_zz": D,
            "theta_z^2": soil.dD(theta),
        }


def load_heterogeneous_soil(folder: str | Path, theta_s: float | None = None, theta_r: float | None = None,
                            depth_max: float = 1.0) -> HeterogeneousSoil:
    """Read ks.npy, alpha.npy and n.npy; theta_s and theta_r come from files when present.

    For the bundled dataset the author identifies Ks as nodal, with a possibly missing last node.
    A one-short Ks array is extended with its last value. This is an endpoint assumption, not an
    inferred cell-to-node conversion; see docs/data_manifest.json.
    """
    folder = Path(folder)
    alpha, n = np.load(folder / "alpha.npy"), np.load(folder / "n.npy")
    Ks = np.load(folder / "ks.npy")
    if Ks.size == alpha.size - 1:
        Ks = np.r_[Ks, Ks[-1]]

    def uniform_or_file(name, value):
        path = folder / f"{name}.npy"
        if path.exists():
            return np.load(path)
        if value is None:
            raise ValueError(f"{name} is neither in {folder} nor given as a value")
        return float(value)

    return HeterogeneousSoil(np.linspace(0.0, depth_max, alpha.size), Ks, alpha, n,
                             uniform_or_file("theta_s", theta_s), uniform_or_file("theta_r", theta_r))


def coefficient_errors(coefficients: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    """Relative L2 error of each term that has a reference; a term without one is left out.

    Points where the reference is undefined, because a predicted theta lies outside (theta_r, theta_s),
    do not count. A term with no valid points is left out.
    """
    errors = {}
    for name, estimate in coefficients.items():
        if name in reference:
            valid = np.isfinite(reference[name]) & np.isfinite(estimate)
            if valid.any() and np.linalg.norm(reference[name][valid]) > 0:
                gap = estimate[valid] - reference[name][valid]
                errors[name] = float(np.linalg.norm(gap) / np.linalg.norm(reference[name][valid]))
    return errors


# Carsel and Parrish (1988) values, Table 1 of Song et al. (2022); Ks in m/day, alpha in 1/m.
SOILS = {
    "sandy_loam": VanGenuchten(Ks=1.061, theta_s=0.41, theta_r=0.065, alpha=7.5, n=1.89),
    "loam": VanGenuchten(Ks=0.2496, theta_s=0.43, theta_r=0.078, alpha=3.6, n=1.56),
    "silt_loam": VanGenuchten(Ks=0.108, theta_s=0.45, theta_r=0.067, alpha=2.0, n=1.41),
}
