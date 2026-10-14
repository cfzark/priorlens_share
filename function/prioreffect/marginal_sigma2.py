from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


VALID_COORDINATES = {"sigma2", "log_sigma2"}


def _check_coordinate(coordinate):
    coordinate = str(coordinate).lower()
    if coordinate not in VALID_COORDINATES:
        raise ValueError("coordinate must be 'sigma2' or 'log_sigma2'.")
    return coordinate


@dataclass(frozen=True)
class InverseGammaParams:
    shape: float
    scale: float

    def __post_init__(self):
        if self.shape <= 0.0 or self.scale <= 0.0:
            raise ValueError("Inverse-gamma shape and scale must be positive.")

    def mean(self, coordinate="sigma2"):
        return float(
            inverse_gamma_coordinate_moments(
                self.shape, self.scale, coordinate
            )[0]
        )

    def var(self, coordinate="sigma2"):
        return float(
            inverse_gamma_coordinate_moments(
                self.shape, self.scale, coordinate
            )[1]
        )

    def cdf(self, x):
        return inverse_gamma_cdf(x, self.shape, self.scale)

    def ppf(self, u):
        return stats.invgamma.ppf(u, a=self.shape, scale=self.scale)


def inverse_gamma_cdf(x, shape, scale):
    x = np.asarray(x, dtype=float)
    shape = np.asarray(shape, dtype=float)
    scale = np.asarray(scale, dtype=float)
    return special.gammaincc(shape, scale / np.maximum(x, 1e-300))


def inverse_gamma_coordinate_moments(shape, scale, coordinate="sigma2"):
    coordinate = _check_coordinate(coordinate)
    shape = np.asarray(shape, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if coordinate == "log_sigma2":
        mean = np.log(scale) - special.digamma(shape)
        var = special.polygamma(1, shape)
        return mean, var
    if np.any(shape <= 2.0):
        raise ValueError("Var(sigma^2) requires inverse-gamma shape > 2.")
    mean = scale / (shape - 1.0)
    var = np.square(scale) / (
        np.square(shape - 1.0) * (shape - 2.0)
    )
    return mean, var


def inverse_gamma_kl(shape_p, scale_p, shape_q, scale_q):
    shape_p = np.asarray(shape_p, dtype=float)
    scale_p = np.asarray(scale_p, dtype=float)
    shape_q = np.asarray(shape_q, dtype=float)
    scale_q = np.asarray(scale_q, dtype=float)
    value = (
        (shape_p - shape_q) * special.digamma(shape_p)
        + special.gammaln(shape_q)
        - special.gammaln(shape_p)
        + shape_q * np.log(scale_p / scale_q)
        + shape_p * (scale_q / scale_p - 1.0)
    )
    return np.maximum(value, 0.0)


def standardized_hdr_interval(shape, alpha=0.05, coordinate="sigma2"):
    coordinate = _check_coordinate(coordinate)
    shape = float(shape)
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    mass = 1.0 - alpha
    eps = min(1e-10, alpha * 1e-6)

    def width(lower_probability):
        lower = stats.invgamma.ppf(
            lower_probability, a=shape, scale=1.0
        )
        upper = stats.invgamma.ppf(
            lower_probability + mass, a=shape, scale=1.0
        )
        if coordinate == "log_sigma2":
            return float(np.log(upper) - np.log(lower))
        return float(upper - lower)

    result = optimize.minimize_scalar(
        width,
        bounds=(eps, alpha - eps),
        method="bounded",
        options={"xatol": 1e-12},
    )
    lower_probability = float(result.x)
    lower = float(
        stats.invgamma.ppf(lower_probability, a=shape, scale=1.0)
    )
    upper = float(
        stats.invgamma.ppf(
            lower_probability + mass, a=shape, scale=1.0
        )
    )
    return {
        "lower_standardized": lower,
        "upper_standardized": upper,
        "lower_probability": lower_probability,
        "upper_probability": lower_probability + mass,
        "mass": mass,
        "coordinate": coordinate,
    }


def hdr_cross_coverage(
    shape_p,
    scale_p,
    shape_q,
    scale_q,
    alpha=0.05,
    coordinate="sigma2",
    q_hdr=None,
):
    coordinate = _check_coordinate(coordinate)
    q_hdr = (
        standardized_hdr_interval(shape_q, alpha, coordinate)
        if q_hdr is None
        else q_hdr
    )
    lower = np.asarray(scale_q, dtype=float) * q_hdr["lower_standardized"]
    upper = np.asarray(scale_q, dtype=float) * q_hdr["upper_standardized"]
    return inverse_gamma_cdf(upper, shape_p, scale_p) - inverse_gamma_cdf(
        lower, shape_p, scale_p
    )


def inverse_gamma_wasserstein2(
    shape_p,
    scale_p,
    shape_q,
    scale_q,
    coordinate="sigma2",
    grid_size=256,
    eps=1e-6,
):
    coordinate = _check_coordinate(coordinate)
    u = np.linspace(float(eps), 1.0 - float(eps), int(grid_size))
    qp = stats.invgamma.ppf(u, a=float(shape_p), scale=1.0)
    qq = stats.invgamma.ppf(u, a=float(shape_q), scale=1.0)
    scale_p = np.asarray(scale_p, dtype=float)
    scale_q = np.asarray(scale_q, dtype=float)
    if coordinate == "log_sigma2":
        delta = np.log(scale_p) - np.log(scale_q)
        standardized_delta = np.log(qp) - np.log(qq)
        value = (
            np.square(delta)
            + 2.0 * delta * np.mean(standardized_delta)
            + np.mean(np.square(standardized_delta))
        )
    else:
        value = (
            np.square(scale_p) * np.mean(np.square(qp))
            + np.square(scale_q) * np.mean(np.square(qq))
            - 2.0 * scale_p * scale_q * np.mean(qp * qq)
        )
    return np.sqrt(np.maximum(value, 0.0))


def truncated_inverse_gamma_kl(
    shape_p,
    scale_p,
    shape_q,
    scale_q,
    alpha=0.05,
    coordinate="sigma2",
    quadrature_size=512,
    p_hdr=None,
):
    coordinate = _check_coordinate(coordinate)
    shape_p = float(shape_p)
    scale_p = np.asarray(scale_p, dtype=float).reshape(-1, 1)
    shape_q = np.asarray(shape_q, dtype=float).reshape(1, -1)
    scale_q = np.asarray(scale_q, dtype=float).reshape(1, -1)
    p_hdr = (
        standardized_hdr_interval(shape_p, alpha, coordinate)
        if p_hdr is None
        else p_hdr
    )
    mass = float(p_hdr["mass"])
    u = np.linspace(
        p_hdr["lower_probability"],
        p_hdr["upper_probability"],
        int(quadrature_size),
    )
    z = stats.invgamma.ppf(u, a=shape_p, scale=1.0)
    mean_log_z = float(np.trapz(np.log(z), u) / mass)
    mean_inv_z = float(np.trapz(1.0 / z, u) / mass)
    expected_log_x = np.log(scale_p) + mean_log_z
    expected_inv_x = mean_inv_z / scale_p

    expected_log_ratio = (
        shape_p * np.log(scale_p)
        - shape_q * np.log(scale_q)
        - special.gammaln(shape_p)
        + special.gammaln(shape_q)
        + (shape_q - shape_p) * expected_log_x
        + (scale_q - scale_p) * expected_inv_x
    )

    lower = scale_p * p_hdr["lower_standardized"]
    upper = scale_p * p_hdr["upper_standardized"]
    q_mass = inverse_gamma_cdf(upper, shape_q, scale_q) - inverse_gamma_cdf(
        lower, shape_q, scale_q
    )
    value = expected_log_ratio + np.log(
        np.maximum(q_mass, 1e-300) / mass
    )
    return np.maximum(value, 0.0)
