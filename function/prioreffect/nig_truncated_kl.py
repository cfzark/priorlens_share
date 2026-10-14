from functools import lru_cache

import numpy as np


@lru_cache(maxsize=128)
def canonical_logspace_hdr_moments(kappa, alpha, sample_power=18, log_t_power=0.5):
    """
    Deterministic Sobol approximation of a NIG HDR in (mu, log sigma^2).

    Under P=NIG(mu, lambda, kappa, eta), let
        T = eta / sigma^2 ~ Gamma(kappa, 1),
        Z = sqrt(lambda / sigma^2) (mu - mu_P) ~ N(0,1).
    T and Z are independent. In (mu, log sigma^2), the density ranking is
        H(T,Z) = (kappa + 1/2) log(T) - T - Z^2/2.
    In (mu, sigma^2), use log_t_power=1.5 instead.
    """
    from scipy import stats
    from scipy.stats import qmc

    kappa = float(kappa)
    alpha = float(alpha)
    sample_power = int(sample_power)
    log_t_power = float(log_t_power)
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if sample_power < 10:
        raise ValueError("sample_power must be at least 10.")

    u = qmc.Sobol(d=2, scramble=True, seed=8675309).random_base2(sample_power)
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)
    t = stats.gamma.ppf(u[:, 0], a=kappa, scale=1.0)
    z = stats.norm.ppf(u[:, 1])
    score = (kappa + log_t_power) * np.log(t) - t - 0.5 * z * z
    threshold = float(np.quantile(score, alpha))
    keep = score >= threshold
    return {
        "threshold": threshold,
        "p_mass": float(np.mean(keep)),
        "mean_log_t": float(np.mean(np.log(t[keep]))),
        "mean_t": float(np.mean(t[keep])),
        "mean_z2": float(np.mean(np.square(z[keep]))),
        "n_samples": int(len(t)),
        "n_inside": int(np.sum(keep)),
    }


def truncated_kl_core_logspace(
    mu_p,
    lambda_p,
    kappa_p,
    eta_p,
    mu_q,
    lambda_q,
    kappa_q,
    eta_q,
    hdr_moments,
):
    """Conditional E_P[log p-log q | HDR_P] for batches of P and Q."""
    from scipy import special

    mu_p = np.asarray(mu_p, dtype=float)
    eta_p = np.asarray(eta_p, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    delta2 = np.square(mu_p[:, None] - mu_q[None, :])

    e_log_t = float(hdr_moments["mean_log_t"])
    e_t = float(hdr_moments["mean_t"])
    e_z2 = float(hdr_moments["mean_z2"])

    constant = (
        kappa_p * np.log(eta_p)[:, None]
        - kappa_q * np.log(float(eta_q))
        - special.gammaln(kappa_p)
        + special.gammaln(kappa_q)
        + 0.5 * np.log(float(lambda_p) / float(lambda_q))
    )
    log_sigma_term = (kappa_q - kappa_p) * (
        np.log(eta_p)[:, None] - e_log_t
    )
    inverse_scale_term = (
        (float(eta_q) / eta_p[:, None] - 1.0) * e_t
        + 0.5
        * float(lambda_q)
        * delta2
        / eta_p[:, None]
        * e_t
        + 0.5 * (float(lambda_q) / float(lambda_p) - 1.0) * e_z2
    )
    return constant + log_sigma_term + inverse_scale_term


def build_q_hdr_probability_table(
    lambda_p,
    kappa_p,
    lambda_q,
    kappa_q,
    eta_q,
    hdr_threshold,
    log_eta_bounds,
    delta_max,
    eta_grid_size=80,
    delta_grid_size=181,
    quadrature_size=40,
    log_t_power=0.5,
):
    """
    Precompute Q(HDR_P) over log(eta_P) and |mu_P-mu_Q|.

    The Gamma expectation under Q is evaluated by generalized Gauss-Laguerre
    quadrature. Bilinear interpolation then makes exhaustive candidate scoring
    inexpensive.
    """
    from scipy import special, stats

    lo, hi = map(float, log_eta_bounds)
    log_t_power = float(log_t_power)
    if hi <= lo:
        pad = 1e-6
        lo -= pad
        hi += pad
    delta_max = max(float(delta_max), 1e-8)
    log_eta_grid = np.linspace(lo, hi, int(eta_grid_size))
    delta_grid = np.linspace(0.0, delta_max, int(delta_grid_size))

    nodes, weights = special.roots_genlaguerre(
        int(quadrature_size), float(kappa_q) - 1.0
    )
    weights = weights / special.gamma(float(kappa_q))
    sqrt_nodes = np.sqrt(nodes)
    c = np.sqrt(float(lambda_p) / float(lambda_q))

    table = np.zeros((len(log_eta_grid), len(delta_grid)), dtype=float)
    for i, log_eta_p in enumerate(log_eta_grid):
        eta_p = np.exp(log_eta_p)
        a = eta_p / float(eta_q)
        t_p = a * nodes
        radius_sq = 2.0 * (
            (float(kappa_p) + log_t_power) * np.log(t_p)
            - t_p
            - float(hdr_threshold)
        )
        valid = radius_sq > 0.0
        if not np.any(valid):
            continue
        radius = np.sqrt(np.maximum(radius_sq, 0.0))
        b = np.sqrt(float(lambda_p) / float(eta_q))
        means = b * delta_grid[:, None] * sqrt_nodes[None, :]
        lower = (-radius[None, :] - means) / c
        upper = (radius[None, :] - means) / c
        conditional = stats.norm.cdf(upper) - stats.norm.cdf(lower)
        conditional[:, ~valid] = 0.0
        table[i] = conditional @ weights

    return {
        "log_eta_grid": log_eta_grid,
        "delta_grid": delta_grid,
        "probability": np.clip(table, 0.0, 1.0),
        "quadrature_size": int(quadrature_size),
    }


def interpolate_q_hdr_probability(log_eta, delta, table):
    """Vectorized bilinear interpolation over a precomputed Q(HDR_P) table."""
    log_eta = np.asarray(log_eta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    x = table["log_eta_grid"]
    y = table["delta_grid"]
    values = table["probability"]

    log_eta = np.clip(log_eta, x[0], x[-1])
    delta = np.clip(np.abs(delta), y[0], y[-1])

    ix = np.searchsorted(x, log_eta, side="right") - 1
    iy = np.searchsorted(y, delta, side="right") - 1
    ix = np.clip(ix, 0, len(x) - 2)
    iy = np.clip(iy, 0, len(y) - 2)

    tx = (log_eta - x[ix]) / (x[ix + 1] - x[ix])
    ty = (delta - y[iy]) / (y[iy + 1] - y[iy])
    v00 = values[ix, iy]
    v10 = values[ix + 1, iy]
    v01 = values[ix, iy + 1]
    v11 = values[ix + 1, iy + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def truncated_kl_logspace_batched(
    mu_p,
    lambda_p,
    kappa_p,
    eta_p,
    mu_q,
    lambda_q,
    kappa_q,
    eta_q,
    hdr_moments,
    probability_table,
    eps=1e-12,
):
    """Fast approximate HDR-truncated KL for all P-candidate/Q-prior pairs."""
    mu_p = np.asarray(mu_p, dtype=float)
    eta_p = np.asarray(eta_p, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    core = truncated_kl_core_logspace(
        mu_p=mu_p,
        lambda_p=lambda_p,
        kappa_p=kappa_p,
        eta_p=eta_p,
        mu_q=mu_q,
        lambda_q=lambda_q,
        kappa_q=kappa_q,
        eta_q=eta_q,
        hdr_moments=hdr_moments,
    )
    q_mass = interpolate_q_hdr_probability(
        np.log(eta_p)[:, None],
        np.abs(mu_p[:, None] - mu_q[None, :]),
        probability_table,
    )
    p_mass = max(float(hdr_moments["p_mass"]), float(eps))
    out = core + np.log(np.maximum(q_mass, float(eps)) / p_mass)
    return np.maximum(out, 0.0)
