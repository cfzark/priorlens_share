import numpy as np


def nig_posterior_params_from_stats(
    n,
    sums,
    sums_sq,
    prior_type,
    prior_loc=0.0,
    tau0_sq=0.2,
    kappa=3.0,
    eta=2.0,
):
    """
    Vectorized NIG posterior parameters from normal sufficient statistics.

    Returns parameters under
        sigma^2 | y ~ IG(kappa_n, eta_n)
        mu | sigma^2, y ~ N(mu_n, sigma^2 / lambda_n).
    """
    n = np.asarray(n, dtype=float)
    sums = np.asarray(sums, dtype=float)
    sums_sq = np.asarray(sums_sq, dtype=float)
    n, sums, sums_sq = np.broadcast_arrays(n, sums, sums_sq)
    if np.any(n <= 0):
        raise ValueError("Posterior parameters require n > 0.")

    ybar = sums / n
    sse = np.maximum(sums_sq - sums * sums / n, 0.0)

    if prior_type == "informative":
        lambda0 = 1.0 / float(tau0_sq)
        lambda_n = lambda0 + n
        mu_n = (lambda0 * float(prior_loc) + sums) / lambda_n
        kappa_n = float(kappa) + 0.5 * n
        eta_n = float(eta) + 0.5 * (
            sse + (lambda0 * n / lambda_n) * (ybar - float(prior_loc)) ** 2
        )
    elif prior_type == "flat":
        lambda_n = n
        mu_n = ybar
        kappa_n = float(kappa) + 0.5 * (n - 1.0)
        eta_n = float(eta) + 0.5 * sse
    elif prior_type == "jeffreys":
        if np.any(n < 2):
            raise ValueError("Jeffreys posterior requires n >= 2.")
        lambda_n = n
        mu_n = ybar
        kappa_n = 0.5 * (n - 1.0)
        eta_n = 0.5 * sse
        if np.any(eta_n <= 0):
            raise ValueError("Jeffreys posterior requires positive residual sum of squares.")
    else:
        raise ValueError("prior_type must be informative, flat, or jeffreys.")

    return {
        "mu": np.asarray(mu_n, dtype=float),
        "lambda": np.asarray(lambda_n, dtype=float),
        "kappa": np.asarray(kappa_n, dtype=float),
        "eta": np.asarray(eta_n, dtype=float),
    }


def augmented_sufficient_stats(y, future, extra_grid):
    """Return sufficient statistics for y plus each requested prefix of future."""
    y = np.asarray(y, dtype=float).ravel()
    future = np.asarray(future, dtype=float).ravel()
    extra_grid = np.asarray(extra_grid, dtype=int)
    if np.any(extra_grid < 0):
        raise ValueError("extra_grid must be nonnegative.")
    if len(extra_grid) and int(np.max(extra_grid)) > len(future):
        raise ValueError("future does not contain enough observations.")

    prefix_sum = np.concatenate(([0.0], np.cumsum(future, dtype=float)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(np.square(future), dtype=float)))
    return (
        len(y) + extra_grid,
        np.sum(y, dtype=float) + prefix_sum[extra_grid],
        np.sum(np.square(y), dtype=float) + prefix_sq[extra_grid],
    )


def stack_nig_params(params):
    return {
        key: np.atleast_1d(np.asarray(value, dtype=float))
        for key, value in params.items()
    }


def sample_nig_logspace(params, size, rng, mu_scale=1.0, log_sigma2_scale=1.0):
    """
    Draw batches from NIG distributions and return standardized
    (mu, log(sigma^2)) coordinates plus unstandardized values.
    """
    params = stack_nig_params(params)
    size = int(size)
    if size <= 0:
        raise ValueError("size must be positive.")
    if mu_scale <= 0 or log_sigma2_scale <= 0:
        raise ValueError("Coordinate scales must be positive.")

    shape = params["kappa"][:, None]
    rate = params["eta"][:, None]
    inv_sigma2 = rng.gamma(shape=shape, scale=1.0 / rate, size=(len(shape), size))
    sigma2 = 1.0 / inv_sigma2
    mu = rng.normal(
        loc=params["mu"][:, None],
        scale=np.sqrt(sigma2 / params["lambda"][:, None]),
        size=(len(shape), size),
    )
    z = np.stack(
        (
            mu / float(mu_scale),
            np.log(sigma2) / float(log_sigma2_scale),
        ),
        axis=-1,
    )
    return z, mu, sigma2


def sample_nig_coordinates(
    params,
    size,
    rng,
    parameterization="mu_log_sigma2",
    mu_scale=1.0,
    sigma2_scale=1.0,
    log_sigma2_scale=1.0,
):
    """Draw NIG samples in either (mu, log sigma^2) or (mu, sigma^2)."""
    _, mu, sigma2 = sample_nig_logspace(
        params,
        size,
        rng,
        mu_scale=1.0,
        log_sigma2_scale=1.0,
    )
    key = str(parameterization).lower()
    if mu_scale <= 0 or sigma2_scale <= 0 or log_sigma2_scale <= 0:
        raise ValueError("Coordinate scales must be positive.")
    if key == "mu_log_sigma2":
        z2 = np.log(sigma2) / float(log_sigma2_scale)
    elif key == "mu_sigma2":
        z2 = sigma2 / float(sigma2_scale)
    else:
        raise ValueError("parameterization must be mu_log_sigma2 or mu_sigma2.")
    z = np.stack((mu / float(mu_scale), z2), axis=-1)
    return z, mu, sigma2


def fixed_unit_directions(n_projections):
    """Deterministic evenly spaced directions on the projective unit circle."""
    n_projections = int(n_projections)
    if n_projections <= 0:
        raise ValueError("n_projections must be positive.")
    angles = (np.arange(n_projections, dtype=float) + 0.5) * np.pi / n_projections
    return np.column_stack((np.cos(angles), np.sin(angles)))


def sliced_wasserstein2_batch(target_z, path_z, directions):
    """
    Sliced W2 from one empirical target to a batch of empirical path distributions.

    target_z: (B, 2) or (1, B, 2)
    path_z:   (K, B, 2)
    """
    target_z = np.asarray(target_z, dtype=float)
    path_z = np.asarray(path_z, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if target_z.ndim == 3:
        target_z = target_z[0]
    target_proj = np.sort(target_z @ directions.T, axis=0).T
    path_proj = np.einsum("kbd,ld->klb", path_z, directions)
    path_proj.sort(axis=2)
    return np.sqrt(np.mean(np.square(path_proj - target_proj[None, :, :]), axis=(1, 2)))


def nig_logpdf_logspace(mu, sigma2, params):
    """
    Joint log density in coordinates (mu, log(sigma^2)).

    The +log(sigma^2) term is the Jacobian of sigma^2 = exp(log(sigma^2)).
    Fixed linear coordinate-scale constants are omitted because they cancel in
    every HDR threshold and membership comparison.
    """
    from scipy import special

    params = stack_nig_params(params)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    log_sigma2 = np.log(sigma2)
    return (
        params["kappa"][:, None] * np.log(params["eta"][:, None])
        - special.gammaln(params["kappa"])[:, None]
        + 0.5 * np.log(params["lambda"][:, None] / (2.0 * np.pi))
        - (params["kappa"][:, None] + 0.5) * log_sigma2
        - (
            params["eta"][:, None]
            + 0.5
            * params["lambda"][:, None]
            * np.square(mu - params["mu"][:, None])
        )
        / sigma2
    )


def nig_logpdf_sigma2space(mu, sigma2, params):
    """Joint log density in coordinates (mu, sigma^2), without Jacobian transform."""
    from scipy import special

    params = stack_nig_params(params)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    log_sigma2 = np.log(sigma2)
    return (
        params["kappa"][:, None] * np.log(params["eta"][:, None])
        - special.gammaln(params["kappa"])[:, None]
        + 0.5 * np.log(params["lambda"][:, None] / (2.0 * np.pi))
        - (params["kappa"][:, None] + 1.5) * log_sigma2
        - (
            params["eta"][:, None]
            + 0.5
            * params["lambda"][:, None]
            * np.square(mu - params["mu"][:, None])
        )
        / sigma2
    )


def joint_hdr_cp_batch(target_params, path_params, alpha, size, rng):
    """
    Bidirectional joint-HDR coverage between one target and a batch of NIGs.

    cp1[k] = P_target{Z in HDR_(1-alpha)(path[k])}
    cp2[k] = P_path[k]{Z in HDR_(1-alpha)(target)}
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    target_params = stack_nig_params(target_params)
    path_params = stack_nig_params(path_params)

    _, target_mu, target_sigma2 = sample_nig_logspace(target_params, size, rng)
    _, path_mu, path_sigma2 = sample_nig_logspace(path_params, size, rng)

    target_own = nig_logpdf_logspace(target_mu, target_sigma2, target_params)[0]
    path_own = nig_logpdf_logspace(path_mu, path_sigma2, path_params)
    target_cut = float(np.quantile(target_own, alpha))
    path_cut = np.quantile(path_own, alpha, axis=1)

    target_under_path = nig_logpdf_logspace(
        np.broadcast_to(target_mu, (len(path_params["mu"]), target_mu.shape[1])),
        np.broadcast_to(target_sigma2, (len(path_params["mu"]), target_sigma2.shape[1])),
        path_params,
    )
    path_under_target = nig_logpdf_logspace(
        path_mu,
        path_sigma2,
        {
            key: np.repeat(value, len(path_params["mu"]))
            for key, value in target_params.items()
        },
    )
    cp1 = np.mean(target_under_path >= path_cut[:, None], axis=1)
    cp2 = np.mean(path_under_target >= target_cut, axis=1)
    r1 = np.maximum(cp1, cp2)
    r2 = np.minimum(cp1, cp2)
    obj = np.square(r1 - r2) + np.square(r2 - (1.0 - alpha))
    return cp1, cp2, r1, r2, obj


def joint_hdr_cp_batch_parameterized(
    target_params,
    path_params,
    alpha,
    size,
    rng,
    parameterization="mu_log_sigma2",
):
    """
    Bidirectional joint-HDR coverage in the requested coordinate system.

    For mu_log_sigma2, HDRs use the log-density with the sigma^2 -> log(sigma^2)
    Jacobian. For mu_sigma2, HDRs use the ordinary NIG density in (mu, sigma^2).
    """
    key = str(parameterization).lower()
    if key == "mu_log_sigma2":
        return joint_hdr_cp_batch(target_params, path_params, alpha, size, rng)
    if key != "mu_sigma2":
        raise ValueError("parameterization must be mu_log_sigma2 or mu_sigma2.")

    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    target_params = stack_nig_params(target_params)
    path_params = stack_nig_params(path_params)

    _, target_mu, target_sigma2 = sample_nig_logspace(target_params, size, rng)
    _, path_mu, path_sigma2 = sample_nig_logspace(path_params, size, rng)

    target_own = nig_logpdf_sigma2space(target_mu, target_sigma2, target_params)[0]
    path_own = nig_logpdf_sigma2space(path_mu, path_sigma2, path_params)
    target_cut = float(np.quantile(target_own, alpha))
    path_cut = np.quantile(path_own, alpha, axis=1)

    target_under_path = nig_logpdf_sigma2space(
        np.broadcast_to(target_mu, (len(path_params["mu"]), target_mu.shape[1])),
        np.broadcast_to(target_sigma2, (len(path_params["mu"]), target_sigma2.shape[1])),
        path_params,
    )
    path_under_target = nig_logpdf_sigma2space(
        path_mu,
        path_sigma2,
        {
            key: np.repeat(value, len(path_params["mu"]))
            for key, value in target_params.items()
        },
    )
    cp1 = np.mean(target_under_path >= path_cut[:, None], axis=1)
    cp2 = np.mean(path_under_target >= target_cut, axis=1)
    r1 = np.maximum(cp1, cp2)
    r2 = np.minimum(cp1, cp2)
    obj = np.square(r1 - r2) + np.square(r2 - (1.0 - alpha))
    return cp1, cp2, r1, r2, obj
