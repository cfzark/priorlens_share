import numpy as np
from scipy import stats
from scipy.stats import norm

def _safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))

def generate_sample_with_target_variance(dist_type, target_var, n, tol=0.01,
                     max_attempts=10000, seed=None):
    """
    Input:
        dist_type: 'normal', 't3', 't10', 'cauchy05', 'cauchy1'
        target_var: target sample variance
        n: sample size
    Return:
        best_sample, best_actual_var, best_seed
    """
    best_seed = seed if seed is not None else 0
    best_var_diff = float('inf')
    best_sample = None
    best_actual_var = None

    for attempt in range(max_attempts):
        current_seed = (seed if seed is not None else 0) + attempt
        rng = np.random.default_rng(current_seed)

        if dist_type == 't3':
            sample = stats.t.rvs(df=3, size=n, random_state=rng)
        elif dist_type == 'cauchy05':
            sample = stats.cauchy.rvs(loc=0, scale=0.5, size=n, random_state=rng)
        elif dist_type == 'cauchy1':
            sample = stats.cauchy.rvs(loc=0, scale=1, size=n, random_state=rng)
        elif dist_type == 't10':
            sample = stats.t.rvs(df=10, size=n, random_state=rng)
        elif dist_type == 'normal':
            sample = stats.norm.rvs(loc=0, scale=np.sqrt(3), size=n, random_state=rng)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        actual_var = np.var(sample, ddof=1)
        var_diff = abs(actual_var - target_var)

        if var_diff < best_var_diff:
            best_var_diff = var_diff
            best_seed = current_seed
            best_sample = sample.copy()
            best_actual_var = actual_var

            if var_diff < tol:
                break

    return best_sample, best_actual_var, best_seed

def posterior_normal(y, sigma, mu0, tau):
    """
    Input:
        y: data
        sigma: likelihood sd
        mu0: prior mean
        tau: prior sd
    Return:
        posterior mean m, posterior variance v
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return float(mu0), float(tau**2)

    ybar = np.mean(y)
    prec = n / (sigma**2) + 1 / (tau**2)
    v = 1 / prec
    m = v * (n * ybar / (sigma**2) + mu0 / (tau**2))
    return float(m), float(v)


def ci_normal(m, v, z):
    s = np.sqrt(v)
    return float(m - z * s), float(m + z * s)


def cp_normal(mA, vA, ciB):
    L, U = ciB
    sA = np.sqrt(vA)
    return float(norm.cdf((U - mA)/sA) - norm.cdf((L - mA)/sA))


def obj_func(mA, vA, mB, vB, alpha=0.05):
    z = norm.ppf(1 - alpha/2)
    ciA = ci_normal(mA, vA, z)
    ciB = ci_normal(mB, vB, z)

    CP1 = cp_normal(mA, vA, ciB)
    CP2 = cp_normal(mB, vB, ciA)

    r1 = max(CP1, CP2)
    r2 = min(CP1, CP2)
    obj = (r1 - r2)**2 + (r2 - (1 - alpha))**2
    return CP1, CP2, r1, r2, obj, ciA, ciB