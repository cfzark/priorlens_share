import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import norm, gaussian_kde
from scipy.special import gammaln, digamma
try:
    from .utils import _safe_log
except ImportError:  # pragma: no cover - keeps old notebook-style imports working
    from utils import _safe_log


# Distribution objects: prior / posterior unified API
class Distribution(ABC):
    """
    Unified interface for proper distributions / posteriors.
    """
    @abstractmethod
    def sample(self, size, rng=None):
        pass

    @abstractmethod
    def logpdf(self, x):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def var(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def interval(self, alpha=0.05, n=5000):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        if n <= 0:
            raise ValueError("n must be positive.")
        x = self.sample(n, rng=np.random.default_rng(123))
        return tuple(np.quantile(x, [alpha / 2, 1 - alpha / 2]))

    def prob_interval(self, a, b, n=5000):
        if a > b:
            raise ValueError("Require a <= b.")
        if n <= 0:
            raise ValueError("n must be positive.")
        x = self.sample(n, rng=np.random.default_rng(456))
        return float(np.mean((x >= a) & (x <= b)))


# N(mu, sigma^2)
class NormalDist(Distribution):
    def __init__(self, mu, sd):
        if sd <= 0:
            raise ValueError("sd must be positive.")
        self.mu = float(mu)
        self.sd = float(sd)

    def __repr__(self):
        return f"NormalDist(mu={self.mu}, sd={self.sd})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(loc=self.mu, scale=self.sd, size=size)

    def logpdf(self, x):
        return norm.logpdf(x, loc=self.mu, scale=self.sd)

    def pdf(self, x):
        return norm.pdf(x, loc=self.mu, scale=self.sd)

    def cdf(self, x):
        return norm.cdf(x, loc=self.mu, scale=self.sd)

    def mean(self):
        return self.mu

    def var(self):
        return self.sd**2

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        z = norm.ppf(1 - alpha / 2)
        return float(self.mu - z * self.sd), float(self.mu + z * self.sd)

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        return float(norm.cdf((b - self.mu) / self.sd) - norm.cdf((a - self.mu) / self.sd))


# t_df(loc, scale^2)
class StudentTDist(Distribution):
    """
    Location-scale Student t:
        X ~ t_df(loc, scale^2)

    Here `scale` is the usual scale parameter, not the variance.
    """
    def __init__(self, df, loc=0.0, scale=1.0):
        if df <= 0:
            raise ValueError("df must be positive.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        self.df = float(df)
        self.loc = float(loc)
        self.scale = float(scale)

    def __repr__(self):
        return f"StudentTDist(df={self.df}, loc={self.loc}, scale={self.scale})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return stats.t.rvs(
            df=self.df,
            loc=self.loc,
            scale=self.scale,
            size=size,
            random_state=rng
        )

    def logpdf(self, x):
        return stats.t.logpdf(x, df=self.df, loc=self.loc, scale=self.scale)

    def pdf(self, x):
        return stats.t.pdf(x, df=self.df, loc=self.loc, scale=self.scale)

    def cdf(self, x):
        return stats.t.cdf(x, df=self.df, loc=self.loc, scale=self.scale)

    def mean(self):
        if self.df <= 1:
            return np.nan
        return self.loc

    def var(self):
        if self.df <= 1:
            return np.nan
        if self.df <= 2:
            return np.inf
        return self.scale**2 * self.df / (self.df - 2)

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        return tuple(stats.t.interval(
            1 - alpha,
            df=self.df,
            loc=self.loc,
            scale=self.scale
        ))

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        return float(self.cdf(b) - self.cdf(a))

    def entropy(self):
        """
        Differential entropy of location-scale Student t.
        If Z ~ standard t_nu, then
            h(X) = h(Z) + log(scale)
        and
            h(Z) = log(sqrt(nu) * B(nu/2, 1/2))
                   + (nu+1)/2 * [psi((nu+1)/2) - psi(nu/2)]
        """
        nu = self.df
        log_B = gammaln(nu / 2) + gammaln(0.5) - gammaln((nu + 1) / 2)
        h_std = (
            0.5 * np.log(nu)
            + log_B
            + 0.5 * (nu + 1) * (digamma((nu + 1) / 2) - digamma(nu / 2))
        )
        return float(h_std + np.log(self.scale))


# IG(kappa, eta)
class InverseGammaDist(Distribution):
    """
    Inverse-Gamma with scipy parameterization:
        X ~ InvGamma(a=shape, scale=scale)

    density:
        f(x) = scale^shape / Gamma(shape) * x^{-shape-1} exp(-scale/x), x>0
    """
    def __init__(self, shape, scale):
        if shape <= 0:
            raise ValueError("shape must be positive.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        self.shape = float(shape)
        self.scale = float(scale)

    def __repr__(self):
        return f"InverseGammaDist(shape={self.shape}, scale={self.scale})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return stats.invgamma.rvs(
            a=self.shape,
            scale=self.scale,
            size=size,
            random_state=rng
        )

    def logpdf(self, x):
        return stats.invgamma.logpdf(x, a=self.shape, scale=self.scale)

    def pdf(self, x):
        return stats.invgamma.pdf(x, a=self.shape, scale=self.scale)

    def cdf(self, x):
        return stats.invgamma.cdf(x, a=self.shape, scale=self.scale)

    def mean(self):
        if self.shape <= 1:
            return np.inf
        return self.scale / (self.shape - 1)

    def var(self):
        if self.shape <= 2:
            return np.inf
        return (self.scale**2) / ((self.shape - 1)**2 * (self.shape - 2))

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        return tuple(stats.invgamma.interval(
            1 - alpha,
            a=self.shape,
            scale=self.scale
        ))

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        return float(self.cdf(b) - self.cdf(a))

    def entropy(self):
        return float(stats.invgamma.entropy(a=self.shape, scale=self.scale))


# Gamma(alpha, rate = beta)
class GammaDist(Distribution):
    """
    Gamma(shape=alpha, rate=beta)

    density:
        f(x) = beta^alpha / Gamma(alpha) * x^(alpha-1) * exp(-beta x), x > 0
    """
    def __init__(self, alpha, beta):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __repr__(self):
        return f"GammaDist(alpha={self.alpha}, beta={self.beta})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        # scipy uses scale = 1 / rate
        return stats.gamma.rvs(
            a=self.alpha,
            scale=1.0 / self.beta,
            size=size,
            random_state=rng
        )

    def logpdf(self, x):
        return stats.gamma.logpdf(x, a=self.alpha, scale=1.0 / self.beta)

    def pdf(self, x):
        return stats.gamma.pdf(x, a=self.alpha, scale=1.0 / self.beta)

    def cdf(self, x):
        return stats.gamma.cdf(x, a=self.alpha, scale=1.0 / self.beta)

    def mean(self):
        return self.alpha / self.beta

    def var(self):
        return self.alpha / (self.beta ** 2)

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        return tuple(stats.gamma.interval(
            1 - alpha,
            a=self.alpha,
            scale=1.0 / self.beta
        ))

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        return float(self.cdf(b) - self.cdf(a))


class BetaDist(Distribution):
    """
    Beta(alpha, beta) distribution on (0, 1).
    """
    def __init__(self, alpha, beta):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __repr__(self):
        return f"BetaDist(alpha={self.alpha}, beta={self.beta})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return rng.beta(self.alpha, self.beta, size=size)

    def logpdf(self, x):
        return stats.beta.logpdf(x, a=self.alpha, b=self.beta)

    def pdf(self, x):
        return stats.beta.pdf(x, a=self.alpha, b=self.beta)

    def cdf(self, x):
        return stats.beta.cdf(x, a=self.alpha, b=self.beta)

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def var(self):
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total ** 2 * (total + 1.0))

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        return tuple(stats.beta.interval(1 - alpha, a=self.alpha, b=self.beta))

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        return float(self.cdf(b) - self.cdf(a))


class NegativeBinomialDist(Distribution):
    """
    Negative-binomial predictive distribution for Poisson-Gamma models.

    Uses scipy.stats.nbinom with pmf proportional to p^size (1-p)^y,
    y = 0, 1, ...
    """
    def __init__(self, size, prob):
        if size <= 0:
            raise ValueError("size must be positive.")
        if not (0 < prob < 1):
            raise ValueError("prob must be in (0, 1).")
        self.size = float(size)
        self.prob = float(prob)

    def __repr__(self):
        return f"NegativeBinomialDist(size={self.size}, prob={self.prob})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return stats.nbinom.rvs(self.size, self.prob, size=size, random_state=rng)

    def logpdf(self, x):
        return stats.nbinom.logpmf(x, self.size, self.prob)

    def mean(self):
        return float(self.size * (1.0 - self.prob) / self.prob)

    def var(self):
        return float(self.size * (1.0 - self.prob) / (self.prob ** 2))

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        lo = stats.nbinom.ppf(alpha / 2.0, self.size, self.prob)
        hi = stats.nbinom.ppf(1.0 - alpha / 2.0, self.size, self.prob)
        return float(lo), float(hi)

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        lo = int(np.ceil(a))
        hi = int(np.floor(b))
        if hi < lo:
            return 0.0
        return float(
            stats.nbinom.cdf(hi, self.size, self.prob)
            - stats.nbinom.cdf(lo - 1, self.size, self.prob)
        )


class BetaBinomialDist(Distribution):
    """
    Beta-binomial posterior predictive distribution for one future binomial
    observation with a fixed number of trials.
    """
    def __init__(self, trials, alpha, beta):
        if trials <= 0:
            raise ValueError("trials must be positive.")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive.")
        self.trials = int(trials)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __repr__(self):
        return f"BetaBinomialDist(trials={self.trials}, alpha={self.alpha}, beta={self.beta})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return stats.betabinom.rvs(self.trials, self.alpha, self.beta, size=size, random_state=rng)

    def logpdf(self, x):
        return stats.betabinom.logpmf(x, self.trials, self.alpha, self.beta)

    def mean(self):
        return float(self.trials * self.alpha / (self.alpha + self.beta))

    def var(self):
        a, b, n = self.alpha, self.beta, self.trials
        return float(n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1.0)))

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        lo = stats.betabinom.ppf(alpha / 2.0, self.trials, self.alpha, self.beta)
        hi = stats.betabinom.ppf(1.0 - alpha / 2.0, self.trials, self.alpha, self.beta)
        return float(lo), float(hi)

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        lo = int(np.ceil(a))
        hi = int(np.floor(b))
        if hi < lo:
            return 0.0
        return float(
            stats.betabinom.cdf(hi, self.trials, self.alpha, self.beta)
            - stats.betabinom.cdf(lo - 1, self.trials, self.alpha, self.beta)
        )


class EmpiricalDist(Distribution):
    """
    Simple empirical distribution based on samples.
    KDE is cached and used only for smooth density evaluation.
    """
    def __init__(self, samples, bw_method=None):
        samples = np.asarray(samples, dtype=float).ravel()
        if samples.size == 0:
            raise ValueError("samples must be non-empty.")
        self.samples = samples
        self.bw_method = bw_method
        self.kde = gaussian_kde(self.samples, bw_method=bw_method)

    def __repr__(self):
        return f"EmpiricalDist(n_samples={len(self.samples)}, bw_method={self.bw_method})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.integers(0, len(self.samples), size=size)
        return self.samples[idx]

    def logpdf(self, x):
        return _safe_log(self.kde(x))

    def pdf(self, x):
        return self.kde(x)

    def mean(self):
        return float(np.mean(self.samples))

    def var(self):
        if len(self.samples) < 2:
            return 0.0
        return float(np.var(self.samples, ddof=1))

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        return tuple(np.quantile(self.samples, [alpha / 2, 1 - alpha / 2]))

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        return float(np.mean((self.samples >= a) & (self.samples <= b)))


class KDEPosterior(Distribution):
    """
    Posterior represented by samples + KDE.
    Density evaluation uses KDE; sampling resamples from stored samples.
    """
    def __init__(self, samples, bw_method=None):
        samples = np.asarray(samples, dtype=float).ravel()
        if samples.size == 0:
            raise ValueError("samples must be non-empty.")
        self.samples = samples
        self.bw_method = bw_method
        self.kde = gaussian_kde(self.samples, bw_method=bw_method)

    def __repr__(self):
        return f"KDEPosterior(n_samples={len(self.samples)}, bw_method={self.bw_method})"

    def sample(self, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.integers(0, len(self.samples), size=size)
        return self.samples[idx]

    def logpdf(self, x):
        return _safe_log(self.kde(x))

    def pdf(self, x):
        return self.kde(x)

    def mean(self):
        return float(np.mean(self.samples))

    def var(self):
        if len(self.samples) < 2:
            return 0.0
        return float(np.var(self.samples, ddof=1))

    def interval(self, alpha=0.05):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1).")
        return tuple(np.quantile(self.samples, [alpha / 2, 1 - alpha / 2]))

    def prob_interval(self, a, b):
        if a > b:
            raise ValueError("Require a <= b.")
        return float(self.kde.integrate_box_1d(a, b))

# NIG
class NormalInverseGammaDist:
    """
    Normal-Inverse-Gamma Distribution:

        sigma^2 ~ IG(kappa, eta)
        mu | sigma^2 ~ N(mu0, tau0_sq * sigma^2)

    Parameterization used here
    --------------------------
    The inverse-gamma density is

        f(x) = eta^kappa / Gamma(kappa) * x^{-kappa-1} * exp(-eta / x),   x > 0

    So:
    - kappa is the shape parameter
    - eta   is the scale-like parameter in the inverse-gamma density above

    Equivalently:
        if sigma^2 ~ IG(kappa, eta),
        then 1 / sigma^2 ~ Gamma(shape=kappa, rate=eta)

    Therefore, when sampling with NumPy's gamma generator, which uses
    shape-scale parameterization, we must use

        scale = 1 / eta

    i.e.
        inv_sigma2 = rng.gamma(shape=kappa, scale=1.0 / eta, size=size)

    Notes
    -----
    This class is a joint prior object, not a univariate Distribution.
    So it intentionally does NOT inherit from the current Distribution base class.
    """

    def __init__(self, mu0, tau0_sq, kappa, eta):
        if tau0_sq <= 0:
            raise ValueError("tau0_sq must be positive.")
        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")

        self.mu0 = float(mu0)
        self.tau0_sq = float(tau0_sq)
        self.kappa = float(kappa)
        self.eta = float(eta)

        # Cached constants for faster repeated logpdf evaluation
        self._log_const_sigma2 = self.kappa * np.log(self.eta) - gammaln(self.kappa)
        self._log_const_mu_given_sigma2 = -0.5 * np.log(2.0 * np.pi * self.tau0_sq)

    def __repr__(self):
        return (
            f"NormalInverseGammaDist(mu0={self.mu0}, "
            f"tau0_sq={self.tau0_sq}, kappa={self.kappa}, eta={self.eta})"
        )

    def sample(self, size, rng=None):
        """
        Sample from the joint NIG prior.

        Returns
        -------
        mu : ndarray, shape (size,)
        sigma2 : ndarray, shape (size,)
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        rng = np.random.default_rng() if rng is None else rng

        # If sigma^2 ~ IG(kappa, eta), then 1/sigma^2 ~ Gamma(shape=kappa, rate=eta)
        # NumPy gamma uses shape-scale, so scale = 1 / eta.
        inv_sigma2 = rng.gamma(shape=self.kappa, scale=1.0 / self.eta, size=size)
        sigma2 = 1.0 / inv_sigma2

        mu = rng.normal(
            loc=self.mu0,
            scale=np.sqrt(self.tau0_sq * sigma2),
            size=size
        )

        return mu, sigma2

    def logpdf(self, mu, sigma2):
        """
        Joint log density log p(mu, sigma^2).

        Supports broadcasting:
        - mu and sigma2 can be scalars or arrays
        - they will be broadcast to a common shape

        Returns
        -------
        out : ndarray or scalar
            Joint log density evaluated elementwise after broadcasting.
        """
        mu = np.asarray(mu, dtype=float)
        sigma2 = np.asarray(sigma2, dtype=float)
        mu, sigma2 = np.broadcast_arrays(mu, sigma2)

        out = np.full(mu.shape, -np.inf, dtype=float)

        mask = sigma2 > 0
        if not np.any(mask):
            return float(out) if out.ndim == 0 else out

        s2 = sigma2[mask]
        mm = mu[mask]

        # log p(sigma^2)
        logp_sigma2 = (
            self._log_const_sigma2
            - (self.kappa + 1.0) * np.log(s2)
            - self.eta / s2
        )

        # log p(mu | sigma^2)
        logp_mu_given_sigma2 = (
            self._log_const_mu_given_sigma2
            - 0.5 * np.log(s2)
            - ((mm - self.mu0) ** 2) / (2.0 * self.tau0_sq * s2)
        )

        out[mask] = logp_sigma2 + logp_mu_given_sigma2
        return float(out) if out.ndim == 0 else out

    def sample_sigma2(self, size, rng=None):
        """
        Sample sigma^2 marginally from IG(kappa, eta).
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        rng = np.random.default_rng() if rng is None else rng
        inv_sigma2 = rng.gamma(shape=self.kappa, scale=1.0 / self.eta, size=size)
        return 1.0 / inv_sigma2

    def mean_mu(self):
        """
        Marginal mean of mu.
        """
        return self.mu0

    def mean_sigma2(self):
        """
        Marginal mean of sigma^2, if it exists.
        """
        if self.kappa <= 1:
            return np.inf
        return self.eta / (self.kappa - 1)

    def var_sigma2(self):
        """
        Marginal variance of sigma^2, if it exists.
        """
        if self.kappa <= 2:
            return np.inf
        return (self.eta ** 2) / ((self.kappa - 1) ** 2 * (self.kappa - 2))
    
class FlatPrior:
    """
    Minimal placeholder for an improper flat prior.
    """
    is_proper = False

    def __repr__(self):
        return "FlatPrior()"


class JeffreysPrior:
    """
    Minimal placeholder for an improper Jeffreys prior.
    """
    is_proper = False

    def __repr__(self):
        return "JeffreysPrior()"

class FlatMuInverseGammaPrior:
    """
    Baseline prior:
        q(mu | sigma^2) ∝ 1
        sigma^2 ~ Inv-Gamma(kappa, eta)

    This prior is improper in mu, but yields a proper posterior once n >= 1.
    """
    def __init__(self, kappa, eta):
        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")

        self.kappa = float(kappa)
        self.eta = float(eta)

    def __repr__(self):
        return f"FlatMuInverseGammaPrior(kappa={self.kappa}, eta={self.eta})"

    def marginal_sigma2(self):
        return InverseGammaDist(shape=self.kappa, scale=self.eta)

    def sample_sigma2(self, size, rng=None):
        return self.marginal_sigma2().sample(size=size, rng=rng)
