import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import norm
try:
    from .distributions import *
    from .utils import posterior_normal
except ImportError:  # pragma: no cover - keeps old notebook-style imports working
    from distributions import *
    from utils import posterior_normal


# Bayesian model objects
class BayesianModel(ABC):
    """
    Interface:
        posterior(y, prior_dist) -> Distribution
        posterior_predictive(posterior_dist, size) -> ndarray
        prior_predictive(prior_dist, size) -> ndarray
    """
    @abstractmethod
    def posterior(self, y, prior_dist):
        pass

    @abstractmethod
    def posterior_predictive(self, posterior_dist, size, rng=None):
        pass

    @abstractmethod
    def prior_predictive(self, prior_dist, size, rng=None):
        pass


class NormalMeanKnownVarModel(BayesianModel):
    """
    X | mu ~ N(mu, sigma^2), sigma known

    Supported priors on mu:
    1) NormalDist(mu0, sd0)
    2) FlatPrior(), corresponding to pi(mu) ∝ 1
    3) JeffreysPrior(), which in this one-parameter location model is treated
       the same as FlatPrior()

    Notes
    -----
    - For FlatPrior / JeffreysPrior, posterior is proper only if n >= 1.
    - Prior predictive is defined only for proper NormalDist prior.
    """
    def __init__(self, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = float(sigma)

    @staticmethod
    def _check_y(y):
        y = np.asarray(y, dtype=float).ravel()
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        return y

    def posterior(self, y, prior_dist):
        y = self._check_y(y)
        n = len(y)

        if isinstance(prior_dist, NormalDist):
            m, v = posterior_normal(y, self.sigma, prior_dist.mu, prior_dist.sd)
            return NormalDist(m, np.sqrt(v))

        if isinstance(prior_dist, (FlatPrior, JeffreysPrior)):
            if n == 0:
                raise ValueError(
                    "FlatPrior / JeffreysPrior requires at least one observation "
                    "for a proper posterior in NormalMeanKnownVarModel."
                )
            return NormalDist(float(np.mean(y)), self.sigma / np.sqrt(n))

        raise ValueError(
            "NormalMeanKnownVarModel requires one of: "
            "NormalDist, FlatPrior, JeffreysPrior."
        )

    def posterior_predictive(self, posterior_dist, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        mu_post = posterior_dist.mean()
        v_post = posterior_dist.var()
        return rng.normal(loc=mu_post, scale=np.sqrt(self.sigma**2 + v_post), size=size)

    def prior_predictive(self, prior_dist, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")

        if isinstance(prior_dist, (FlatPrior, JeffreysPrior)):
            raise ValueError(
                "Prior predictive is not defined for FlatPrior / JeffreysPrior "
                "because the prior is improper."
            )

        if not isinstance(prior_dist, NormalDist):
            raise ValueError(
                "NormalMeanKnownVarModel prior_predictive requires NormalDist prior."
            )

        rng = np.random.default_rng() if rng is None else rng
        mu_star = prior_dist.sample(size, rng=rng)
        return rng.normal(loc=mu_star, scale=self.sigma, size=size)

    def loglikelihood_theta(self, theta, y):
        y = self._check_y(y)
        return float(np.sum(norm.logpdf(y, loc=theta, scale=self.sigma)))

    def sample_given_theta(self, theta, m, rng=None):
        if m <= 0:
            raise ValueError("m must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(loc=theta, scale=self.sigma, size=m)


class TLocationModelIS(BayesianModel):
    """
    X | mu ~ t_df(mu, scale), only location mu unknown.

    This is the importance-sampling posterior approximation that previously
    lived in the notebooks. It is useful for misspecification / heavy-tailed
    sensitivity experiments while preserving the original notebook logic.
    """

    def __init__(
        self,
        df=3,
        scale=1.0,
        n_proposal=20000,
        n_resample=4000,
        proposal_loc="median",
        proposal_scale_factor=2.0,
        proposal_scale_min=0.5,
        eps=1e-300,
    ):
        self.df = df
        self.scale = scale
        self.n_proposal = n_proposal
        self.n_resample = n_resample
        self.proposal_loc = proposal_loc
        self.proposal_scale_factor = proposal_scale_factor
        self.proposal_scale_min = proposal_scale_min
        self.eps = eps

    def _proposal_params(self, y, prior_dist):
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n == 0:
            return prior_dist.mu, prior_dist.sd

        if self.proposal_loc == "mean":
            loc = float(np.mean(y))
        else:
            loc = float(np.median(y))

        if n > 1:
            emp_sd = float(np.std(y, ddof=1))
        else:
            emp_sd = (
                self.scale * np.sqrt(self.df / max(self.df - 2, 1e-8))
                if self.df > 2
                else self.scale
            )

        loc_sd = emp_sd / np.sqrt(max(n, 1))
        prop_sd = max(loc_sd, self.proposal_scale_min) * self.proposal_scale_factor
        return loc, prop_sd

    def _log_prior(self, mu, prior_dist):
        return norm.logpdf(mu, loc=prior_dist.mu, scale=prior_dist.sd)

    def _log_likelihood(self, mu, y):
        mu = np.asarray(mu, dtype=float)
        y = np.asarray(y, dtype=float)
        loglik_mat = stats.t.logpdf(
            y[:, None],
            df=self.df,
            loc=mu[None, :],
            scale=self.scale,
        )
        return np.sum(loglik_mat, axis=0)

    def posterior(self, y, prior_dist, random_seed=123, return_diagnostics=False):
        if not isinstance(prior_dist, NormalDist):
            raise ValueError("TLocationModelIS currently requires NormalDist prior.")

        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(random_seed)

        if len(y) == 0:
            post = NormalDist(prior_dist.mu, prior_dist.sd)
            if return_diagnostics:
                return post, {
                    "ess": np.nan,
                    "proposal_loc": prior_dist.mu,
                    "proposal_sd": prior_dist.sd,
                }
            return post

        g_loc, g_sd = self._proposal_params(y, prior_dist)
        mu_prop = rng.normal(loc=g_loc, scale=g_sd, size=self.n_proposal)

        log_target = self._log_prior(mu_prop, prior_dist) + self._log_likelihood(mu_prop, y)
        log_g = norm.logpdf(mu_prop, loc=g_loc, scale=g_sd)
        log_w = log_target - log_g

        log_w_max = np.max(log_w)
        w_unnorm = np.exp(log_w - log_w_max)
        w_sum = np.sum(w_unnorm)

        if not np.isfinite(w_sum) or w_sum <= 0:
            raise RuntimeError(
                "Importance weights collapsed. Try larger proposal_scale_factor "
                "or different proposal."
            )

        w = w_unnorm / w_sum
        ess = 1.0 / np.sum(w**2)
        idx = rng.choice(len(mu_prop), size=self.n_resample, replace=True, p=w)
        post = KDEPosterior(mu_prop[idx])

        if return_diagnostics:
            return post, {
                "ess": float(ess),
                "proposal_loc": float(g_loc),
                "proposal_sd": float(g_sd),
                "max_weight": float(np.max(w)),
                "min_weight": float(np.min(w)),
            }

        return post

    def posterior_predictive(self, posterior_dist, size, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        mu_star = posterior_dist.sample(size=size, rng=rng)
        return stats.t.rvs(
            df=self.df,
            loc=mu_star,
            scale=self.scale,
            size=size,
            random_state=rng,
        )

    def prior_predictive(self, prior_dist, size, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        mu_star = prior_dist.sample(size=size, rng=rng)
        return stats.t.rvs(
            df=self.df,
            loc=mu_star,
            scale=self.scale,
            size=size,
            random_state=rng,
        )

    def loglikelihood_theta(self, theta, y):
        y = np.asarray(y, dtype=float)
        return float(np.sum(stats.t.logpdf(y, df=self.df, loc=theta, scale=self.scale)))

    def sample_given_theta(self, theta, m, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return stats.t.rvs(
            df=self.df,
            loc=theta,
            scale=self.scale,
            size=m,
            random_state=rng,
        )


class NormalMeanUnknownVarModel(BayesianModel):
    """
    X_i | mu, sigma^2 ~ N(mu, sigma^2), both mu and sigma^2 unknown.

    Supported priors:

    1) Informative conjugate prior:
        prior_dist = NormalInverseGammaDist(mu0, tau0_sq, kappa, eta)

    2) Flat-in-mu baseline with proper IG on sigma^2:
        prior_dist = FlatMuInverseGammaPrior(kappa, eta)
        corresponding to q(mu | sigma^2) ∝ 1 and sigma^2 ~ Inv-Gamma(kappa, eta)

    3) Independent Jeffreys prior:
        prior_dist = IndependentJeffreysPrior()
        corresponding to pi(mu, sigma^2) ∝ 1 / sigma^2

    Main design choice:
    - posterior(y, prior_dist) returns the marginal posterior of mu,
      as a StudentTDist, so it plugs directly into your KL machinery.
    """
    def __init__(self):
        pass

    @staticmethod
    def _check_y(y):
        y = np.asarray(y, dtype=float).ravel()
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        return y

    @staticmethod
    def _sample_stats(y):
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        if n == 0:
            return {
                "n": 0,
                "ybar": np.nan,
                "S": 0.0
            }

        ybar = float(np.mean(y))
        S = float(np.sum((y - ybar) ** 2))
        return {
            "n": n,
            "ybar": ybar,
            "S": S
        }

    def posterior_params(self, y, prior_dist):
        """
        Return posterior parameters in a unified dict.

        For informative NIG prior:
            lambda0 = 1 / tau0_sq
            lambdan = lambda0 + n
            mun = (lambda0*mu0 + n*ybar) / lambdan
            kappan = kappa + n/2
            etan = eta + 0.5 * [S + lambda0*n/lambdan * (ybar-mu0)^2]

            mu | y ~ t_{2*kappan}(mun, scale^2 = etan / (kappan * lambdan))
            sigma^2 | y ~ Inv-Gamma(kappan, etan)

        For flat-mu + IG prior:
            mu | y ~ t_{2*kappa + n - 1}(ybar, scale^2 = (eta + S/2)/(kappa + (n-1)/2) * 1/n)
            sigma^2 | y ~ Inv-Gamma(kappa + (n-1)/2, eta + S/2)

        For independent Jeffreys prior:
            mu | y ~ t_{n-1}(ybar, scale^2 = (S/(n-1)) * 1/n)
            sigma^2 | y ~ Inv-Gamma((n-1)/2, S/2)
        """
        y = self._check_y(y)
        stats_dict = self._sample_stats(y)
        n, ybar, S = stats_dict["n"], stats_dict["ybar"], stats_dict["S"]

        if isinstance(prior_dist, NormalInverseGammaDist):
            if n == 0:
                return {
                    "prior_type": "informative",
                    "mu_df": 2.0 * prior_dist.kappa,
                    "mu_loc": prior_dist.mu0,
                    "mu_scale_sq": prior_dist.eta * prior_dist.tau0_sq / prior_dist.kappa,
                    "sigma2_shape": prior_dist.kappa,
                    "sigma2_scale": prior_dist.eta,
                    "mu_posterior_mean": prior_dist.mu0,
                    "lambda_n": 1.0 / prior_dist.tau0_sq
                }

            lambda0 = 1.0 / prior_dist.tau0_sq
            lambdan = lambda0 + n
            mun = (lambda0 * prior_dist.mu0 + n * ybar) / lambdan
            kappan = prior_dist.kappa + 0.5 * n
            etan = prior_dist.eta + 0.5 * (
                S + (lambda0 * n / lambdan) * (ybar - prior_dist.mu0) ** 2
            )

            mu_df = 2.0 * kappan
            mu_scale_sq = etan / (kappan * lambdan)

            return {
                "prior_type": "informative",
                "mu_df": float(mu_df),
                "mu_loc": float(mun),
                "mu_scale_sq": float(mu_scale_sq),
                "sigma2_shape": float(kappan),
                "sigma2_scale": float(etan),
                "mu_posterior_mean": float(mun),
                "lambda_n": float(lambdan)
            }

        if isinstance(prior_dist, FlatMuInverseGammaPrior):
            if n == 0:
                raise ValueError(
                    "FlatMuInverseGammaPrior is improper in mu, so posterior requires "
                    "at least one observation."
                )

            sigma2_shape = prior_dist.kappa + 0.5 * (n - 1)
            sigma2_scale = prior_dist.eta + 0.5 * S

            mu_df = 2.0 * prior_dist.kappa + n - 1.0
            mu_scale_sq = (sigma2_scale / sigma2_shape) * (1.0 / n)

            return {
                "prior_type": "flat_mu",
                "mu_df": float(mu_df),
                "mu_loc": float(ybar),
                "mu_scale_sq": float(mu_scale_sq),
                "sigma2_shape": float(sigma2_shape),
                "sigma2_scale": float(sigma2_scale),
                "mu_posterior_mean": float(ybar),
                "lambda_n": float(n)
            }

        if isinstance(prior_dist, JeffreysPrior):
            if n < 2:
                raise ValueError(
                    "IndependentJeffreysPrior requires at least n >= 2 "
                    "for a proper posterior."
                )
            if S <= 0:
                raise ValueError(
                    "IndependentJeffreysPrior posterior requires S > 0."
                )

            sigma2_shape = 0.5 * (n - 1)
            sigma2_scale = 0.5 * S

            mu_df = n - 1.0
            mu_scale_sq = (sigma2_scale / sigma2_shape) * (1.0 / n)

            return {
                "prior_type": "independent_jeffreys",
                "mu_df": float(mu_df),
                "mu_loc": float(ybar),
                "mu_scale_sq": float(mu_scale_sq),
                "sigma2_shape": float(sigma2_shape),
                "sigma2_scale": float(sigma2_scale),
                "mu_posterior_mean": float(ybar),
                "lambda_n": float(n)
            }

        raise ValueError(
            "NormalMeanUnknownVarModel requires one of: "
            "NormalInverseGammaDist, FlatMuInverseGammaPrior, IndependentJeffreysPrior."
        )

    def posterior_mu(self, y, prior_dist):
        pars = self.posterior_params(y, prior_dist)
        return StudentTDist(
            df=pars["mu_df"],
            loc=pars["mu_loc"],
            scale=np.sqrt(pars["mu_scale_sq"])
        )

    def posterior_sigma2(self, y, prior_dist):
        pars = self.posterior_params(y, prior_dist)
        return InverseGammaDist(
            shape=pars["sigma2_shape"],
            scale=pars["sigma2_scale"]
        )

    def posterior_joint_summary(self, y, prior_dist):
        """
        Convenience wrapper returning both marginals and raw params.
        """
        pars = self.posterior_params(y, prior_dist)
        return {
            "mu_marginal": StudentTDist(
                df=pars["mu_df"],
                loc=pars["mu_loc"],
                scale=np.sqrt(pars["mu_scale_sq"])
            ),
            "sigma2_marginal": InverseGammaDist(
                shape=pars["sigma2_shape"],
                scale=pars["sigma2_scale"]
            ),
            "params": pars
        }

    def posterior(self, y, prior_dist):
        """
        By design, return the marginal posterior of mu,
        because your current ESS/KL workflow compares distributions on mu.
        """
        return self.posterior_mu(y, prior_dist)

    def posterior_predictive_dist(self, y, prior_dist):
        """
        Posterior predictive distribution for one future observation.

        Informative NIG prior:
            Y_new | y ~ t_{2*kappan}(mun, scale^2 = etan/kappan * (1 + 1/lambdan))

        Flat-mu + IG prior:
            Y_new | y ~ t_{2*kappa+n-1}(ybar, scale^2 = (sigma2_scale/sigma2_shape) * (1 + 1/n))

        Independent Jeffreys prior:
            Y_new | y ~ t_{n-1}(ybar, scale^2 = (sigma2_scale/sigma2_shape) * (1 + 1/n))
        """
        pars = self.posterior_params(y, prior_dist)

        pred_scale_sq = (pars["sigma2_scale"] / pars["sigma2_shape"]) * (
            1.0 + 1.0 / pars["lambda_n"]
        )

        return StudentTDist(
            df=pars["mu_df"],
            loc=pars["mu_loc"],
            scale=np.sqrt(pred_scale_sq)
        )

    def posterior_predictive(self, posterior_dist, size, rng=None):
        """
        Generic sampling from a posterior predictive distribution object.
        Here posterior_dist is expected to be a Distribution, usually StudentTDist.
        """
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        return posterior_dist.sample(size=size, rng=rng)

    def prior_predictive_dist(self, prior_dist):
        """
        Prior predictive for one future observation.

        Only available for the proper informative NIG prior.
        """
        if isinstance(prior_dist, (FlatMuInverseGammaPrior, JeffreysPrior)):
            raise ValueError(
                "Prior predictive is not defined for FlatMuInverseGammaPrior "
                "or IndependentJeffreysPrior because the prior is improper."
            )

        if not isinstance(prior_dist, NormalInverseGammaDist):
            raise ValueError(
                "prior_dist must be one of: NormalInverseGammaDist, "
                "FlatMuInverseGammaPrior, IndependentJeffreysPrior."
            )

        # If mu|sigma2 ~ N(mu0, tau0_sq*sigma2), Y|mu,sigma2 ~ N(mu,sigma2),
        # then Y marginal ~ t_{2kappa}(mu0, scale^2 = eta/kappa * (1 + tau0_sq))
        return StudentTDist(
            df=2.0 * prior_dist.kappa,
            loc=prior_dist.mu0,
            scale=np.sqrt((prior_dist.eta / prior_dist.kappa) * (1.0 + prior_dist.tau0_sq))
        )

    def prior_predictive(self, prior_dist, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        pred = self.prior_predictive_dist(prior_dist)
        return pred.sample(size=size, rng=rng)

    def loglikelihood_theta(self, theta, y):
        """
        theta should be a tuple/list: (mu, sigma2)
        """
        y = self._check_y(y)
        mu, sigma2 = theta
        sigma2 = float(sigma2)

        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")

        return float(np.sum(norm.logpdf(y, loc=mu, scale=np.sqrt(sigma2))))

    def sample_given_theta(self, theta, m, rng=None):
        """
        theta should be a tuple/list: (mu, sigma2)
        """
        if m <= 0:
            raise ValueError("m must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        mu, sigma2 = theta
        sigma2 = float(sigma2)

        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")

        return rng.normal(loc=mu, scale=np.sqrt(sigma2), size=m)


class PoissonModel(BayesianModel):
    """
    X_i | lambda ~ Poisson(lambda)

    Supported priors:
    1) GammaPrior(alpha0, beta0)
    2) FlatPriorOnLambda()          corresponding to pi(lambda) ∝ 1
    3) JeffreysPriorOnLambda()      corresponding to pi(lambda) ∝ lambda^{-1/2}

    posterior(...) returns the posterior distribution of lambda, as GammaDist.
    """
    def __init__(self):
        pass

    @staticmethod
    def _check_y(y):
        y = np.asarray(y, dtype=float).ravel()
        if np.any(y < 0):
            raise ValueError("Poisson data must be nonnegative.")
        return y

    def posterior_params(self, y, prior_dist):
        y = self._check_y(y)
        n = len(y)
        s = float(np.sum(y))

        if isinstance(prior_dist, GammaDist):
            alpha_post = prior_dist.alpha + s
            beta_post = prior_dist.beta + n
            return {
                "alpha": float(alpha_post),
                "beta": float(beta_post)
            }

        if isinstance(prior_dist, FlatPrior):
            # pi(lambda) ∝ 1  ->  Gamma(sum y + 1, n)
            alpha_post = s + 1.0
            beta_post = n
            return {
                "alpha": float(alpha_post),
                "beta": float(beta_post)
            }

        if isinstance(prior_dist, JeffreysPrior):
            # pi(lambda) ∝ lambda^{-1/2} -> Gamma(sum y + 1/2, n)
            alpha_post = s + 0.5
            beta_post = n
            return {
                "alpha": float(alpha_post),
                "beta": float(beta_post)
            }

        raise ValueError(
            "PoissonModel requires one of: "
            "GammaPrior, FlatPriorOnLambda, JeffreysPriorOnLambda."
        )

    def posterior(self, y, prior_dist):
        pars = self.posterior_params(y, prior_dist)
        return GammaDist(pars["alpha"], pars["beta"])

    def prior_predictive(self, prior_dist, size, rng=None):
        """
        Only for proper Gamma prior.
        """
        if size <= 0:
            raise ValueError("size must be positive.")

        if isinstance(prior_dist, (FlatPrior, JeffreysPrior)):
            raise ValueError(
                "Prior predictive is not defined for FlatPriorOnLambda or "
                "JeffreysPriorOnLambda because the prior is improper."
            )

        if not isinstance(prior_dist, GammaDist):
            raise ValueError(
                "PoissonModel prior_predictive requires GammaPrior."
            )

        rng = np.random.default_rng() if rng is None else rng
        lam = prior_dist.sample(size=size, rng=rng)
        return rng.poisson(lam=lam, size=size)

    def posterior_predictive(self, posterior_dist, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        lam = posterior_dist.sample(size=size, rng=rng)
        return rng.poisson(lam=lam, size=size)

    def loglikelihood_theta(self, theta, y):
        y = self._check_y(y)
        lam = float(theta)
        if lam <= 0:
            raise ValueError("lambda must be positive.")
        return float(np.sum(stats.poisson.logpmf(y, mu=lam)))

    def sample_given_theta(self, theta, m, rng=None):
        if m <= 0:
            raise ValueError("m must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        lam = float(theta)
        if lam <= 0:
            raise ValueError("lambda must be positive.")
        return rng.poisson(lam=lam, size=m)


class BinomialModel(BayesianModel):
    """
    Y_i | p ~ Binomial(trials, p)

    Supported priors:
    1) BetaDist(alpha0, beta0)
    2) FlatPrior(), corresponding to Beta(1, 1)
    3) JeffreysPrior(), corresponding to Beta(1/2, 1/2)

    posterior(...) returns the posterior distribution of p, as BetaDist.
    """
    def __init__(self, trials):
        if trials <= 0:
            raise ValueError("trials must be positive.")
        self.trials = int(trials)

    def _check_y(self, y):
        y = np.asarray(y, dtype=float).ravel()
        if np.any(y < 0) or np.any(y > self.trials):
            raise ValueError("Binomial data must be in [0, trials].")
        if not np.all(np.isclose(y, np.round(y))):
            raise ValueError("Binomial data must contain integer counts.")
        return y

    def posterior_params(self, y, prior_dist):
        y = self._check_y(y)
        n = len(y)
        s = float(np.sum(y))
        failures = float(n * self.trials - s)

        if isinstance(prior_dist, BetaDist):
            alpha_post = prior_dist.alpha + s
            beta_post = prior_dist.beta + failures
            return {
                "alpha": float(alpha_post),
                "beta": float(beta_post)
            }

        if isinstance(prior_dist, FlatPrior):
            return {
                "alpha": float(s + 1.0),
                "beta": float(failures + 1.0)
            }

        if isinstance(prior_dist, JeffreysPrior):
            return {
                "alpha": float(s + 0.5),
                "beta": float(failures + 0.5)
            }

        raise ValueError(
            "BinomialModel requires one of: BetaDist, FlatPrior, JeffreysPrior."
        )

    def posterior(self, y, prior_dist):
        pars = self.posterior_params(y, prior_dist)
        return BetaDist(pars["alpha"], pars["beta"])

    def prior_predictive(self, prior_dist, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")

        if isinstance(prior_dist, (FlatPrior, JeffreysPrior)):
            raise ValueError(
                "Prior predictive is not defined for improper baseline prior placeholders."
            )

        if not isinstance(prior_dist, BetaDist):
            raise ValueError("BinomialModel prior_predictive requires BetaDist.")

        rng = np.random.default_rng() if rng is None else rng
        p = prior_dist.sample(size=size, rng=rng)
        return rng.binomial(n=self.trials, p=p, size=size)

    def posterior_predictive(self, posterior_dist, size, rng=None):
        if size <= 0:
            raise ValueError("size must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        p = posterior_dist.sample(size=size, rng=rng)
        return rng.binomial(n=self.trials, p=p, size=size)

    def loglikelihood_theta(self, theta, y):
        y = self._check_y(y)
        p = float(theta)
        if not (0 < p < 1):
            raise ValueError("p must be in (0, 1).")
        return float(np.sum(stats.binom.logpmf(y, n=self.trials, p=p)))

    def sample_given_theta(self, theta, m, rng=None):
        if m <= 0:
            raise ValueError("m must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        p = float(theta)
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1].")
        return rng.binomial(n=self.trials, p=p, size=m)
