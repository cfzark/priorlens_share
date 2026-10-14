import numpy as np

try:
    from .risk_metrics import SquaredPosteriorMeanError, make_risk_metric
except ImportError:  # pragma: no cover
    from risk_metrics import SquaredPosteriorMeanError, make_risk_metric


class WiesenfarthCalderazzoECSSRunner1D:
    """
    Wiesenfarth-Calderazzo ECSS based on posterior risk matching.

    The baseline prior is evaluated at n observations and the informative
    prior is evaluated at n - m observations. Therefore positive ECSS means
    the informative prior can replace m likelihood observations.
    """

    def __init__(
        self,
        model,
        prior_p,
        prior_b,
        theta_star,
        B=2000,
        risk_metric=None,
        data_sampler=None,
    ):
        self.model = model
        self.prior_p = prior_p
        self.prior_b = prior_b
        self.theta_star = float(theta_star)
        self.B = int(B)
        self.risk_metric = self._coerce_risk_metric(risk_metric)
        self.data_sampler = data_sampler

    @staticmethod
    def _coerce_risk_metric(risk_metric):
        if risk_metric is None:
            return SquaredPosteriorMeanError()
        if isinstance(risk_metric, str):
            return make_risk_metric(risk_metric)
        return risk_metric

    def _simulate_dataset(self, r, rng, sigma=None):
        if self.data_sampler is not None:
            return np.asarray(
                self.data_sampler(theta_star=self.theta_star, r=r, rng=rng),
                dtype=float,
            ).ravel()
        if hasattr(self.model, "sample_given_theta"):
            return np.asarray(
                self.model.sample_given_theta(self.theta_star, r, rng=rng),
                dtype=float,
            ).ravel()
        if sigma is None:
            raise ValueError(
                "sigma is required when model does not implement sample_given_theta."
            )
        return rng.normal(loc=self.theta_star, scale=sigma, size=r)

    def estimate_risk(self, prior, r, rng, sigma=None):
        vals = np.empty(self.B, dtype=float)
        for b in range(self.B):
            yb = self._simulate_dataset(r=r, rng=rng, sigma=sigma)
            posterior = self.model.posterior(yb, prior)
            vals[b] = self.risk_metric.compute(
                posterior=posterior,
                theta_star=self.theta_star,
                rng=rng,
            )
        return float(np.mean(vals))

    def estimate_mse(self, prior, r, rng, sigma=None):
        return self.estimate_risk(prior=prior, r=r, rng=rng, sigma=sigma)

    def run(self, n, m_grid, rng, sigma=None):
        m_grid = np.asarray(m_grid, dtype=int)
        m_grid = m_grid[n - m_grid >= 1]
        if len(m_grid) == 0:
            raise ValueError("No valid m in m_grid such that n - m >= 1.")

        loss_baseline = self.estimate_risk(
            prior=self.prior_b,
            r=n,
            rng=rng,
            sigma=sigma,
        )
        loss_prior = np.empty(len(m_grid), dtype=float)
        for i, m in enumerate(m_grid):
            loss_prior[i] = self.estimate_risk(
                prior=self.prior_p,
                r=int(n - m),
                rng=rng,
                sigma=sigma,
            )

        abs_diff = np.abs(loss_prior - loss_baseline)
        return {
            "n": int(n),
            "m_grid": m_grid,
            "loss_prior": loss_prior,
            "loss_baseline": float(loss_baseline),
            "abs_diff": abs_diff,
            "ecss": int(m_grid[np.argmin(abs_diff)]),
            "risk_metric": repr(self.risk_metric),
        }


def compute_one_ess(
    mu0,
    r,
    y,
    model,
    prior_q0,
    m_grid,
    B=2000,
    prior_sd=np.sqrt(0.3),
    true_mu=0.0,
    sigma=np.sqrt(3.0),
    base_seed=20260312,
):
    try:
        from .distributions import NormalDist
    except ImportError:  # pragma: no cover
        from distributions import NormalDist

    rng = np.random.default_rng(base_seed + 10000 * (int(r) + 1))
    runner = WiesenfarthCalderazzoECSSRunner1D(
        model=model,
        prior_p=NormalDist(mu0, prior_sd),
        prior_b=prior_q0,
        theta_star=true_mu,
        B=B,
    )
    out = runner.run(n=len(y), m_grid=m_grid, rng=rng, sigma=sigma)
    return mu0, out["ecss"]
