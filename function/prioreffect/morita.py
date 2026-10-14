import numpy as np

try:
    from .distributions import NormalDist
    from .models import NormalMeanKnownVarModel
except ImportError:  # pragma: no cover
    from distributions import NormalDist
    from models import NormalMeanKnownVarModel


def second_derivative_central(f, x0, h=1e-4):
    return (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h ** 2)


def morita_prior_curvature_1d(prior_dist, theta_bar, h=1e-4):
    if isinstance(prior_dist, NormalDist):
        return 1.0 / prior_dist.var()
    f = lambda th: prior_dist.logpdf(np.array([th]))[0]
    return float(-second_derivative_central(f, theta_bar, h=h))


def morita_posterior_curvature_1d(y, model, q0_dist, theta_bar, h=1e-4):
    if isinstance(model, NormalMeanKnownVarModel) and isinstance(q0_dist, NormalDist):
        m = len(y)
        return m / (model.sigma ** 2) + 1.0 / q0_dist.var()

    def log_post_unnorm(th):
        return q0_dist.logpdf(np.array([th]))[0] + model.loglikelihood_theta(th, y)

    return float(-second_derivative_central(log_post_unnorm, theta_bar, h=h))


class MoritaESSRunner1D:
    """
    Morita prior ESS runner for one-dimensional models.
    """

    def __init__(self, model, prior_p, prior_q0, theta_bar=None, h=1e-4, T=1000, seed=321):
        self.model = model
        self.prior_p = prior_p
        self.prior_q0 = prior_q0
        self.theta_bar = prior_p.mean() if theta_bar is None else float(theta_bar)
        self.h = h
        self.T = int(T)
        self.seed = int(seed)

    def draw_Ym_from_marginal(self, m, rng):
        theta_t = self.prior_p.sample(1, rng=rng)[0]
        return self.model.sample_given_theta(theta_t, m, rng=rng)

    def estimate_Dp(self):
        return morita_prior_curvature_1d(self.prior_p, self.theta_bar, h=self.h)

    def estimate_Dq(self, m):
        rng = np.random.default_rng(self.seed + 1000 + int(m))
        vals = np.zeros(self.T, dtype=float)
        for t in range(self.T):
            y_m = self.draw_Ym_from_marginal(int(m), rng)
            vals[t] = morita_posterior_curvature_1d(
                y=y_m,
                model=self.model,
                q0_dist=self.prior_q0,
                theta_bar=self.theta_bar,
                h=self.h,
            )
        return float(np.mean(vals)), vals

    def run(self, m_grid):
        m_grid = np.asarray(m_grid, dtype=int)
        Dp = self.estimate_Dp()
        Dq_mean = []
        Dq_samples = []
        for m in m_grid:
            mean_curv, vals = self.estimate_Dq(m)
            Dq_mean.append(mean_curv)
            Dq_samples.append(vals)

        Dq_mean = np.asarray(Dq_mean, dtype=float)
        delta = np.abs(Dp - Dq_mean)
        return {
            "theta_bar": self.theta_bar,
            "Dp": float(Dp),
            "Dq": Dq_mean,
            "delta": delta,
            "m_hat": int(m_grid[np.argmin(delta)]),
            "Dq_samples": Dq_samples,
            "prior_p": self.prior_p,
            "prior_q0": self.prior_q0,
        }


def compute_one_ess(mu0, r, y, model, prior_q0, m_grid, prior_sd=np.sqrt(0.3)):
    prior_p = NormalDist(mu0, prior_sd)
    runner = MoritaESSRunner1D(
        model=model,
        prior_p=prior_p,
        prior_q0=prior_q0,
        seed=1000 + int(r),
    )
    out = runner.run(m_grid)
    return mu0, out["m_hat"]
