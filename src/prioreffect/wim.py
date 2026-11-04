try:
    from .metrics import Wasserstein1Normal
    from .distributions import NormalDist
    from .reimherr import ReimherrRunner1D
except ImportError:  # pragma: no cover
    from metrics import Wasserstein1Normal
    from distributions import NormalDist
    from reimherr import ReimherrRunner1D

import numpy as np


class NormalNormalWIMRunner1D:
    """
    WIM for the normal mean model with known variance.
    """

    def __init__(self, model, prior_p, prior_b):
        self.model = model
        self.prior_p = prior_p
        self.prior_b = prior_b
        self.metric = Wasserstein1Normal()

    def run(self, y):
        post_p = self.model.posterior(y, self.prior_p)
        post_b = self.model.posterior(y, self.prior_b)
        return {
            "posterior_prior": post_p,
            "posterior_baseline": post_b,
            "wim": float(self.metric.compute(post_p, post_b)),
        }


def compute_one_ess_wim(
    mu0,
    r,
    y,
    model,
    prior_q0,
    m_grid,
    B=2000,
    prior_sd=np.sqrt(0.3),
    true_mu=0.0,
    base_seed=20260312,
):
    rng = np.random.default_rng(base_seed + 10000 * (int(r) + 1))
    prior_p = NormalDist(mu0, prior_sd)

    runner_ecss = ReimherrRunner1D(
        model=model,
        prior_p=prior_p,
        prior_b=prior_q0,
        theta_star=true_mu,
        y_obs=y,
        B=B,
    )
    out_ecss = runner_ecss.run(n=len(y), m_grid=m_grid, rng=rng)

    runner_wim = NormalNormalWIMRunner1D(
        model=model,
        prior_p=prior_p,
        prior_b=prior_q0,
    )
    out_wim = runner_wim.run(y)

    return mu0, out_ecss["ecss"], out_wim["wim"]
