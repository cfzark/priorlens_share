"""Source-faithful Jones augmentation paths for one-dimensional posteriors.

Jones, Trangucci, and Chen (2021), equations (3.3)--(3.4), draw one
parameter from the informative observed-data posterior for each outer path.
Conditional on that parameter, the candidate datasets are independent over
candidate sample sizes and matching directions.  In particular, they are not
nested prefixes of one maximum-length posterior-predictive draw.

This module also supports a deliberately labelled directional-PPD sensitivity:
the positive direction uses the informative posterior and the negative
direction uses the baseline posterior, while retaining the remaining Jones
path construction and selection rule.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .cp import obj_func_exact
from .metrics import WassersteinDistance
from .models import BinomialModel, PoissonModel


SOURCE_VARIANTS = (
    "jones_source_faithful",
    "cp_source_faithful",
    "cp_directional_ppd",
)


def select_source_signed(positive, negative, extra_grid):
    """Apply the Jones pathwise bidirectional selection rule.

    ``positive`` compares the observed informative posterior with a baseline
    posterior augmented to size ``n + r``.  ``negative`` compares the observed
    baseline posterior with an informative posterior augmented to that size.
    Each direction is minimized first; the two minima are then compared.
    Exact objective ties select the positive direction, as in Jones' OPESS
    definition.
    """

    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    extra_grid = np.asarray(extra_grid, dtype=int)
    if positive.shape != negative.shape or positive.shape != extra_grid.shape:
        raise ValueError("positive, negative, and extra_grid must have one shape.")
    if positive.ndim != 1 or len(positive) == 0:
        raise ValueError("Directional objectives must be nonempty vectors.")
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        raise ValueError("Directional objectives must be finite.")

    positive_index = int(np.argmin(positive))
    negative_index = int(np.argmin(negative))
    positive_m = int(extra_grid[positive_index])
    negative_m = int(extra_grid[negative_index])
    positive_min = float(positive[positive_index])
    negative_min = float(negative[negative_index])
    positive_wins = positive_min <= negative_min
    return {
        "signed_m": positive_m if positive_wins else -negative_m,
        "positive_m": positive_m,
        "negative_m": negative_m,
        "positive_min": positive_min,
        "negative_min": negative_min,
        "selected_direction": "positive" if positive_wins else "negative",
    }


def _draw_scalar(distribution, rng):
    value = np.asarray(distribution.sample(1, rng=rng), dtype=float).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError("Posterior parameter draw must be one finite scalar.")
    return float(value[0])


def draw_path_parameters(post_inf, post_base, variant, rng):
    """Draw the latent parameter(s) used by one complete candidate path."""

    if variant not in SOURCE_VARIANTS:
        raise ValueError(f"Unknown source-path variant: {variant}")
    theta_positive = _draw_scalar(post_inf, rng)
    if variant == "cp_directional_ppd":
        theta_negative = _draw_scalar(post_base, rng)
    else:
        theta_negative = theta_positive
    return theta_positive, theta_negative


class JonesSourcePathRunner1D:
    """Run source-factorized Jones paths with Wasserstein or CP loss."""

    def __init__(
        self,
        model,
        prior_inf,
        prior_base,
        variant,
        *,
        alpha=0.05,
        wasserstein_order=2,
        gamma_method="grid",
        gamma_grid_size=2000,
    ):
        if variant not in SOURCE_VARIANTS:
            raise ValueError(f"variant must be one of {SOURCE_VARIANTS}.")
        self.model = model
        self.prior_inf = prior_inf
        self.prior_base = prior_base
        self.variant = str(variant)
        self.alpha = float(alpha)
        self.loss_name = (
            "wasserstein" if variant == "jones_source_faithful" else "cp_exact"
        )
        self.path_scheme = (
            "directional_factorized_ppd"
            if variant == "cp_directional_ppd"
            else "jones_shared_informative_factorized_ppd"
        )
        self.wasserstein = WassersteinDistance(
            order=wasserstein_order,
            gamma_method=gamma_method,
            gamma_grid_size=gamma_grid_size,
        )
        self._use_discrete_cache = isinstance(model, (PoissonModel, BinomialModel))

    def _loss(self, post_a, post_b):
        if self.loss_name == "wasserstein":
            return float(self.wasserstein.compute(post_a, post_b))
        return float(obj_func_exact(post_a, post_b, alpha=self.alpha)[4])

    @staticmethod
    def _update_path_hash(hasher, outer, theta_positive, theta_negative):
        hasher.update(np.asarray([outer], dtype="<i8").tobytes())
        hasher.update(
            np.asarray([theta_positive, theta_negative], dtype="<f8").tobytes()
        )

    @staticmethod
    def _update_candidate_hash(hasher, r, positive_additions, negative_additions):
        hasher.update(np.asarray([r], dtype="<i8").tobytes())
        for additions in (positive_additions, negative_additions):
            values = np.asarray(additions, dtype="<f8").reshape(-1)
            hasher.update(np.asarray([len(values)], dtype="<i8").tobytes())
            hasher.update(values.tobytes())

    def run(self, y, extra_grid, *, S=500, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        y = np.asarray(y, dtype=float).reshape(-1)
        extra_grid = np.unique(np.asarray(extra_grid, dtype=int))
        if len(y) == 0:
            raise ValueError("Observed data must be nonempty.")
        if len(extra_grid) == 0 or np.any(extra_grid < 0):
            raise ValueError("extra_grid must contain nonnegative integers.")
        if int(S) <= 0:
            raise ValueError("S must be positive.")

        post_inf0 = self.model.posterior(y, self.prior_inf)
        post_base0 = self.model.posterior(y, self.prior_base)
        signed = np.empty(int(S), dtype=float)
        positive_mstars = np.empty(int(S), dtype=float)
        negative_mstars = np.empty(int(S), dtype=float)
        positive_sum = np.zeros(len(extra_grid), dtype=float)
        negative_sum = np.zeros(len(extra_grid), dtype=float)
        positive_cache = {}
        negative_cache = {}
        cache_hits = 0
        cache_misses = 0
        path_hasher = hashlib.sha256()

        for outer in range(int(S)):
            theta_positive, theta_negative = draw_path_parameters(
                post_inf0, post_base0, self.variant, rng
            )
            self._update_path_hash(
                path_hasher, outer, theta_positive, theta_negative
            )
            positive = np.empty(len(extra_grid), dtype=float)
            negative = np.empty(len(extra_grid), dtype=float)

            for index, r_value in enumerate(extra_grid):
                r = int(r_value)
                if r == 0:
                    positive_additions = np.empty(0, dtype=float)
                    negative_additions = np.empty(0, dtype=float)
                    post_base_r = post_base0
                    post_inf_r = post_inf0
                else:
                    # Two independent researchers for each r.  Separate calls
                    # also ensure candidate datasets are not nested over r.
                    positive_additions = np.asarray(
                        self.model.sample_given_theta(
                            theta_positive, r, rng=rng
                        ),
                        dtype=float,
                    )
                    negative_additions = np.asarray(
                        self.model.sample_given_theta(
                            theta_negative, r, rng=rng
                        ),
                        dtype=float,
                    )
                    post_base_r = self.model.posterior(
                        np.concatenate([y, positive_additions]), self.prior_base
                    )
                    post_inf_r = self.model.posterior(
                        np.concatenate([y, negative_additions]), self.prior_inf
                    )

                self._update_candidate_hash(
                    path_hasher,
                    r,
                    positive_additions,
                    negative_additions,
                )

                if self._use_discrete_cache:
                    positive_key = (r, int(round(float(np.sum(positive_additions)))))
                    negative_key = (r, int(round(float(np.sum(negative_additions)))))
                else:
                    positive_key = negative_key = None

                if positive_key is not None and positive_key in positive_cache:
                    positive[index] = positive_cache[positive_key]
                    cache_hits += 1
                else:
                    positive[index] = self._loss(post_inf0, post_base_r)
                    cache_misses += 1
                    if positive_key is not None:
                        positive_cache[positive_key] = positive[index]

                if negative_key is not None and negative_key in negative_cache:
                    negative[index] = negative_cache[negative_key]
                    cache_hits += 1
                else:
                    negative[index] = self._loss(post_base0, post_inf_r)
                    cache_misses += 1
                    if negative_key is not None:
                        negative_cache[negative_key] = negative[index]

            selected = select_source_signed(positive, negative, extra_grid)
            signed[outer] = selected["signed_m"]
            positive_mstars[outer] = selected["positive_m"]
            negative_mstars[outer] = selected["negative_m"]
            positive_sum += positive
            negative_sum += negative

        positive_mean = positive_sum / int(S)
        negative_mean = negative_sum / int(S)
        max_extra = int(np.max(extra_grid))
        return {
            "n": int(len(y)),
            "extra_grid": extra_grid.tolist(),
            "signed_mstars": signed.tolist(),
            "epss": float(np.mean(signed)),
            "sd_signed_mstar": (
                float(np.std(signed, ddof=1)) if len(signed) > 1 else 0.0
            ),
            "median_signed_mstar": float(np.median(signed)),
            "q05_signed_mstar": float(np.quantile(signed, 0.05)),
            "q95_signed_mstar": float(np.quantile(signed, 0.95)),
            "positive_mstars": positive_mstars.tolist(),
            "negative_mstars": negative_mstars.tolist(),
            "mean_positive_mstar": float(np.mean(positive_mstars)),
            "mean_negative_mstar": float(np.mean(negative_mstars)),
            "mean_positive_path": positive_mean.tolist(),
            "mean_negative_path": negative_mean.tolist(),
            "best_m_positive_mean_path": int(
                extra_grid[int(np.argmin(positive_mean))]
            ),
            "best_m_negative_mean_path": int(
                extra_grid[int(np.argmin(negative_mean))]
            ),
            "boundary_rate": float(np.mean(np.abs(signed) == max_extra)),
            "positive_boundary_rate": float(np.mean(signed == max_extra)),
            "negative_boundary_rate": float(np.mean(signed == -max_extra)),
            "path_checksum": path_hasher.hexdigest(),
            "loss_cache_hits": int(cache_hits),
            "loss_cache_misses": int(cache_misses),
            "variant": self.variant,
            "loss": self.loss_name,
            "path_scheme": self.path_scheme,
            "m_grid_semantics": "number_augmented",
            "outer_selection_rule": (
                "minimize_each_direction_then_compare_minima; ties_positive"
            ),
        }
