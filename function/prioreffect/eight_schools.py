"""Deterministic posterior engine for the Gelman eight-schools model.

The school effects are integrated out when fitting the hyperparameters:

    y_j | mu, tau ~ Normal(mu, s_j^2 + tau^2).

For the priors used by the EPSS experiment, ``mu | eta, y`` is Normal and
only ``eta = log(tau)`` requires numerical integration.  Posterior draws are
iid quadrature-mixture draws; no MCMC is used by the experiment runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats
from scipy.special import logsumexp


EIGHT_SCHOOLS_Y = np.asarray([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
EIGHT_SCHOOLS_S = np.asarray([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
MU_COORD_SCALE = 10.0
TAU_COORD_SCALE = 10.0
TAU2_COORD_SCALE = TAU_COORD_SCALE**2
LOG_PARAMETERIZATION = "mu_over_10_log_tau_over_10"
DIRECT_VARIANCE_PARAMETERIZATION = "mu_over_10_tau2_over_100"

V2_MEAN_LOCATIONS = tuple(np.arange(-20.0, 40.0 + 0.1, 5.0))
V2_MEAN_SDS = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0)
V2_TAU_MEDIANS = (1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0)
V2_TAU_LOG_SDS = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
V2_LITERATURE_MU_SDS = V2_MEAN_SDS
V2_LITERATURE_TAU_SCALES = (2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0)
V2_MEAN_HETEROGENEITY_LOCATIONS = V2_MEAN_LOCATIONS
V2_MEAN_HETEROGENEITY_TAU_SCALES = V2_LITERATURE_TAU_SCALES


@dataclass(frozen=True)
class PriorSpec:
    """Independent prior specification for ``mu`` and ``tau``."""

    prior_id: str
    prior_block: str
    mu_mean: float
    mu_sd: float
    tau_kind: str
    tau_scale: float
    tau_df: float | None = None
    tau_log_sd: float | None = None
    grid_value: float | None = None
    grid_x_name: str | None = None
    grid_x_value: float | None = None
    grid_y_name: str | None = None
    grid_y_value: float | None = None

    def log_mu_density(self, mu):
        return stats.norm.logpdf(mu, loc=self.mu_mean, scale=self.mu_sd)

    def log_eta_density(self, eta):
        """Log density with respect to eta = log(tau)."""
        eta = np.asarray(eta, dtype=float)
        tau = np.exp(eta)
        key = self.tau_kind.lower()
        if key == "half_t":
            if self.tau_df is None:
                raise ValueError("half_t prior requires tau_df")
            out = (
                np.log(2.0)
                + stats.t.logpdf(tau / self.tau_scale, df=self.tau_df)
                - np.log(self.tau_scale)
                + eta
            )
        elif key == "half_cauchy":
            out = (
                np.log(2.0)
                + stats.cauchy.logpdf(tau / self.tau_scale)
                - np.log(self.tau_scale)
                + eta
            )
        elif key == "lognormal":
            if self.tau_log_sd is None:
                raise ValueError("lognormal prior requires tau_log_sd")
            out = stats.norm.logpdf(
                eta, loc=np.log(self.tau_scale), scale=self.tau_log_sd
            )
        elif key == "bounded_uniform":
            # If tau ~ Uniform(0, A), then p_eta(eta)=exp(eta)/A on
            # eta <= log(A).  Keeping the finite support exact matters for
            # the baseline-sensitivity experiment and its HDR geometry.
            out = eta - np.log(self.tau_scale)
            out = np.where(eta <= np.log(self.tau_scale), out, -np.inf)
        else:
            raise ValueError(f"Unknown tau prior kind: {self.tau_kind}")
        return out

    def eta_cdf(self, eta):
        """CDF of the prior for eta=log(tau)."""
        eta = np.asarray(eta, dtype=float)
        tau = np.exp(eta)
        key = self.tau_kind.lower()
        if key == "half_t":
            value = 2.0 * stats.t.cdf(
                tau / self.tau_scale, df=self.tau_df
            ) - 1.0
        elif key == "half_cauchy":
            value = 2.0 * stats.cauchy.cdf(
                tau / self.tau_scale
            ) - 1.0
        elif key == "lognormal":
            value = stats.norm.cdf(
                eta, loc=np.log(self.tau_scale), scale=self.tau_log_sd
            )
        elif key == "bounded_uniform":
            value = np.clip(tau / self.tau_scale, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown tau prior kind: {self.tau_kind}")
        return value

    def sample_hyper(self, size, rng):
        size = int(size)
        mu = rng.normal(self.mu_mean, self.mu_sd, size=size)
        key = self.tau_kind.lower()
        if key == "half_t":
            tau = np.abs(rng.standard_t(self.tau_df, size=size)) * self.tau_scale
        elif key == "half_cauchy":
            tau = np.abs(rng.standard_cauchy(size=size)) * self.tau_scale
        elif key == "lognormal":
            tau = rng.lognormal(np.log(self.tau_scale), self.tau_log_sd, size=size)
        elif key == "bounded_uniform":
            tau = rng.uniform(0.0, self.tau_scale, size=size)
        else:  # pragma: no cover - guarded by construction
            raise ValueError(f"Unknown tau prior kind: {self.tau_kind}")
        return mu, tau


def baseline_prior():
    return PriorSpec(
        prior_id="baseline",
        prior_block="baseline",
        mu_mean=0.0,
        mu_sd=25.0,
        tau_kind="half_t",
        tau_scale=25.0,
        tau_df=4.0,
    )


def bounded_uniform_baseline(upper=100.0):
    """Proper bounded-uniform heterogeneity baseline used for sensitivity."""
    upper = float(upper)
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("bounded-uniform upper limit must be positive")
    return PriorSpec(
        prior_id=f"bounded_uniform_0_{_number_slug(upper)}",
        prior_block="baseline",
        mu_mean=0.0,
        mu_sd=25.0,
        tau_kind="bounded_uniform",
        tau_scale=upper,
    )


def _number_slug(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _values_or_default(values, default):
    return tuple(default if values is None else values)


def informative_prior_grid(
    block="all",
    smoke=False,
    design="v1",
    mean_locations=None,
    mean_sds=None,
    tau_medians=None,
    tau_log_sds=None,
    literature_mu_sds=None,
    literature_tau_scales=None,
    mean_heterogeneity_locations=None,
    mean_heterogeneity_tau_scales=None,
):
    """Build the independent hyperprior grid for the requested design.

    ``v1`` preserves the original one-dimensional blocks.  ``v2`` uses three
    two-dimensional blocks: mean location by mean-prior SD, lognormal tau
    median by log-SD, and mean location by half-Cauchy heterogeneity scale.
    The former ``literature_strength`` block remains available only as a
    legacy explicit block and is no longer included by ``block='all'``.
    """
    block = str(block).lower()
    design = str(design).lower()
    if design not in {"v1", "v2"}:
        raise ValueError("design must be v1 or v2")
    if design == "v2":
        aliases = {
            "tau_conflict": "heterogeneity_conflict",
            "literature_reference": "literature_strength",
        }
        block = aliases.get(block, block)
        return _informative_prior_grid_v2(
            block=block,
            smoke=smoke,
            mean_locations=mean_locations,
            mean_sds=mean_sds,
            tau_medians=tau_medians,
            tau_log_sds=tau_log_sds,
            literature_mu_sds=literature_mu_sds,
            literature_tau_scales=literature_tau_scales,
            mean_heterogeneity_locations=mean_heterogeneity_locations,
            mean_heterogeneity_tau_scales=mean_heterogeneity_tau_scales,
        )

    priors = []
    if block in {"all", "mean_conflict"}:
        values = [0.0, 20.0] if smoke else [-20, -10, 0, 5, 10, 15, 20, 30, 40]
        for value in values:
            priors.append(
                PriorSpec(
                    prior_id=f"mean_mu0_{value:g}",
                    prior_block="mean_conflict",
                    mu_mean=float(value),
                    mu_sd=5.0,
                    tau_kind="half_t",
                    tau_scale=25.0,
                    tau_df=4.0,
                    grid_value=float(value),
                )
            )
    if block in {"all", "tau_conflict"}:
        values = [5.0, 20.0] if smoke else [1, 5, 10, 20, 40]
        for value in values:
            priors.append(
                PriorSpec(
                    prior_id=f"tau_tau0_{value:g}",
                    prior_block="tau_conflict",
                    mu_mean=0.0,
                    mu_sd=25.0,
                    tau_kind="lognormal",
                    tau_scale=float(value),
                    tau_log_sd=0.5,
                    grid_value=float(value),
                )
            )
    if block in {"all", "literature_reference"}:
        priors.append(
            PriorSpec(
                prior_id="literature_mu5_halfcauchy5",
                prior_block="literature_reference",
                mu_mean=0.0,
                mu_sd=5.0,
                tau_kind="half_cauchy",
                tau_scale=5.0,
                grid_value=0.0,
            )
        )
    if not priors:
        raise ValueError(
            "prior block must be all, mean_conflict, tau_conflict, or "
            "literature_reference"
        )
    return priors


def _informative_prior_grid_v2(
    block,
    smoke,
    mean_locations,
    mean_sds,
    tau_medians,
    tau_log_sds,
    literature_mu_sds,
    literature_tau_scales,
    mean_heterogeneity_locations,
    mean_heterogeneity_tau_scales,
):
    priors = []
    if block in {"all", "mean_conflict"}:
        locations = _values_or_default(
            mean_locations, (0.0, 20.0) if smoke else V2_MEAN_LOCATIONS
        )
        sds = _values_or_default(mean_sds, (5.0, 25.0) if smoke else V2_MEAN_SDS)
        for location in locations:
            for sd in sds:
                priors.append(
                    PriorSpec(
                        prior_id=(
                            f"mean_mu0_{_number_slug(location)}"
                            f"_musd_{_number_slug(sd)}"
                        ),
                        prior_block="mean_conflict",
                        mu_mean=float(location),
                        mu_sd=float(sd),
                        tau_kind="half_t",
                        tau_scale=25.0,
                        tau_df=4.0,
                        grid_x_name="mu_prior_mean",
                        grid_x_value=float(location),
                        grid_y_name="mu_prior_sd",
                        grid_y_value=float(sd),
                    )
                )
    if block in {"all", "heterogeneity_conflict"}:
        medians = _values_or_default(
            tau_medians, (5.0, 20.0) if smoke else V2_TAU_MEDIANS
        )
        log_sds = _values_or_default(
            tau_log_sds, (0.5, 1.0) if smoke else V2_TAU_LOG_SDS
        )
        for median in medians:
            for log_sd in log_sds:
                priors.append(
                    PriorSpec(
                        prior_id=(
                            f"heterogeneity_taumed_{_number_slug(median)}"
                            f"_logsd_{_number_slug(log_sd)}"
                        ),
                        prior_block="heterogeneity_conflict",
                        mu_mean=0.0,
                        mu_sd=25.0,
                        tau_kind="lognormal",
                        tau_scale=float(median),
                        tau_log_sd=float(log_sd),
                        grid_x_name="tau_prior_median",
                        grid_x_value=float(median),
                        grid_y_name="tau_prior_log_sd",
                        grid_y_value=float(log_sd),
                    )
                )
    if block == "literature_strength":
        mu_sds = _values_or_default(
            literature_mu_sds,
            (5.0, 25.0) if smoke else V2_LITERATURE_MU_SDS,
        )
        tau_scales = _values_or_default(
            literature_tau_scales,
            (5.0, 25.0) if smoke else V2_LITERATURE_TAU_SCALES,
        )
        for mu_sd in mu_sds:
            for tau_scale in tau_scales:
                priors.append(
                    PriorSpec(
                        prior_id=(
                            f"literature_musd_{_number_slug(mu_sd)}"
                            f"_hcscale_{_number_slug(tau_scale)}"
                        ),
                        prior_block="literature_strength",
                        mu_mean=0.0,
                        mu_sd=float(mu_sd),
                        tau_kind="half_cauchy",
                        tau_scale=float(tau_scale),
                        grid_x_name="mu_prior_sd",
                        grid_x_value=float(mu_sd),
                        grid_y_name="tau_prior_scale",
                        grid_y_value=float(tau_scale),
                    )
                )
    if block in {"all", "mean_heterogeneity_conflict"}:
        locations = _values_or_default(
            mean_heterogeneity_locations,
            (0.0, 20.0) if smoke else V2_MEAN_HETEROGENEITY_LOCATIONS,
        )
        tau_scales = _values_or_default(
            mean_heterogeneity_tau_scales,
            (5.0, 25.0) if smoke else V2_MEAN_HETEROGENEITY_TAU_SCALES,
        )
        for location in locations:
            for tau_scale in tau_scales:
                priors.append(
                    PriorSpec(
                        prior_id=(
                            f"meanhet_mu0_{_number_slug(location)}"
                            f"_hcscale_{_number_slug(tau_scale)}"
                        ),
                        prior_block="mean_heterogeneity_conflict",
                        mu_mean=float(location),
                        mu_sd=5.0,
                        tau_kind="half_cauchy",
                        tau_scale=float(tau_scale),
                        grid_x_name="mu_prior_mean",
                        grid_x_value=float(location),
                        grid_y_name="tau_prior_scale",
                        grid_y_value=float(tau_scale),
                    )
                )
    if not priors:
        raise ValueError(
            "v2 prior block must be all, mean_conflict, "
            "heterogeneity_conflict, mean_heterogeneity_conflict, or "
            "the legacy literature_strength"
        )
    return priors


def _trap_weights(x):
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    out = np.empty_like(x)
    out[0] = dx[0] / 2.0
    out[-1] = dx[-1] / 2.0
    out[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return out


def _weighted_quantile(values, masses, probs):
    values = np.asarray(values, dtype=float)
    masses = np.asarray(masses, dtype=float)
    order = np.argsort(values)
    cdf = np.cumsum(masses[order])
    cdf /= cdf[-1]
    return np.interp(np.asarray(probs, dtype=float), cdf, values[order])


class EightSchoolsPosterior:
    """Collapsed quadrature representation of p(mu, eta | y, s)."""

    def __init__(
        self,
        y,
        s,
        prior: PriorSpec,
        grid_size=801,
        tail_tol=1e-10,
        convergence_tol=1e-5,
        max_expansions=8,
        eta_draw_scheme="piecewise_linear",
    ):
        self.y = np.asarray(y, dtype=float)
        self.s = np.asarray(s, dtype=float)
        if self.y.ndim != 1 or self.s.shape != self.y.shape:
            raise ValueError("y and s must be one-dimensional arrays of equal length")
        if np.any(self.s <= 0):
            raise ValueError("all standard errors must be positive")
        self.prior = prior
        self.grid_size = int(grid_size)
        self.tail_tol = float(tail_tol)
        self.convergence_tol = float(convergence_tol)
        self.max_expansions = int(max_expansions)
        self.eta_draw_scheme = str(eta_draw_scheme).lower()
        self._hyper_marginal_cache = {}
        if self.eta_draw_scheme not in {"node", "piecewise_linear"}:
            raise ValueError("eta_draw_scheme must be node or piecewise_linear")
        self._fit()

    def _conditional_mu(self, eta):
        eta = np.asarray(eta, dtype=float)
        tau2 = np.exp(2.0 * eta)
        obs_var = self.s[None, :] ** 2 + tau2[..., None]
        weights = 1.0 / obs_var
        precision = 1.0 / self.prior.mu_sd**2 + np.sum(weights, axis=-1)
        variance = 1.0 / precision
        mean = variance * (
            self.prior.mu_mean / self.prior.mu_sd**2
            + np.sum(weights * self.y, axis=-1)
        )
        return mean, variance

    def _log_eta_kernel(self, eta):
        eta = np.asarray(eta, dtype=float)
        tau2 = np.exp(2.0 * eta)
        obs_var = self.s[None, :] ** 2 + tau2[..., None]
        precision = 1.0 / self.prior.mu_sd**2 + np.sum(1.0 / obs_var, axis=-1)
        linear = (
            self.prior.mu_mean / self.prior.mu_sd**2
            + np.sum(self.y / obs_var, axis=-1)
        )
        quadratic = (
            self.prior.mu_mean**2 / self.prior.mu_sd**2
            + np.sum(self.y**2 / obs_var, axis=-1)
        )
        log_const = (
            -0.5 * np.sum(np.log(2.0 * np.pi * obs_var), axis=-1)
            - 0.5 * np.log(2.0 * np.pi * self.prior.mu_sd**2)
        )
        log_integral = (
            log_const
            - 0.5 * quadratic
            + 0.5 * linear**2 / precision
            + 0.5 * np.log(2.0 * np.pi / precision)
        )
        return self.prior.log_eta_density(eta) + log_integral

    def _fit(self):
        center = np.log(max(1.0, self.prior.tau_scale))
        lo = min(-12.0, center - 10.0)
        hi = max(np.log(500.0), center + 8.0)
        bounded_above = self.prior.tau_kind.lower() == "bounded_uniform"
        if bounded_above:
            # End the quadrature exactly at the support boundary.  Extending
            # a trapezoid from a positive endpoint density to a zero node
            # above log(A) would otherwise add spurious posterior mass.
            hi = np.log(self.prior.tau_scale)
        previous = None
        for expansion in range(self.max_expansions + 1):
            eta = np.linspace(lo, hi, self.grid_size)
            log_kernel = self._log_eta_kernel(eta)
            weights = _trap_weights(eta)
            log_norm = logsumexp(log_kernel + np.log(weights))
            masses = np.exp(log_kernel + np.log(weights) - log_norm)
            mu_mean, mu_var = self._conditional_mu(eta)
            moments = np.asarray(
                [
                    np.sum(masses * eta),
                    np.sum(masses * eta**2),
                    np.sum(masses * mu_mean),
                    np.sum(masses * (mu_var + mu_mean**2)),
                    *_weighted_quantile(eta, masses, [0.025, 0.5, 0.975]),
                ]
            )
            lower_edge = float(np.sum(masses[:8]))
            upper_edge = 0.0 if bounded_above else float(np.sum(masses[-8:]))
            edge = max(lower_edge, upper_edge)
            converged = previous is not None and np.max(np.abs(moments - previous)) < self.convergence_tol
            if edge < self.tail_tol and (converged or expansion >= 1):
                break
            previous = moments
            width = hi - lo
            lo -= 0.35 * width
            if not bounded_above:
                hi += 0.35 * width
        self.eta = eta
        self.trap_weights = weights
        self.log_norm = float(log_norm)
        self.log_eta_density = log_kernel - log_norm
        self.masses = masses / np.sum(masses)
        self.mu_cond_mean = mu_mean
        self.mu_cond_var = mu_var
        self.tail_mass_bound = edge
        self.n_expansions = expansion
        self._prepare_piecewise_linear_cdf()

    def _prepare_piecewise_linear_cdf(self):
        """Build the continuous CDF implied by trapezoidal quadrature.

        Between adjacent eta nodes the marginal density is treated as linear.
        The resulting interval probabilities are exactly the trapezoidal areas
        after normalization, so continuous draws and numerical integration use
        the same quadrature approximation.
        """
        shifted = self.log_eta_density - np.max(self.log_eta_density)
        density = np.exp(shifted)
        widths = np.diff(self.eta)
        areas = 0.5 * (density[:-1] + density[1:]) * widths
        total = float(np.sum(areas))
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("invalid piecewise-linear eta density")
        probabilities = areas / total
        self.eta_node_density_scaled = density
        self.eta_interval_probabilities = probabilities
        self.eta_interval_cdf = np.r_[0.0, np.cumsum(probabilities)]
        self.eta_interval_cdf[-1] = 1.0

    def _invert_piecewise_linear_cdf(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("CDF probabilities must lie in [0, 1]")
        flat = probabilities.ravel()
        clipped = np.clip(flat, 0.0, np.nextafter(1.0, 0.0))
        index = np.searchsorted(
            self.eta_interval_cdf, clipped, side="right"
        ) - 1
        index = np.clip(index, 0, len(self.eta) - 2)
        cdf_lo = self.eta_interval_cdf[index]
        interval_probability = self.eta_interval_probabilities[index]
        local_u = np.divide(
            clipped - cdf_lo,
            interval_probability,
            out=np.zeros_like(clipped),
            where=interval_probability > 0,
        )
        local_u = np.clip(local_u, 0.0, 1.0)

        f0 = self.eta_node_density_scaled[index]
        f1 = self.eta_node_density_scaled[index + 1]
        delta = f1 - f0
        average_height = 0.5 * (f0 + f1)
        discriminant = np.maximum(
            f0**2 + 2.0 * delta * local_u * average_height, 0.0
        )
        denominator = f0 + np.sqrt(discriminant)
        linear_case = np.abs(delta) <= 1e-12 * np.maximum(f0 + f1, 1e-300)
        fraction = np.empty_like(local_u)
        fraction[linear_case] = local_u[linear_case]
        nonlinear = ~linear_case
        fraction[nonlinear] = np.divide(
            2.0 * local_u[nonlinear] * average_height[nonlinear],
            denominator[nonlinear],
            out=local_u[nonlinear].copy(),
            where=denominator[nonlinear] > 0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        result = self.eta[index] + fraction * (
            self.eta[index + 1] - self.eta[index]
        )
        result[flat >= 1.0] = self.eta[-1]
        return result.reshape(probabilities.shape)

    @property
    def n_schools(self):
        return int(len(self.y))

    def eta_quantile(self, probs):
        result = self._invert_piecewise_linear_cdf(probs)
        return float(result) if np.ndim(probs) == 0 else result

    def eta_cdf(self, values):
        """Evaluate the continuous piecewise-linear posterior eta CDF."""
        values = np.asarray(values, dtype=float)
        flat = values.ravel()
        result = np.zeros_like(flat)
        result[flat >= self.eta[-1]] = 1.0
        inside = (flat > self.eta[0]) & (flat < self.eta[-1])
        if np.any(inside):
            current = flat[inside]
            index = np.searchsorted(self.eta, current, side="right") - 1
            index = np.clip(index, 0, len(self.eta) - 2)
            width = self.eta[index + 1] - self.eta[index]
            fraction = (current - self.eta[index]) / width
            f0 = self.eta_node_density_scaled[index]
            f1 = self.eta_node_density_scaled[index + 1]
            partial_area = width * (
                f0 * fraction
                + 0.5 * (f1 - f0) * fraction**2
            )
            total_scaled = np.sum(
                0.5
                * (
                    self.eta_node_density_scaled[:-1]
                    + self.eta_node_density_scaled[1:]
                )
                * np.diff(self.eta)
            )
            result[inside] = (
                self.eta_interval_cdf[index]
                + partial_area / total_scaled
            )
        result = result.reshape(values.shape)
        return float(result) if values.ndim == 0 else result

    def hyper_marginal_moments(self, label):
        """Mean and variance of a standardized scalar hyperparameter target."""
        if label == "marginal_mu":
            component = self.mu_cond_mean / MU_COORD_SCALE
            conditional_var = self.mu_cond_var / MU_COORD_SCALE**2
        elif label == "marginal_log_tau":
            component = self.eta - np.log(TAU_COORD_SCALE)
            conditional_var = np.zeros_like(component)
        elif label == "marginal_tau2":
            component = np.exp(2.0 * self.eta) / TAU2_COORD_SCALE
            conditional_var = np.zeros_like(component)
        else:
            raise ValueError(f"Unknown hyperparameter marginal: {label}")
        mean = float(np.sum(self.masses * component))
        variance = float(
            np.sum(
                self.masses
                * (conditional_var + (component - mean) ** 2)
            )
        )
        return mean, max(variance, 0.0)

    def hyper_marginal_representation(self, label, grid_size=321):
        """Deterministic density/CDF grid for a scalar posterior marginal."""
        grid_size = max(int(grid_size), 81)
        cache_key = (str(label), grid_size)
        if cache_key in self._hyper_marginal_cache:
            return self._hyper_marginal_cache[cache_key]
        if label == "marginal_log_tau":
            values = self.eta - np.log(TAU_COORD_SCALE)
            log_density = self.log_eta_density.copy()
            masses = self.masses.copy()
        elif label == "marginal_tau2":
            values = np.exp(2.0 * self.eta) / TAU2_COORD_SCALE
            log_density = self.log_eta_density - np.log(2.0 * values)
            masses = self.masses.copy()
        elif label == "marginal_mu":
            conditional_sd = np.sqrt(self.mu_cond_var) / MU_COORD_SCALE
            conditional_mean = self.mu_cond_mean / MU_COORD_SCALE
            mean, variance = self.hyper_marginal_moments(label)
            sd = np.sqrt(max(variance, 1e-12))
            lower = min(
                mean - 8.0 * sd,
                float(np.min(conditional_mean - 8.0 * conditional_sd)),
            )
            upper = max(
                mean + 8.0 * sd,
                float(np.max(conditional_mean + 8.0 * conditional_sd)),
            )
            values = np.linspace(lower, upper, grid_size)
            log_components = (
                np.log(np.maximum(self.masses, 1e-300))[None, :]
                + stats.norm.logpdf(
                    values[:, None],
                    loc=conditional_mean[None, :],
                    scale=conditional_sd[None, :],
                )
            )
            log_density = logsumexp(log_components, axis=1)
            weights = _trap_weights(values)
            log_normalizer = logsumexp(log_density + np.log(weights))
            log_density = log_density - log_normalizer
            masses = np.exp(log_density) * weights
            masses = masses / np.sum(masses)
        else:
            raise ValueError(f"Unknown hyperparameter marginal: {label}")
        cdf = np.cumsum(masses)
        cdf[-1] = 1.0
        result = {
            "label": label,
            "values": np.asarray(values, dtype=float),
            "log_density": np.asarray(log_density, dtype=float),
            "masses": np.asarray(masses, dtype=float),
            "cdf": cdf,
        }
        self._hyper_marginal_cache[cache_key] = result
        return result

    def hyper_marginal_logpdf(self, label, values, grid_size=321):
        """Evaluate a scalar posterior marginal by cached grid interpolation."""
        representation = self.hyper_marginal_representation(
            label, grid_size=grid_size
        )
        values = np.asarray(values, dtype=float)
        result = np.interp(
            values,
            representation["values"],
            representation["log_density"],
            left=-np.inf,
            right=-np.inf,
        )
        return float(result) if values.ndim == 0 else result

    @staticmethod
    def _prior_hyper_log_density(prior, label, values):
        values = np.asarray(values, dtype=float)
        if label == "marginal_mu":
            return (
                prior.log_mu_density(values * MU_COORD_SCALE)
                + np.log(MU_COORD_SCALE)
            )
        if label == "marginal_log_tau":
            eta = values + np.log(TAU_COORD_SCALE)
            return prior.log_eta_density(eta)
        if label == "marginal_tau2":
            valid = values > 0.0
            safe = np.where(valid, values, 1.0)
            eta = 0.5 * np.log(TAU2_COORD_SCALE * safe)
            result = prior.log_eta_density(eta) - np.log(2.0 * safe)
            return np.where(valid, result, -np.inf)
        raise ValueError(f"Unknown hyperparameter marginal: {label}")

    @staticmethod
    def _prior_hyper_cdf(prior, label, values):
        values = np.asarray(values, dtype=float)
        if label == "marginal_mu":
            result = stats.norm.cdf(
                values * MU_COORD_SCALE,
                loc=prior.mu_mean,
                scale=prior.mu_sd,
            )
        elif label == "marginal_log_tau":
            result = prior.eta_cdf(values + np.log(TAU_COORD_SCALE))
        elif label == "marginal_tau2":
            safe = np.maximum(values, 0.0)
            eta = np.where(
                safe > 0.0,
                0.5 * np.log(TAU2_COORD_SCALE * safe),
                -np.inf,
            )
            result = prior.eta_cdf(eta)
        else:
            raise ValueError(f"Unknown hyperparameter marginal: {label}")
        return result

    def marginal_kl_to_prior(self, prior, label, grid_size=321):
        """Deterministic KL(q marginal || informative prior marginal)."""
        representation = self.hyper_marginal_representation(
            label, grid_size=grid_size
        )
        prior_log = self._prior_hyper_log_density(
            prior, label, representation["values"]
        )
        integrand = representation["log_density"] - prior_log
        return float(np.sum(representation["masses"] * integrand))

    def marginal_hdr_geometry(
        self, label, hdr_mass=0.95, grid_size=321, interval_search=65
    ):
        """Grid approximation to a scalar posterior density-level HDR."""
        hdr_mass = float(hdr_mass)
        if not 0.0 < hdr_mass < 1.0:
            raise ValueError("hdr_mass must lie strictly between zero and one")
        representation = self.hyper_marginal_representation(
            label, grid_size=grid_size
        )
        values = representation["values"]
        masses = representation["masses"]
        order = np.argsort(-representation["log_density"], kind="stable")
        allocated = np.zeros_like(masses)
        remaining = hdr_mass
        for index in order:
            if remaining <= 0.0:
                break
            amount = min(float(masses[index]), remaining)
            allocated[index] = amount
            remaining -= amount
        if remaining > 1e-10:
            raise RuntimeError("Marginal HDR grid does not contain requested mass.")
        inside = allocated > 0.0
        selected_mass = float(np.sum(masses[inside]))
        restricted_weights = allocated / hdr_mass

        indices = np.flatnonzero(inside)
        splits = np.flatnonzero(np.diff(indices) > 1) + 1
        groups = np.split(indices, splits)
        intervals = []
        for group in groups:
            first, last = int(group[0]), int(group[-1])
            lower = (
                float(values[0])
                if first == 0
                else float(0.5 * (values[first - 1] + values[first]))
            )
            upper = (
                float(values[-1])
                if last == len(values) - 1
                else float(0.5 * (values[last] + values[last + 1]))
            )
            intervals.append((lower, upper))
        lower = min(value[0] for value in intervals)
        upper = max(value[1] for value in intervals)
        return {
            **representation,
            "lower": lower,
            "upper": upper,
            "hdr_mass_requested": hdr_mass,
            "p_hdr_mass": hdr_mass,
            "grid_selected_mass": selected_mass,
            "restricted_weights": restricted_weights,
            "intervals": intervals,
            "density_cutoff": float(
                np.min(representation["log_density"][inside])
            ),
        }

    @staticmethod
    def marginal_truncated_kl_from_geometry(prior, geometry):
        """KL after restricting a scalar posterior/prior to a P-HDR."""
        label = geometry["label"]
        lower = geometry["lower"]
        upper = geometry["upper"]
        p_mass = float(geometry["p_hdr_mass"])
        prior_cdf = EightSchoolsPosterior._prior_hyper_cdf
        intervals = geometry.get("intervals", [(lower, upper)])
        q_mass = float(sum(
            prior_cdf(prior, label, interval_upper)
            - prior_cdf(prior, label, interval_lower)
            for interval_lower, interval_upper in intervals
        ))
        if q_mass <= 0.0 or not np.isfinite(q_mass):
            return np.inf, {
                "p_hdr_mass": p_mass,
                "q_hdr_mass": q_mass,
                "lower": lower,
                "upper": upper,
            }
        prior_log = EightSchoolsPosterior._prior_hyper_log_density(
            prior, label, geometry["values"]
        )
        log_ratio = geometry["log_density"] - prior_log
        conditional_expectation = float(
            np.sum(geometry["restricted_weights"] * log_ratio)
        )
        value = conditional_expectation + np.log(q_mass / p_mass)
        return float(value), {
            "p_hdr_mass": p_mass,
            "q_hdr_mass": q_mass,
            "lower": lower,
            "upper": upper,
            "grid_selected_mass": geometry["grid_selected_mass"],
        }

    def mean_z(self, parameterization=LOG_PARAMETERIZATION):
        mu = np.sum(self.masses * self.mu_cond_mean) / MU_COORD_SCALE
        if parameterization == LOG_PARAMETERIZATION:
            second = np.sum(self.masses * self.eta) - np.log(TAU_COORD_SCALE)
        elif parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
            second = np.sum(self.masses * np.exp(2.0 * self.eta)) / TAU2_COORD_SCALE
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        return np.asarray([mu, second])

    def covariance_z(self, parameterization=LOG_PARAMETERIZATION):
        """Deterministic posterior covariance of the two joint coordinates."""
        mu = self.mu_cond_mean / MU_COORD_SCALE
        mu_var = self.mu_cond_var / MU_COORD_SCALE**2
        if parameterization == LOG_PARAMETERIZATION:
            second = self.eta - np.log(TAU_COORD_SCALE)
        elif parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
            second = np.exp(2.0 * self.eta) / TAU2_COORD_SCALE
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        mean_mu = float(np.sum(self.masses * mu))
        mean_second = float(np.sum(self.masses * second))
        var_mu = float(
            np.sum(self.masses * (mu_var + (mu - mean_mu) ** 2))
        )
        var_second = float(
            np.sum(self.masses * (second - mean_second) ** 2)
        )
        covariance = float(
            np.sum(self.masses * (mu - mean_mu) * (second - mean_second))
        )
        return np.asarray([[var_mu, covariance], [covariance, var_second]])

    def posterior_mean_hyper(self):
        return np.asarray(
            [
                np.sum(self.masses * self.mu_cond_mean),
                np.sum(self.masses * np.exp(self.eta)),
            ]
        )

    def sample_hyper(
        self, size, rng, standardized=True, parameterization=LOG_PARAMETERIZATION
    ):
        size = int(size)
        if self.eta_draw_scheme == "node":
            idx = rng.choice(len(self.eta), size=size, p=self.masses)
            eta = self.eta[idx]
            mu_mean = self.mu_cond_mean[idx]
            mu_var = self.mu_cond_var[idx]
        else:
            eta = self._invert_piecewise_linear_cdf(rng.random(size))
            mu_mean, mu_var = self._conditional_mu(eta)
        mu = rng.normal(mu_mean, np.sqrt(mu_var))
        if standardized:
            if parameterization == LOG_PARAMETERIZATION:
                second = eta - np.log(TAU_COORD_SCALE)
            elif parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
                second = np.exp(2.0 * eta) / TAU2_COORD_SCALE
            else:
                raise ValueError(f"Unknown parameterization: {parameterization}")
            z = np.column_stack([mu / MU_COORD_SCALE, second])
            return z
        return mu, np.exp(eta), eta

    def sample_theta(self, school_id, size, rng):
        j = int(school_id) - 1
        if not 0 <= j < self.n_schools:
            raise ValueError("school_id must be between 1 and the number of schools")
        mu, tau, _ = self.sample_hyper(size, rng, standardized=False)
        tau2 = tau**2
        sj2 = self.s[j] ** 2
        var = 1.0 / (1.0 / tau2 + 1.0 / sj2)
        mean = var * (mu / tau2 + self.y[j] / sj2)
        return rng.normal(mean, np.sqrt(var))

    def theta_mean(self, school_id):
        j = int(school_id) - 1
        tau2 = np.exp(2.0 * self.eta)
        sj2 = self.s[j] ** 2
        var = 1.0 / (1.0 / tau2 + 1.0 / sj2)
        mean = var * (self.mu_cond_mean / tau2 + self.y[j] / sj2)
        return float(np.sum(self.masses * mean))

    def theta_moments(self, school_id):
        """Return exact quadrature mean and variance of p(theta_j | y)."""
        j = int(school_id) - 1
        if not 0 <= j < self.n_schools:
            raise ValueError("school_id must be between 1 and the number of schools")
        tau2 = np.exp(2.0 * self.eta)
        sj2 = self.s[j] ** 2
        weight_mu = sj2 / (sj2 + tau2)
        weight_y = tau2 / (sj2 + tau2)
        conditional_mean = weight_mu * self.mu_cond_mean + weight_y * self.y[j]
        conditional_var = (
            tau2 * sj2 / (tau2 + sj2)
            + weight_mu**2 * self.mu_cond_var
        )
        mean = float(np.sum(self.masses * conditional_mean))
        variance = float(
            np.sum(
                self.masses
                * (conditional_var + (conditional_mean - mean) ** 2)
            )
        )
        return mean, variance

    def logpdf_z(self, z, parameterization=LOG_PARAMETERIZATION):
        z = np.asarray(z, dtype=float)
        mu = z[..., 0] * MU_COORD_SCALE
        if parameterization == LOG_PARAMETERIZATION:
            eta = z[..., 1] + np.log(TAU_COORD_SCALE)
            log_jacobian = np.log(MU_COORD_SCALE)
        elif parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
            second = z[..., 1]
            valid = second > 0.0
            safe_second = np.where(valid, second, 1.0)
            eta = 0.5 * np.log(TAU2_COORD_SCALE * safe_second)
            log_jacobian = (
                np.log(MU_COORD_SCALE) - np.log(2.0) - np.log(safe_second)
            )
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        tau2 = np.exp(2.0 * eta)
        obs_var = self.s**2 + tau2[..., None]
        log_like = np.sum(
            stats.norm.logpdf(self.y, loc=mu[..., None], scale=np.sqrt(obs_var)),
            axis=-1,
        )
        log_joint = (
            self.prior.log_mu_density(mu)
            + self.prior.log_eta_density(eta)
            + log_like
            - self.log_norm
            + log_jacobian
        )
        if parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
            log_joint = np.where(valid, log_joint, -np.inf)
        return log_joint

    def kl_to_prior(self, prior: PriorSpec):
        """KL(posterior || independent prior), integrated over eta."""
        eta = self.eta
        q_log_eta = self.log_eta_density
        p_log_eta = prior.log_eta_density(eta)
        q_mean = self.mu_cond_mean
        q_var = self.mu_cond_var
        p_var = prior.mu_sd**2
        normal_kl = 0.5 * (
            (q_var + (q_mean - prior.mu_mean) ** 2) / p_var
            - 1.0
            + np.log(p_var / q_var)
        )
        integrand = q_log_eta - p_log_eta + normal_kl
        return float(np.sum(self.masses * integrand))

    def truncated_kl_to_prior(
        self,
        prior: PriorSpec,
        parameterization=LOG_PARAMETERIZATION,
        hdr_mass=0.95,
        bisection_steps=70,
    ):
        """KL between posterior/prior restricted to a posterior HDR.

        The candidate posterior is ``P`` and ``prior`` is ``Q``.  For each
        eta node the P-HDR is an analytic interval in mu.  Integrating its
        probability and quadratic log-density ratio uses truncated-normal
        moments, leaving only deterministic one-dimensional eta quadrature.
        """
        hdr_mass = float(hdr_mass)
        if not 0.0 < hdr_mass < 1.0:
            raise ValueError("hdr_mass must lie strictly between zero and one")
        geometry = self.posterior_hdr_geometry(
            parameterization=parameterization,
            hdr_mass=hdr_mass,
            bisection_steps=bisection_steps,
        )
        return self.truncated_kl_from_geometry(prior, geometry)

    def posterior_hdr_geometry(
        self,
        parameterization=LOG_PARAMETERIZATION,
        hdr_mass=0.95,
        bisection_steps=70,
    ):
        """Precompute a posterior HDR independently of the comparison prior."""
        hdr_mass = float(hdr_mass)
        if not 0.0 < hdr_mass < 1.0:
            raise ValueError("hdr_mass must lie strictly between zero and one")
        eta = self.eta
        weights = self.trap_weights
        p_eta = np.exp(self.log_eta_density)
        mean = self.mu_cond_mean
        variance = self.mu_cond_var
        if parameterization == LOG_PARAMETERIZATION:
            log_jacobian = np.full_like(eta, np.log(MU_COORD_SCALE))
        elif parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
            second = np.exp(2.0 * eta) / TAU2_COORD_SCALE
            log_jacobian = (
                np.log(MU_COORD_SCALE) - np.log(2.0) - np.log(second)
            )
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        peak = (
            self.log_eta_density
            - 0.5 * np.log(2.0 * np.pi * variance)
            + log_jacobian
        )

        def p_mass_at(level):
            radius2 = np.maximum(2.0 * (peak - level), 0.0)
            t = np.sqrt(radius2)
            inside = np.where(peak > level, 2.0 * stats.norm.cdf(t) - 1.0, 0.0)
            return float(np.sum(weights * p_eta * inside))

        high = float(np.max(peak))
        low = float(np.min(peak[np.isfinite(peak)]) - 80.0)
        for _ in range(int(bisection_steps)):
            middle = 0.5 * (low + high)
            if p_mass_at(middle) < hdr_mass:
                high = middle
            else:
                low = middle
        level = 0.5 * (low + high)
        t = np.sqrt(np.maximum(2.0 * (peak - level), 0.0))
        active = peak > level
        h = np.where(active, 2.0 * stats.norm.cdf(t) - 1.0, 0.0)
        phi = stats.norm.pdf(t)
        central_second = variance * np.where(active, h - 2.0 * t * phi, 0.0)
        radius = np.sqrt(variance) * t
        lower = mean - radius
        upper = mean + radius

        return {
            "parameterization": parameterization,
            "hdr_mass_requested": hdr_mass,
            "hdr_level": level,
            "eta": eta,
            "weights": weights,
            "p_eta": p_eta,
            "p_log_eta": self.log_eta_density,
            "mean": mean,
            "variance": variance,
            "h": h,
            "central_second": central_second,
            "lower": lower,
            "upper": upper,
            "p_hdr_mass": float(np.sum(weights * p_eta * h)),
        }

    @staticmethod
    def truncated_kl_from_geometry(prior: PriorSpec, geometry):
        """Score one informative prior against a cached P-HDR geometry."""
        eta = geometry["eta"]
        weights = geometry["weights"]
        p_eta = geometry["p_eta"]
        p_log_eta = geometry["p_log_eta"]
        mean = geometry["mean"]
        variance = geometry["variance"]
        h = geometry["h"]
        central_second = geometry["central_second"]
        lower = geometry["lower"]
        upper = geometry["upper"]
        level = geometry["hdr_level"]
        p_mass = geometry["p_hdr_mass"]
        active = h > 0.0

        q_log_eta = prior.log_eta_density(eta)
        q_eta = np.exp(q_log_eta)
        q_sd = float(prior.mu_sd)
        q_interval = stats.norm.cdf((upper - prior.mu_mean) / q_sd) - stats.norm.cdf(
            (lower - prior.mu_mean) / q_sd
        )
        q_interval = np.where(active, q_interval, 0.0)
        q_mass = float(np.sum(weights * q_eta * q_interval))

        delta = mean - prior.mu_mean
        conditional_log_ratio = (
            h * 0.5 * np.log(prior.mu_sd**2 / variance)
            - 0.5 * central_second / variance
            + 0.5
            * (central_second + delta**2 * h)
            / prior.mu_sd**2
        )
        eta_log_ratio = np.where(
            h > 0.0,
            h * (p_log_eta - q_log_eta),
            0.0,
        )
        integral = float(
            np.sum(weights * p_eta * (eta_log_ratio + conditional_log_ratio))
        )
        if q_mass <= 0.0 or not np.isfinite(q_mass):
            return np.inf, {
                "hdr_level": level,
                "p_hdr_mass": p_mass,
                "q_hdr_mass": q_mass,
            }
        value = integral / p_mass + np.log(q_mass / p_mass)
        return float(value), {
            "hdr_level": level,
            "p_hdr_mass": p_mass,
            "q_hdr_mass": q_mass,
        }

    def shrinkage_diagnostics(self, probs=(0.1, 0.5, 0.9)):
        probs = np.asarray(probs, dtype=float)
        # Both diagnostics are decreasing functions of eta, so their q-th
        # quantile is the transformed (1-q)-th continuous eta quantile.
        eta_for_quantiles = self.eta_quantile(1.0 - probs)
        tau2_for_quantiles = np.exp(2.0 * eta_for_quantiles)
        rows = []
        for j, sj in enumerate(self.s, start=1):
            eq_q = sj**2 / tau2_for_quantiles
            sh_q = sj**2 / (sj**2 + tau2_for_quantiles)
            rows.append(
                {
                    "school_id": j,
                    "y_j": float(self.y[j - 1]),
                    "s_j": float(sj),
                    "equiv_q10": float(eq_q[0]),
                    "equiv_median": float(eq_q[1]),
                    "equiv_q90": float(eq_q[2]),
                    "shrinkage_q10": float(sh_q[0]),
                    "shrinkage_median": float(sh_q[1]),
                    "shrinkage_q90": float(sh_q[2]),
                }
            )
        return rows


def simulate_future_schools(mu, tau, count, rng, s_pool=EIGHT_SCHOOLS_S):
    count = int(count)
    if count <= 0:
        return np.empty(0), np.empty(0)
    s_new = rng.choice(np.asarray(s_pool, dtype=float), size=count, replace=True)
    theta = rng.normal(mu, tau, size=count)
    y_new = rng.normal(theta, s_new)
    return y_new, s_new


def posterior_predictive_path(posterior, max_extra, rng, s_pool=EIGHT_SCHOOLS_S):
    mu, tau, _ = posterior.sample_hyper(1, rng, standardized=False)
    return simulate_future_schools(mu[0], tau[0], max_extra, rng, s_pool=s_pool)


def sliced_wasserstein2(x, y, directions):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    directions = np.asarray(directions, dtype=float)
    n = min(len(x), len(y))
    if n == 0:
        return np.nan
    xp = np.sort(x[:n] @ directions.T, axis=0)
    yp = np.sort(y[:n] @ directions.T, axis=0)
    return float(np.mean((xp - yp) ** 2))


def fixed_directions(n=16):
    angles = np.linspace(0.0, np.pi, int(n), endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def wasserstein_1d(x, y):
    n = min(len(x), len(y))
    return float(np.mean((np.sort(x[:n]) - np.sort(y[:n])) ** 2))


def joint_cp_objective(
    p, q, alpha, size, rng, parameterization=LOG_PARAMETERIZATION
):
    p_draw = p.sample_hyper(
        size, rng, standardized=True, parameterization=parameterization
    )
    q_draw = q.sample_hyper(
        size, rng, standardized=True, parameterization=parameterization
    )
    p_logp = p.logpdf_z(p_draw, parameterization=parameterization)
    q_logq = q.logpdf_z(q_draw, parameterization=parameterization)
    p_cut = float(np.quantile(p_logp, alpha))
    q_cut = float(np.quantile(q_logq, alpha))
    cp_pq = float(
        np.mean(q.logpdf_z(p_draw, parameterization=parameterization) >= q_cut)
    )
    cp_qp = float(
        np.mean(p.logpdf_z(q_draw, parameterization=parameterization) >= p_cut)
    )
    r1, r2 = max(cp_pq, cp_qp), min(cp_pq, cp_qp)
    obj = (r1 - r2) ** 2 + (r2 - (1.0 - alpha)) ** 2
    return cp_pq, cp_qp, r1, r2, float(obj)


def scalar_cp_objective(p_draw, q_draw, alpha):
    p_lo, p_hi = np.quantile(p_draw, [alpha / 2.0, 1.0 - alpha / 2.0])
    q_lo, q_hi = np.quantile(q_draw, [alpha / 2.0, 1.0 - alpha / 2.0])
    cp_pq = float(np.mean((p_draw >= q_lo) & (p_draw <= q_hi)))
    cp_qp = float(np.mean((q_draw >= p_lo) & (q_draw <= p_hi)))
    r1, r2 = max(cp_pq, cp_qp), min(cp_pq, cp_qp)
    obj = (r1 - r2) ** 2 + (r2 - (1.0 - alpha)) ** 2
    return cp_pq, cp_qp, r1, r2, float(obj)


def append_schools(y, s, y_new, s_new):
    return np.concatenate([np.asarray(y), np.asarray(y_new)]), np.concatenate(
        [np.asarray(s), np.asarray(s_new)]
    )


def resample_school_pairs(y, s, size, rng, exclude=None):
    indices = np.arange(len(y))
    if exclude is not None:
        indices = indices[indices != int(exclude)]
    chosen = rng.choice(indices, size=int(size), replace=True)
    return np.asarray(y)[chosen], np.asarray(s)[chosen]


def prior_log_moments(prior: PriorSpec, grid_size=4001):
    """Deterministic mean/variance of (mu/10, log(tau/10))."""
    if prior.tau_kind.lower() == "bounded_uniform":
        mean_eta = np.log(prior.tau_scale) - 1.0 - np.log(TAU_COORD_SCALE)
        mean = np.asarray([prior.mu_mean / MU_COORD_SCALE, mean_eta])
        var = np.asarray([(prior.mu_sd / MU_COORD_SCALE) ** 2, 1.0])
        return mean, var
    eta = np.linspace(-25.0, 20.0, int(grid_size))
    weights = _trap_weights(eta)
    logp = prior.log_eta_density(eta)
    masses = np.exp(logp + np.log(weights) - logsumexp(logp + np.log(weights)))
    shifted = eta - np.log(TAU_COORD_SCALE)
    mean_eta = float(np.sum(masses * shifted))
    var_eta = float(np.sum(masses * (shifted - mean_eta) ** 2))
    mean = np.asarray([prior.mu_mean / MU_COORD_SCALE, mean_eta])
    var = np.asarray([(prior.mu_sd / MU_COORD_SCALE) ** 2, var_eta])
    return mean, var


def prior_coordinate_moments(
    prior: PriorSpec, parameterization=LOG_PARAMETERIZATION, grid_size=4001
):
    """Coordinate-wise prior moments used by multivariate Morita matching.

    Direct-variance moments are returned analytically.  Infinite moments are
    retained as ``np.inf`` so callers can mark Morita undefined rather than
    introduce a hidden numerical truncation.
    """
    if parameterization == LOG_PARAMETERIZATION:
        return prior_log_moments(prior, grid_size=grid_size)
    if parameterization != DIRECT_VARIANCE_PARAMETERIZATION:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    mean_mu = prior.mu_mean / MU_COORD_SCALE
    var_mu = (prior.mu_sd / MU_COORD_SCALE) ** 2
    key = prior.tau_kind.lower()
    if key == "lognormal":
        omega = float(prior.tau_log_sd)
        mean_tau2 = prior.tau_scale**2 * np.exp(2.0 * omega**2)
        var_tau2 = (
            prior.tau_scale**4
            * np.exp(4.0 * omega**2)
            * np.expm1(4.0 * omega**2)
        )
    elif key == "half_t":
        df = float(prior.tau_df)
        mean_tau2 = (
            prior.tau_scale**2 * df / (df - 2.0) if df > 2.0 else np.inf
        )
        if df > 4.0:
            fourth = prior.tau_scale**4 * 3.0 * df**2 / (
                (df - 2.0) * (df - 4.0)
            )
            var_tau2 = fourth - mean_tau2**2
        else:
            var_tau2 = np.inf
    elif key == "half_cauchy":
        mean_tau2 = np.inf
        var_tau2 = np.inf
    elif key == "bounded_uniform":
        upper = float(prior.tau_scale)
        mean_tau2 = upper**2 / 3.0
        var_tau2 = upper**4 * (1.0 / 5.0 - 1.0 / 9.0)
    else:
        raise ValueError(f"Unknown tau prior kind: {prior.tau_kind}")
    mean = np.asarray([mean_mu, mean_tau2 / TAU2_COORD_SCALE])
    var = np.asarray([var_mu, var_tau2 / TAU2_COORD_SCALE**2])
    return mean, var


def log_prior_z(z, prior: PriorSpec, parameterization=LOG_PARAMETERIZATION):
    z = np.asarray(z, dtype=float)
    mu = z[..., 0] * MU_COORD_SCALE
    if parameterization == LOG_PARAMETERIZATION:
        eta = z[..., 1] + np.log(TAU_COORD_SCALE)
        log_jacobian = np.log(MU_COORD_SCALE)
        valid = np.ones(np.shape(eta), dtype=bool)
    elif parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
        second = z[..., 1]
        valid = second > 0.0
        safe_second = np.where(valid, second, 1.0)
        eta = 0.5 * np.log(TAU2_COORD_SCALE * safe_second)
        log_jacobian = (
            np.log(MU_COORD_SCALE) - np.log(2.0) - np.log(safe_second)
        )
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")
    result = (
        prior.log_mu_density(mu)
        + prior.log_eta_density(eta)
        + log_jacobian
    )
    return np.where(valid, result, -np.inf)


def log_unnormalized_posterior_z(
    z, y, s, prior: PriorSpec, parameterization=LOG_PARAMETERIZATION
):
    z = np.asarray(z, dtype=float)
    mu = z[..., 0] * MU_COORD_SCALE
    if parameterization == LOG_PARAMETERIZATION:
        eta = z[..., 1] + np.log(TAU_COORD_SCALE)
    elif parameterization == DIRECT_VARIANCE_PARAMETERIZATION:
        safe_second = np.where(z[..., 1] > 0.0, z[..., 1], 1.0)
        eta = 0.5 * np.log(TAU2_COORD_SCALE * safe_second)
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")
    variance = np.asarray(s) ** 2 + np.exp(2.0 * eta)[..., None]
    return log_prior_z(z, prior, parameterization=parameterization) + np.sum(
        stats.norm.logpdf(np.asarray(y), loc=mu[..., None], scale=np.sqrt(variance)),
        axis=-1,
    )


def negative_diagonal_hessian(func, point, step=1e-4):
    point = np.asarray(point, dtype=float)
    f0 = float(func(point))
    result = np.empty(len(point), dtype=float)
    for j in range(len(point)):
        delta = np.zeros_like(point)
        delta[j] = step
        result[j] = -(float(func(point + delta)) - 2.0 * f0 + float(func(point - delta))) / step**2
    return result


def count_vectors(total, categories):
    """Yield integer count vectors summing to total (with replacement)."""
    total = int(total)
    categories = int(categories)
    if categories == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in count_vectors(total - first, categories - 1):
            yield (first,) + rest


def expand_by_counts(y, s, counts: Iterable[int]):
    counts = np.asarray(tuple(counts), dtype=int)
    return np.repeat(np.asarray(y), counts), np.repeat(np.asarray(s), counts)
