"""Shared design objects for the R=50 Eight Schools simulation.

This module deliberately contains no command-line or storage logic.  It is
used by both method groups so the independently generated DGP
tables and prior grids are byte-for-byte reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import pandas as pd

from .eight_schools import (
    EIGHT_SCHOOLS_S,
    PriorSpec,
    baseline_prior,
    bounded_uniform_baseline,
)


SIMULATION_ROOT = "results/eight_schools_simulation_mu8_tau6_fixeds_v1"
MU_TRUE = 8.0
TAU_TRUE = 6.0
DATA_SEED = 321
BASE_SEED = 20260312
N_REPLICATIONS = 50

MEAN_OFFSETS = (-24.0, -16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0, 24.0)
MEAN_SDS = (2.5, 5.0, 10.0, 20.0)
TAU_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
TAU_LOG_SDS = (0.2, 0.35, 0.5, 0.75, 1.0)
COMBINED_MEAN_OFFSETS = (-16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0)
COMBINED_TAU_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)

# Fine design: dense enough to reveal continuous structure while leaving a
# conservative margin under the 48-hour, 20-cores-per-block allocation.
FINE_MEAN_OFFSETS = tuple(np.round(np.arange(-24.0, 24.0 + 0.1, 2.0), 10))
FINE_MEAN_SDS = tuple(np.round(2.5 * 2.0 ** (np.arange(10) / 3.0), 10))
FINE_TAU_RATIOS = tuple(np.round(2.0 ** np.arange(-2.0, 2.0 + 0.01, 0.25), 10))
FINE_TAU_LOG_SDS = tuple(np.round(np.arange(0.2, 1.0 + 0.01, 0.1), 10))
FINE_COMBINED_MEAN_OFFSETS = tuple(np.round(np.arange(-16.0, 16.0 + 0.1, 2.0), 10))
FINE_COMBINED_TAU_RATIOS = tuple(np.round(2.0 ** np.arange(-2.0, 2.0 + 0.01, 0.5), 10))

BASELINE_IDS = ("half_t4_25", "bounded_uniform_0_100")
PRIOR_BLOCKS = (
    "mean_conflict",
    "heterogeneity_conflict",
    "mean_heterogeneity_conflict",
)


def _slug(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def simulation_baselines():
    half_t = baseline_prior()
    half_t = PriorSpec(**{**asdict(half_t), "prior_id": "half_t4_25"})
    return {
        "half_t4_25": half_t,
        "bounded_uniform_0_100": bounded_uniform_baseline(100.0),
    }


def simulation_prior_grid(block="all", smoke=False, design="standard"):
    """Return a standard or fine truth-centred informative-prior grid."""
    block = str(block).lower()
    design = str(design).lower()
    if design not in {"standard", "fine"}:
        raise ValueError("design must be standard or fine")
    if block not in {"all", *PRIOR_BLOCKS}:
        raise ValueError(f"unknown prior block: {block}")
    priors = []
    if block in {"all", "mean_conflict"}:
        offsets = (0.0, 16.0) if smoke else (
            FINE_MEAN_OFFSETS if design == "fine" else MEAN_OFFSETS
        )
        sds = (5.0, 20.0) if smoke else (
            FINE_MEAN_SDS if design == "fine" else MEAN_SDS
        )
        for offset in offsets:
            for sd in sds:
                location = MU_TRUE + offset
                priors.append(
                    PriorSpec(
                        prior_id=f"mean_delta_{_slug(offset)}_sd_{_slug(sd)}",
                        prior_block="mean_conflict",
                        mu_mean=location,
                        mu_sd=sd,
                        tau_kind="half_t",
                        tau_scale=25.0,
                        tau_df=4.0,
                        grid_x_name="mu_prior_offset",
                        grid_x_value=offset,
                        grid_y_name="mu_prior_sd",
                        grid_y_value=sd,
                    )
                )
    if block in {"all", "heterogeneity_conflict"}:
        ratios = (1.0, 2.0) if smoke else (
            FINE_TAU_RATIOS if design == "fine" else TAU_RATIOS
        )
        log_sds = (0.35, 0.75) if smoke else (
            FINE_TAU_LOG_SDS if design == "fine" else TAU_LOG_SDS
        )
        for ratio in ratios:
            for log_sd in log_sds:
                median = TAU_TRUE * ratio
                priors.append(
                    PriorSpec(
                        prior_id=f"heterogeneity_ratio_{_slug(ratio)}_logsd_{_slug(log_sd)}",
                        prior_block="heterogeneity_conflict",
                        mu_mean=0.0,
                        mu_sd=25.0,
                        tau_kind="lognormal",
                        tau_scale=median,
                        tau_log_sd=log_sd,
                        grid_x_name="tau_prior_ratio",
                        grid_x_value=ratio,
                        grid_y_name="tau_prior_log_sd",
                        grid_y_value=log_sd,
                    )
                )
    if block in {"all", "mean_heterogeneity_conflict"}:
        offsets = (0.0, 12.0) if smoke else (
            FINE_COMBINED_MEAN_OFFSETS
            if design == "fine" else COMBINED_MEAN_OFFSETS
        )
        ratios = (1.0, 2.0) if smoke else (
            FINE_COMBINED_TAU_RATIOS
            if design == "fine" else COMBINED_TAU_RATIOS
        )
        for offset in offsets:
            for ratio in ratios:
                location = MU_TRUE + offset
                scale = TAU_TRUE * ratio
                priors.append(
                    PriorSpec(
                        prior_id=f"meanhet_delta_{_slug(offset)}_ratio_{_slug(ratio)}",
                        prior_block="mean_heterogeneity_conflict",
                        mu_mean=location,
                        mu_sd=5.0,
                        tau_kind="half_cauchy",
                        tau_scale=scale,
                        grid_x_name="mu_prior_offset",
                        grid_x_value=offset,
                        grid_y_name="tau_prior_scale_ratio",
                        grid_y_value=ratio,
                    )
                )
    return priors


def simulate_replication(rep, data_seed=DATA_SEED):
    rep = int(rep)
    rng = np.random.default_rng(int(data_seed) + rep)
    theta = rng.normal(MU_TRUE, TAU_TRUE, size=len(EIGHT_SCHOOLS_S))
    y = rng.normal(theta, EIGHT_SCHOOLS_S)
    return theta, y, EIGHT_SCHOOLS_S.copy()


def dgp_frame(replications=range(N_REPLICATIONS), data_seed=DATA_SEED):
    rows = []
    for rep in replications:
        theta, y, s = simulate_replication(rep, data_seed=data_seed)
        for school, (theta_j, y_j, s_j) in enumerate(zip(theta, y, s), start=1):
            rows.append(
                {
                    "rep": int(rep),
                    "data_seed": int(data_seed) + int(rep),
                    "school_id": school,
                    "theta_true": float(theta_j),
                    "y": float(y_j),
                    "s": float(s_j),
                    "mu_true": MU_TRUE,
                    "tau_true": TAU_TRUE,
                }
            )
    return pd.DataFrame(rows)


def dgp_checksum(frame):
    ordered = frame.sort_values(["rep", "school_id"]).reset_index(drop=True)
    # Twelve decimal places are far below any scientifically relevant scale
    # here and survive a CSV round trip identically on both accounts.
    numeric = ordered.select_dtypes(include=[np.number]).columns
    ordered = ordered.copy()
    ordered[numeric] = ordered[numeric].round(12)
    payload = ordered.to_csv(index=False, float_format="%.12f").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prior_row(prior):
    return {
        "prior_id": prior.prior_id,
        "prior_block": prior.prior_block,
        "mu_prior_mean": prior.mu_mean,
        "mu_prior_sd": prior.mu_sd,
        "tau_prior_kind": prior.tau_kind,
        "tau_prior_scale": prior.tau_scale,
        "tau_prior_df": prior.tau_df,
        "tau_prior_log_sd": prior.tau_log_sd,
        "grid_x_name": prior.grid_x_name,
        "grid_x_value": prior.grid_x_value,
        "grid_y_name": prior.grid_y_name,
        "grid_y_value": prior.grid_y_value,
    }
