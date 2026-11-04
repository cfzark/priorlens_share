"""Frozen design for the lupus-style logistic location simulation.

The predictor positions are the reconstructed 55-patient lupus design.  Only
the Bernoulli outcomes are regenerated.  The complete slope-location grid has
quarter-unit spacing and a stable ``q`` index; phase 1 evaluates even ``q``
only, so later odd-q refinement cannot change any existing key or seed.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from .lupus_logistic import lupus_patient_data


EXPERIMENT_ID = "lupus_logistic_simulation_location_v1"
DATA_SEED = 20260817
BASE_SEED = 20260312
PLANNED_REPLICATIONS = 50
PHASE1_REPLICATIONS = 30
TRUE_BETA0 = -1.0
TRUE_BETA1 = 3.0
PRIOR_BETA0_VALUES = np.arange(-3.0, 3.0 + 0.25, 0.5)
PRIOR_BETA1_MIN = -5.0
PRIOR_BETA1_MAX = 11.0
PRIOR_BETA1_STEP = 0.25
SLOPE_SCALE = 2.5
Q_VALUES = np.arange(
    int(round((PRIOR_BETA1_MAX - PRIOR_BETA1_MIN) / PRIOR_BETA1_STEP)) + 1,
    dtype=int,
)
EVEN_Q_VALUES = Q_VALUES[Q_VALUES % 2 == 0]


def stable_seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def simulation_prior_grid(q_parity: str = "even") -> list[dict]:
    """Return stable full-grid keys and a contiguous phase-grid index."""
    q_parity = str(q_parity)
    if q_parity == "even":
        selected_q = EVEN_Q_VALUES
    elif q_parity == "odd":
        selected_q = Q_VALUES[Q_VALUES % 2 == 1]
    elif q_parity == "all":
        selected_q = Q_VALUES
    else:
        raise ValueError("q_parity must be even, odd, or all.")
    rows = []
    phase_index = 0
    for q in selected_q:
        mu_beta1 = PRIOR_BETA1_MIN + int(q) * PRIOR_BETA1_STEP
        for beta0_index, mu_beta0 in enumerate(PRIOR_BETA0_VALUES):
            rows.append(
                {
                    "grid_index": int(q) * len(PRIOR_BETA0_VALUES)
                    + int(beta0_index),
                    "phase_grid_index": phase_index,
                    "q": int(q),
                    "q_parity": "even" if int(q) % 2 == 0 else "odd",
                    "beta0_index": int(beta0_index),
                    "mu_beta0": float(mu_beta0),
                    "mu_beta1": float(mu_beta1),
                    "s_beta1": float(SLOPE_SCALE),
                }
            )
            phase_index += 1
    return rows


def grid_checksum(q_parity: str = "even") -> str:
    payload = json.dumps(
        simulation_prior_grid(q_parity), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def simulate_replication(rep: int, data_seed: int = DATA_SEED) -> dict:
    rep = int(rep)
    if not 0 <= rep < PLANNED_REPLICATIONS:
        raise ValueError(
            f"rep must lie in [0, {PLANNED_REPLICATIONS}); got {rep}."
        )
    design = lupus_patient_data()
    probability = expit(TRUE_BETA0 + TRUE_BETA1 * design["x_std"])
    rng = np.random.default_rng(stable_seed(data_seed, "outcomes", rep))
    y = rng.binomial(1, probability).astype(int)
    return {
        "x": design["x"].copy(),
        "x_std": design["x_std"].copy(),
        "y": y,
        "x_mean": float(design["x_mean"]),
        "x_sd": float(design["x_sd"]),
        "probability": probability,
        "rep": rep,
        "data_seed": int(data_seed),
        "rep_seed": stable_seed(data_seed, "outcomes", rep),
    }


def replication_checksum(rep: int, data_seed: int = DATA_SEED) -> str:
    data = simulate_replication(rep, data_seed=data_seed)
    hasher = hashlib.sha256()
    hasher.update(np.asarray(data["x_std"], dtype="<f8").tobytes())
    hasher.update(np.asarray(data["y"], dtype="<i8").tobytes())
    return hasher.hexdigest()


def dgp_frame(
    n_replications: int = PLANNED_REPLICATIONS,
    data_seed: int = DATA_SEED,
) -> pd.DataFrame:
    rows = []
    for rep in range(int(n_replications)):
        data = simulate_replication(rep, data_seed=data_seed)
        rows.append(
            {
                "rep": rep,
                "rep_seed": int(data["rep_seed"]),
                "n": len(data["y"]),
                "n_cases": int(np.sum(data["y"])),
                "sample_rate": float(np.mean(data["y"])),
                "replication_checksum": replication_checksum(rep, data_seed),
            }
        )
    return pd.DataFrame(rows)


def dgp_checksum(
    n_replications: int = PLANNED_REPLICATIONS,
    data_seed: int = DATA_SEED,
) -> str:
    frame = dgp_frame(n_replications, data_seed=data_seed)
    payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
