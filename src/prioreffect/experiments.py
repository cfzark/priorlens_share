import json
import time
from pathlib import Path

import numpy as np

try:
    from .distributions import FlatPrior, JeffreysPrior
except ImportError:  # pragma: no cover
    from distributions import FlatPrior, JeffreysPrior


def default_poisson_prior_lambda_grid():
    return np.concatenate([
        np.linspace(0.5, 2.0, 31)[:-1],
        np.linspace(2.0, 4.0, 101)[:-1],
        np.linspace(4.0, 8.0, 41),
    ])


def default_normal_prior_loc_grid():
    return np.concatenate([
        np.linspace(-4.0, -1.0, 31)[:-1],
        np.linspace(-1.0, 1.0, 101)[:-1],
        np.linspace(1.0, 4.0, 31),
    ])


def default_beta_prior_mean_grid():
    return np.concatenate([
        np.linspace(0.05, 0.20, 16)[:-1],
        np.linspace(0.20, 0.40, 41)[:-1],
        np.linspace(0.40, 0.95, 36),
    ])


def make_baseline_prior(name):
    key = str(name).lower()
    if key == "flat":
        return FlatPrior()
    if key == "jeffreys":
        return JeffreysPrior()
    raise ValueError("baseline must be one of: flat, jeffreys")


def start_timer():
    return time.perf_counter()


def add_runtime_metadata(df, summary, config, start_time, variant=None):
    elapsed = time.perf_counter() - start_time
    n_rows = int(len(df))
    runtime = {
        "method": config.get("method"),
        "runtime_seconds": float(elapsed),
        "runtime_minutes": float(elapsed / 60.0),
        "n_rows": n_rows,
        "seconds_per_row": float(elapsed / n_rows) if n_rows else None,
    }
    if variant is not None:
        runtime["variant"] = str(variant)

    config = {**config, "runtime": runtime}
    summary = summary.copy()
    summary["runtime_seconds_total"] = runtime["runtime_seconds"]
    summary["seconds_per_row"] = runtime["seconds_per_row"]
    return config, summary, runtime


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_experiment_outputs(
    df,
    summary,
    out_dir,
    config,
    grid_filename=None,
    grid_values=None,
    save_json=False,
    save_pickle=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "results.csv", index=False)
    if save_json:
        df.to_json(out_dir / "results.json", orient="records")
    if save_pickle:
        df.to_pickle(out_dir / "results.pkl")

    summary.to_csv(out_dir / "summary.csv", index=False)
    if save_json:
        summary.to_json(out_dir / "summary.json", orient="records")
    if save_pickle:
        summary.to_pickle(out_dir / "summary.pkl")

    if grid_filename is not None and grid_values is not None:
        np.savetxt(out_dir / grid_filename, np.asarray(grid_values), delimiter=",")

    config_to_save = _json_ready(config)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    runtime = config_to_save.get("runtime")
    if runtime is not None:
        with open(out_dir / "runtime.json", "w", encoding="utf-8") as f:
            json.dump(runtime, f, indent=2)
