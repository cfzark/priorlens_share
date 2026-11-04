"""Frozen, compact repeated-data workflow for the five-by-three NIG experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import yaml

from .core import digest, seed_for, write_json, DefinitionError
from .nig_cp import LOSSES, coverage_grid, summarize

SCHEMES = ("nested_directional", "nested_baseline_directional", "bootstrap_augmentation",
           "bootstrap_deletion", "subsample_deletion")
CURVE_FIELDS = ("loss", "term_gap", "term_coverage", "cp1_used", "cp2_used")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def design(config):
    if "diagnostic_mu" in config:
        dense = config["diagnostic_mu"]
    else:
        dense = np.r_[np.linspace(-4., -1., 31)[:-1], np.linspace(-1., 1., 101)[:-1], np.linspace(1., 4., 31)]
    diag = {round(float(x), 10) for x in dense}
    coarse = {(round(float(mu), 10), float(v)) for mu in config["mu_heatmap"] for v in config["sigma2_prior_means"]}
    points = sorted(coarse | {(mu, config["diagnostic_sigma2_prior_mean"]) for mu in diag})
    return [{"cell_id": i, "mu0": mu, "sigma2_prior_mean": v,
             "heatmap": (mu, v) in coarse,
             "diagnostic": mu in diag and v == config["diagnostic_sigma2_prior_mean"]}
            for i, (mu, v) in enumerate(points)]


def tasks(config):
    return [{"task_id": i, "rep": rep, "sigma2_prior_var": var, "scheme": scheme}
            for i, (rep, var, scheme) in enumerate(
                (rep, var, scheme) for rep in range(config["replications"])
                for var in config["sigma2_prior_variances"] for scheme in config["schemes"])]


def validate_config(c):
    if tuple(c["schemes"]) != SCHEMES or tuple(c["loss_ids"]) != LOSSES:
        raise DefinitionError("The frozen experiment requires exactly five schemes and three losses")
    if c["alpha"] != .05 or c["region"] != "equal_tail" or c["tie"] != "negative" or c["baseline"] != {"family": "jeffreys"}:
        raise DefinitionError("Unsupported experiment definition")
    if c["n"] != 20 or c["paths"] < 2 or c["replications"] < 1:
        raise DefinitionError("Invalid experiment size")
    for key in ("augmentation_grid", "deletion_grid"):
        g = c[key]
        if g != sorted(set(g)) or g[0] != 0 or any(int(m) != m or m<0 for m in g):
            raise DefinitionError("Invalid M grid")
    if max(c["deletion_grid"]) > c["n"] - 3:
        raise DefinitionError("NIG deletion must retain at least three observations")


def prepare(root, config_path, small=False):
    root = Path(root).resolve()
    if root.exists():
        raise DefinitionError("Preserve existing root; use a new root or the saved missing-task manifest")
    c = yaml.safe_load(Path(config_path).read_text())
    if small:
        c.update(replications=1, paths=12, mu_heatmap=[-4., 0., 4.],
                 sigma2_prior_means=[1.], diagnostic_mu=[-4., 0., 4.],
                 augmentation_grid=[0, 1, 3, 20], deletion_grid=[0, 1, 3, 17])
    validate_config(c)
    source = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True)
    package = root / "code" / "src" / "prioreffect_v2"
    package.mkdir(parents=True)
    for p in (source / "src" / "prioreffect_v2").glob("*.py"):
        shutil.copy2(p, package / p.name)
    for folder in ("scripts", "shell", "tests"):
        (root / "code" / folder).mkdir()
    for folder, pattern in (("scripts", "*cp_nig*"), ("shell", "*cp_nig*"), ("tests", "test_cp_nig.py")):
        for p in (source / folder).glob(pattern):
            shutil.copy2(p, root / "code" / folder / p.name)
    shutil.copy2(source/"docs/CP_NIG_SAMPLING_LOSS.md", root/"EXPERIMENT_PLAN.md")
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}
    files = {str(p.relative_to(root)): file_hash(p) for p in (root/"code").rglob("*") if p.is_file()}
    files["EXPERIMENT_PLAN.md"] = file_hash(root/"EXPERIMENT_PLAN.md")
    frozen = {"config": c, "versions": versions, "source_hashes": files}
    frozen["scientific_hash"] = digest(frozen)
    write_json(root/"config.json", frozen)
    pd.DataFrame(design(c)).to_csv(root/"prior_grid.csv", index=False)
    pd.DataFrame(tasks(c)).to_csv(root/"manifest.tsv", sep="\t", index=False)
    (root/"leaves").mkdir()
    print(json.dumps({"root": str(root), "tasks": len(tasks(c)), "cells_per_task": len(design(c)),
                      "result_rows": len(tasks(c))*len(design(c))*3, "scientific_hash": frozen["scientific_hash"]}), flush=True)


def load(root):
    root = Path(root)
    frozen = json.loads((root/"config.json").read_text())
    if frozen["scientific_hash"] != digest({k:v for k,v in frozen.items() if k != "scientific_hash"}):
        raise DefinitionError("Configuration checksum mismatch")
    validate_config(frozen["config"])
    for name, h in frozen["source_hashes"].items():
        if file_hash(root/name) != h:
            raise DefinitionError(f"Frozen source changed: {name}")
    current = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}
    if current != frozen["versions"]:
        raise DefinitionError("Numerical runtime differs from frozen plan")
    manifest = pd.read_csv(root/"manifest.tsv", sep="\t").to_dict("records")
    if manifest != tasks(frozen["config"]):
        raise DefinitionError("Task manifest changed")
    saved_grid = pd.read_csv(root/"prior_grid.csv", float_precision="round_trip").to_dict("records")
    if saved_grid != design(frozen["config"]):
        raise DefinitionError("Prior grid file changed")
    return frozen


def checked_leaf(root, task, frozen):
    leaf = Path(root)/"leaves"/f"task_{task['task_id']:04d}"
    done = json.loads((leaf/"done.json").read_text())
    if done["task"] != task or done["scientific_hash"] != frozen["scientific_hash"]:
        raise DefinitionError("Leaf scientific identity mismatch")
    if set(done["hashes"]) != {"results.csv.gz", "candidate_means.npz", "runtime.json"}:
        raise DefinitionError("Incomplete leaf contract")
    for name, h in done["hashes"].items():
        if file_hash(leaf/name) != h:
            raise DefinitionError(f"Corrupt leaf {leaf/name}")
    return leaf


def run_task(root, task_id):
    root = Path(root)
    frozen = load(root)
    c = frozen["config"]
    task = tasks(c)[task_id]
    leaf = root/"leaves"/f"task_{task_id:04d}"
    if leaf.exists():
        checked_leaf(root, task, frozen)
        print(f"Already verified: {leaf}", flush=True)
        return
    start, cpu = perf_counter(), __import__("time").process_time()
    y = np.random.default_rng(c["data_seed"]+task["rep"]).normal(c["true_mu"], c["true_sigma"], c["n"])
    data = {"y": y.tolist()}
    checksum = digest(data)
    grid = c["deletion_grid"] if task["scheme"].endswith("deletion") else c["augmentation_grid"]
    rows, curve_values, raw_values, diag_ids = [], [], [], []
    timing = {"path_seconds": 0., "coverage_seconds": 0., "bootstrap_redraws": 0, "posterior_updates": 0}
    for i, cell in enumerate(design(c)):
        mean, var = cell["sigma2_prior_mean"], task["sigma2_prior_var"]
        kappa = 2+mean**2/var
        prior = {"family": "nig", "mu0": cell["mu0"], "tau0_sq": c["tau0_sq"], "kappa": kappa, "eta": mean*(kappa-1)}
        seed = seed_for(c["base_seed"], c["experiment"], task["rep"], var, cell["mu0"], mean, task["scheme"])
        cp, bank, times = coverage_grid(data, prior, grid, task["scheme"], c["paths"], seed)
        del bank
        result, curves, loss_times = summarize(cp, grid)
        for key in timing:
            timing[key] += times[key]
        common = times["path_seconds"] + times["coverage_seconds"]
        for r, loss_time in zip(result, loss_times):
            rows.append({**task, **cell, **r, "data_checksum": checksum, "paths": c["paths"],
                         "m_max": max(grid), "loss_seconds": loss_time,
                         "standalone_compute_seconds": common+loss_time,
                         "amortized_compute_seconds": common/3+loss_time})
        if cell["diagnostic"]:
            diag_ids.append(cell["cell_id"])
            curve_values.append(np.stack([np.stack([r[k] for k in CURVE_FIELDS], axis=-1) for r in curves]))
            raw_values.append(cp.mean(axis=0))
        if i % 50 == 0:
            print(f"task={task_id} cell={i+1}/{len(design(c))} elapsed={perf_counter()-start:.1f}s", flush=True)
    stage = Path(tempfile.mkdtemp(prefix=f"stage_{task_id:04d}_", dir=root/"leaves"))
    pd.DataFrame(rows).to_csv(stage/"results.csv.gz", index=False, compression="gzip")
    np.savez_compressed(stage/"candidate_means.npz", cell_ids=diag_ids, m=grid,
                        values=np.stack(curve_values), raw_coverage=np.stack(raw_values))
    runtime = {**task, **timing, "elapsed_seconds": perf_counter()-start,
               "cpu_seconds": __import__("time").process_time()-cpu,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               "loss_seconds": {name: sum(r["loss_seconds"] for r in rows if r["loss_id"] == name) for name in LOSSES},
               "data_checksum": checksum, "rows": len(rows), "engine": c["backend"],
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
    write_json(stage/"runtime.json", runtime)
    write_json(stage/"done.json", {"task": task, "scientific_hash": frozen["scientific_hash"],
                                  "hashes": {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}})
    stage.rename(leaf)
    print(json.dumps(runtime), flush=True)


def merge(root):
    root = Path(root)
    frozen = load(root)
    c = frozen["config"]
    expected_cells = design(c)
    frames, runtimes, curves, checksums, leaf_hashes = [], [], [], {}, {}
    for task in tasks(c):
        leaf = checked_leaf(root, task, frozen)
        frame = pd.read_csv(leaf/"results.csv.gz", float_precision="round_trip")
        required = {(cell["cell_id"], loss) for cell in expected_cells for loss in LOSSES}
        if len(frame) != len(required) or set(zip(frame.cell_id, frame.loss_id)) != required:
            raise DefinitionError("Incomplete or duplicate leaf cells")
        for key, value in task.items():
            if not frame[key].eq(value).all():
                raise DefinitionError("Unexpected task identity in rows")
        for cell in expected_cells:
            part = frame[frame.cell_id == cell["cell_id"]]
            for key in ("mu0", "sigma2_prior_mean", "heatmap", "diagnostic"):
                if not np.all(part[key].to_numpy() == cell[key]):
                    raise DefinitionError("Prior grid metadata mismatch")
        numeric = frame.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy()).all() or not frame.paths.eq(c["paths"]).all():
            raise DefinitionError("Nonfinite or wrong-size results")
        limit = max(c["deletion_grid"] if task["scheme"].endswith("deletion") else c["augmentation_grid"])
        if (frame.epss.abs()>limit).any() or not frame.m_max.eq(limit).all():
            raise DefinitionError("Selected EPSS exceeds candidate support")
        for field in ("pr_positive", "pr_zero", "boundary_rate", "negative_boundary", "positive_boundary"):
            if not frame[field].between(0,1).all():
                raise DefinitionError("Invalid selection probability")
        expected_checksum = digest({"y": np.random.default_rng(c["data_seed"]+task["rep"]).normal(c["true_mu"],c["true_sigma"],c["n"]).tolist()})
        if not frame.data_checksum.eq(expected_checksum).all():
            raise DefinitionError("DGP checksum mismatch")
        checksums[task["rep"]] = expected_checksum
        frames.append(frame)
        runtimes.append(json.loads((leaf/"runtime.json").read_text()))
        leaf_hashes[str(leaf.relative_to(root))] = file_hash(leaf/"done.json")
        grid = c["deletion_grid"] if task["scheme"].endswith("deletion") else c["augmentation_grid"]
        with np.load(leaf/"candidate_means.npz") as archive:
            ids = [cell["cell_id"] for cell in expected_cells if cell["diagnostic"]]
            values, raw = archive["values"], archive["raw_coverage"]
            if archive["cell_ids"].tolist()!=ids or archive["m"].tolist()!=grid or values.shape!=(len(ids),3,2,len(grid),5) or raw.shape!=(len(ids),2,len(grid),2):
                raise DefinitionError("Malformed diagnostic candidate grid")
            if not np.isfinite(values).all() or not np.isfinite(raw).all():
                raise DefinitionError("Nonfinite candidate means")
            if np.any(raw<0) or np.any(raw>1) or np.any(values[...,0]<0):
                raise DefinitionError("Invalid candidate probability/loss")
            if not np.allclose(values[...,0],values[...,1]+values[...,2],atol=2e-13,rtol=0):
                raise DefinitionError("Loss components do not reconstruct mean loss")
            curves.append((task, ids, grid, values.copy(), raw.copy()))
    # All checks above precede publication.
    out = root/"combined"
    out.mkdir(exist_ok=True)
    all_rows = pd.concat(frames, ignore_index=True)
    keys = ["sigma2_prior_var", "scheme", "loss_id", "cell_id", "mu0", "sigma2_prior_mean", "heatmap", "diagnostic"]
    exclude = set(keys+list(tasks(c)[0])+["paths", "m_max"])
    metrics = [k for k in all_rows.select_dtypes(include="number").columns if k not in exclude]
    summary = all_rows.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out/"summary.csv.gz", index=False, compression="gzip")
    # Keep replication summaries only once, in the leaves. Candidate means are
    # reduced across replications here; both means and between-replication SD survive.
    arrays = {}
    for var in c["sigma2_prior_variances"]:
        for si, scheme in enumerate(c["schemes"]):
            group = [r for r in curves if r[0]["sigma2_prior_var"]==var and r[0]["scheme"]==scheme]
            value = np.stack([r[3] for r in group])
            raw = np.stack([r[4] for r in group])
            prefix = f"v{c['sigma2_prior_variances'].index(var)}_s{si}"
            arrays[prefix+"_mean"] = value.mean(axis=0)
            arrays[prefix+"_sd"] = value.std(axis=0, ddof=1) if len(group)>1 else np.zeros_like(value[0])
            arrays[prefix+"_raw_mean"] = raw.mean(axis=0)
            arrays[prefix+"_m"] = group[0][2]
            arrays[prefix+"_mu"] = [expected_cells[i]["mu0"] for i in group[0][1]]
    np.savez_compressed(out/"candidate_summary.npz", **arrays)
    pd.DataFrame(runtimes).drop(columns="loss_seconds").to_csv(out/"runtime.csv", index=False)
    all_rows.groupby(["scheme", "loss_id"])[["standalone_compute_seconds", "amortized_compute_seconds", "loss_seconds"]].agg(["sum", "mean", "median"]).to_csv(out/"timing_comparison.csv")
    write_json(out/"validation.json", {"status": "pass", "scientific_hash": frozen["scientific_hash"],
               "tasks": len(frames), "replications": c["replications"], "result_rows": len(all_rows),
               "summary_rows": len(summary), "cells_per_task": len(expected_cells), "dgp_checksums": checksums,
               "leaf_done_hashes": leaf_hashes, "curve_fields": CURVE_FIELDS,
               "summary_hashes": {name:file_hash(out/name) for name in ("summary.csv.gz", "candidate_summary.npz", "runtime.csv", "timing_comparison.csv")}})
    print(json.dumps({"status": "pass", "rows": len(all_rows), "root": str(root)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "run", "merge", "missing"])
    p.add_argument("--root", required=True)
    p.add_argument("--config")
    p.add_argument("--small", action="store_true")
    p.add_argument("--task-id", type=int)
    a = p.parse_args()
    if a.action=="prepare":
        prepare(a.root, a.config, a.small)
    elif a.action=="run":
        run_task(a.root, a.task_id)
    elif a.action=="merge":
        merge(a.root)
    else:
        f = load(a.root)
        missing = []
        for task in tasks(f["config"]):
            if not (Path(a.root)/"leaves"/f"task_{task['task_id']:04d}"/"done.json").exists():
                missing.append(task)
            else:
                checked_leaf(a.root, task, f)
        pd.DataFrame(missing, columns=list(tasks(f["config"])[0])).to_csv(Path(a.root)/"missing_manifest.tsv", sep="\t", index=False)
        print(",".join(str(t["task_id"]) for t in missing))


if __name__ == "__main__":
    main()
