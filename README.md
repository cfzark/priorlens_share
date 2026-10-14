# PriorLens: three case studies

Computation and plotting code accompanying `paper/realdata.tex`.
The four displayed methods are **Reimherr, Wiesenfarth, Jones, CP**.
See [description.md](description.md) for the experimental definitions and
reproducibility scope, and [AGENTS.md](AGENTS.md) for development rules.

## Install

Python 3.10 or newer is required. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The plotting-only workflow needs no compiler, posterior sampler, or original
large result archive. Lupus computation additionally requires a C++17 compiler:

```bash
python compute/build_lupus.py
```

The C++ extension is built for your active Python installation. Do not copy a
compiled extension between Python versions or operating systems. Linux is the
tested computation platform; other platforms are not certified here.

## Reproduce the seven figures

```bash
python plotting/plot_figures.py --output figures
python tests/check_bundle.py
```

This checks the bundled table hashes and writes PDF/PNG pairs under
`figures/{NIG,lupus,eight_school}/`, without fitting any model. Original
figure assets are also included under `paper/V2_figure/realdata/`.
Use a fresh output directory when rerunning.
The current manuscript activates six figures; its NIG v=0.25 figure is
commented out. That optional seventh asset and its source table are retained.

## Small computational examples

```bash
python compute/run_case.py --case midge_real --output results/midge_smoke
python compute/run_case.py --case eight_school_real --output results/schools_smoke
python compute/run_case.py --case lupus_real --output results/lupus_smoke
```

These defaults are **smoke tests**, not publication estimates: one prior cell,
small candidate grids, and very few internal repetitions. They exercise all
four methods and the applicable targets. A successful new run does not assert
bitwise equality with the historical results.

## New production-precision computations

`--size full` uses the supplied production settings. `--cell` selects a row
of the prior grid without renumbering its identity. `--family` optionally
selects `reimherr`, `wiesenfarth`, `jones`, or `cp`.

```bash
# One full-precision cell, one simulated dataset:
python compute/run_case.py --case lupus_simulation --size full --rep 0 --cell 0 --output results/lupus_rep0_cell0
python compute/run_case.py --case eight_school_simulation --size full --rep 0 --cell 0 --output results/schools_rep0_cell0

# A complete real-data prior grid (potentially expensive):
python compute/run_case.py --case midge_real --size full --output results/midge_full
```

Omitting `--cell` in full mode evaluates the entire prior grid. Simulation
commands process **one** observed-data replication: Lupus permits 0--29 and
Eight Schools 0--49. Repeat into different output directories for more datasets.
The portable sequential runner prioritizes inspectability; it does not promise
the throughput of a separately scheduled, shared-cache campaign. It does not
automatically merge different replication directories into a new paper dataset.

## Historical NIG simulation

The original model-specific runners and 40 method/target/variance settings
are retained separately; `configs/nig_simulation/index.csv` lists them.
Identify a setting from its `source_id` and
`configuration`; do not infer its scientific meaning from its number.

```bash
python compute/run_legacy_nig.py --setting configs/nig_simulation/setting_00.json --output results/nig_smoke
python compute/run_legacy_nig.py --setting configs/nig_simulation/setting_00.json --full --rep 0 --output results/nig_rep0 --dry-run
```

Remove `--dry-run` to compute. `--full` without `--rep` requests all 50
datasets. Historical NIG results are not silently replaced by the current
generic implementation. See `description.md` for retained differences.

## Manuscript

Upload `paper/realdata.tex` and `paper/V2_figure/` together, preserving case
and relative paths. No additional TeX input or bibliography file is needed.
Alternatively, run `pdflatex realdata.tex` from `paper/`.

## What is included

- `v2/src/prioreffect_v2/`: composable inference, paths, losses, regions,
  selectors, model adapters, and model-specific accelerators.
- `function/prioreffect/`, `scripts/`: retained numerical engines and NIG runners.
- `cpp/`: source for the Lupus accelerator; no prebuilt binary.
- `compute/`, `configs/`: portable execution entry points and settings.
- `plotting/`, `data/`: plotting code and hash-checked compact source tables.
- `paper/`: standalone manuscript and seven figure pairs.

The numerical package versions used for the included verification are recorded
in `requirements-tested.txt`. No original large posterior/path archives are included.
This folder is not an
assertion of a new software license: the authors should select a license and
confirm redistribution rights before making a public release.
