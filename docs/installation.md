# Installation and execution

Use Linux, including WSL on Windows, for the complete workflow. The EPSS task
runner and built-in Stan compilation depend on POSIX `fcntl` file locks;
native Windows execution is not supported. Package installation or import
alone does not establish that the workflows can run. CI checks Linux with
Python 3.10 and 3.12.

Create a virtual environment and install from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Use `pip install -e .` for editable development. The source checkout contains
examples, tables and tools; the wheel contains the Python packages and built-in
Stan source. NumPy 2 provides the trapezoidal integration API used by the kernels.

## Stan/NUTS

```bash
python -m pip install '.[stan]'
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Select the installation with CMDSTAN or inference.cmdstan. Compiled models are
cached under XDG_CACHE_HOME/priorlens, or the default user cache directory.
Compilation uses a POSIX file lock. The package's Stan installation command
above also applies to retained internal error messages mentioning the old name.

## C++ acceleration

```bash
python -m pip install -e '.[fast]'
python tools/build_lupus.py
```

This builds prioreffect._lupus_fast with C++17/OpenMP for the specialized Lupus
engines. The default tutorials use the Python posterior providers.

## Batch execution

```bash
sbatch --account=YOUR_ACCOUNT --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --export=ALL,SPEC=examples/nig/all_methods.yaml,OUTPUT=results/nig_batch \
  tools/run_slurm.sh
```

Use an immutable installed version for each batch. The common runner verifies
the planned numerical runtime when executing tasks.

For task-level scheduling:

```bash
priorlens plan --spec examples/nig/all_methods.yaml
priorlens run-task-group --manifest results/nig_all/manifest.tsv --group-index 0 --group-size 10
priorlens merge --root results/nig_all
```

The group command executes one chunk; schedule the remaining group indices to
cover the complete manifest before merging. Lin groups reuse all-target fits.
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for independent serial workers.
Use `run --resume` to continue a partially completed experiment.

## Stan-only posterior workflow

The Lupus walkthrough uses a Stan model, JSON data and YAML settings:

```bash
priorlens posterior --spec examples/lupus_stan/posterior.yaml --output results/lupus_posterior
```

For IS using the same Stan density, install the BridgeStan interface:

```bash
python -m pip install '.[bridge]'
priorlens posterior --spec examples/lupus_stan/posterior.yaml --backend importance --output results/lupus_posterior_is
```

BridgeStan obtains its C++ sources on first use and compiles the model. On
offline systems, supply an existing source installation through BRIDGESTAN or
inference.bridgestan. Both backends need a C++ toolchain. Each posterior run
builds in its output directory and reuses that build across the prior grid.
