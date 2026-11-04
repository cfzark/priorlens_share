#!/usr/bin/env bash
# Generic worker. Set resources/account with sbatch CLI, not in source code.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SPEC:?Set SPEC to an experiment YAML}"
: "${OUTPUT:?Set OUTPUT to a fresh output directory}"
python -m priorlens run --spec "$SPEC" --output "$OUTPUT"
