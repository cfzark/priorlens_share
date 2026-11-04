#!/usr/bin/env bash
# Compatibility with the retained kernel's build-help message.
set -euo pipefail
PRIORLENS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$PRIORLENS_ROOT/tools/build_lupus.py"
