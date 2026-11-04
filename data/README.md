# Result tables

The bundled real-data tables contain five-method comparisons:
NIG has five targets and R/W mean+median, Schools has four targets and mean risk,
and Lupus has three coefficient targets, both risk reductions and two joint
CP region definitions.

Two Lupus simulation tables retain all-available and common-replication cohorts.
Their uncertainty summaries are between-dataset SD. Numerical warning fields
remain available alongside the plotted values.

manifest.json records source-relative paths and source/export SHA256.
Lupus curve files are byte-identical copies; NIG/Schools exports retain the
numeric curves in compact columns. The plotting script verifies the hashes.
[Case settings](../docs/cases.md) describe the saved comparisons and executable
example recipes.
