# Verification record

Checked on 2026-10-13 with Python 3.10.20 on Linux; numerical versions are
listed in `requirements-tested.txt`.

| Check | Result |
|---|---|
| Source-table SHA256 and full plotted key coverage | Pass: all seven tables |
| Independent redraw using only this package | Pass: seven PDF/PNG pairs |
| Redrawn PNGs against supplied manuscript PNGs | All seven byte-identical in the tested environment |
| Portable C++ build from source | Pass, compiled for the active Python ABI |
| NIG midge real-data smoke | 12/12 tasks, strict merge passes |
| Eight Schools real-data smoke | 16/16 tasks, strict merge passes |
| Eight Schools simulation replication 0 smoke | 16/16 tasks, strict merge passes |
| Lupus real-data smoke | 16/16 selected method/target rows, numerical gates pass |
| Lupus simulation replication 0 smoke | 12/12 selected method/target rows, numerical gates pass |
| Historical NIG simulation smoke | All 20 method/target branches pass at v=0.25 |
| Historical NIG configuration coverage | 40 saved settings: two variances x five targets x four methods |
| Standalone supplied manuscript | Compiles to 12 pages; six active figures, one optional commented figure; no overfull boxes |

The historical NIG checks each used one prior cell, one dataset, and reduced
inner Monte Carlo sizes. The second variance's configurations are included
and checked structurally; that second set was not separately smoke-computed.
The current-model smoke sizes are explicitly set in `compute/run_case.py`.

The bundle check validates finite plotted values, duplicate-free target/method/
prior keys, R=30/R=50 summary counts, manuscript image references, and hashes.
PDF metadata may differ on regeneration; the byte-identical comparison above
refers to PNGs under the tested numerical/plotting environment.

Two integration issues were found and fixed before the passing runs:

- The portable generic wrapper now serializes its execution specification as
  YAML so that scientific-notation tolerances remain numbers when parsed.
- The retained marginal-mu runner has a fixed baseline and does not accept a
  `--baseline` flag; its wrapper no longer sends that unsupported argument.

These were wrapper corrections; no posterior, sampling, loss, or selection
implementation was changed. The retained package's environment-specific
docstring wording was also made neutral without changing computation.

No full-scale simulations were rerun for this export. Smoke success does not
certify risk Monte Carlo stability or bitwise equivalence of newly computed
EPSSs to the historical scientific results. The original large diagnostics
archives are not distributed. Build products and temporary smoke outputs are
excluded from the share folder.
