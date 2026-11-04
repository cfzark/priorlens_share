"""Convenience facade plus access to the retained validated manifest CLI."""
import argparse
import json
import sys
from .api import run, validate
from prioreffect_v2.core import native


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {'plan', 'run-task', 'run-task-group', 'merge', 'calibrate-cp', 'validate-spec'}:
        from prioreffect_v2.cli import main as kernel_cli
        return kernel_cli()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('run', 'validate', 'lin', 'risk-summary', 'posterior'):
        c = sub.add_parser(name)
        c.add_argument('--spec', required=True)
        if name != 'validate':
            c.add_argument('--output', required=True, help='Fresh output directory')
        if name == 'run':
            c.add_argument('--resume', action='store_true')
        if name == 'posterior':
            c.add_argument('--backend', choices=['importance', 'general'], help='Override the posterior backend')
    a = p.parse_args()
    if a.command == 'validate':
        import yaml
        from pathlib import Path
        if yaml.safe_load(Path(a.spec).read_text()).get('workflow') == 'posterior':
            from .stan_posterior import validate_posterior
            result = validate_posterior(a.spec)
        else:
            tasks = validate(a.spec)
            result = {'tasks': len(tasks), 'status': 'valid'}
    elif a.command == 'posterior':
        from .stan_posterior import posterior
        result = posterior(a.spec, a.output, a.backend)
    elif a.command == 'run':
        result = run(a.spec, a.output, a.resume)
    elif a.command == 'lin':
        from .workflows import run_lin
        result = run_lin(a.spec, a.output)
    else:
        from .workflows import run_risk_summary
        result = run_risk_summary(a.spec, a.output)
    print(json.dumps(native(result), indent=2))
