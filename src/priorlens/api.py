"""Small stable facade, deliberately without hidden numerical changes."""
from pathlib import Path
from prioreffect_v2.experiments import load_spec, resolved_tasks, prepare, run_task_group, merge


def validate(spec):
    return resolved_tasks(load_spec(spec))


def run(spec, output=None, resume=False):
    settings = load_spec(spec)
    return run_settings(settings, output, resume)


def run_settings(settings, output=None, resume=False):
    """Execute an already loaded specification through the common manifest runner."""
    settings = dict(settings)
    if output is not None:
        settings['output_root'] = str(Path(output).resolve())
    plan = prepare(settings, resume)
    if plan['n_pending']:
        run_task_group(plan['manifest'], list(range(plan['n_pending'])))
    return merge(settings['output_root'])
