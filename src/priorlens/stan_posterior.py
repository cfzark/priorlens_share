"""Posterior-only Stan workflow: one model definition, no Python model adapter."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats
from scipy.stats import qmc
import yaml

from prioreffect_v2.core import DefinitionError, NumericalFailure, TargetSpec, digest, write_json, weighted_quantile
from prioreffect_v2.inference import weight_diagnostics


def _keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise DefinitionError(f"Unexpected {label} fields; accepted: {sorted(allowed)}")


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DefinitionError(f"{label} must be a positive integer")


def load_posterior_spec(path):
    """Resolve the small posterior schema without importing optional backends."""
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    _keys(spec, {'schema_version', 'workflow', 'model', 'data', 'priors', 'prior_grid',
                 'targets', 'inference', 'seed'}, 'posterior specification')
    if spec.get('schema_version') != 1 or spec.get('workflow') != 'posterior':
        raise DefinitionError('Use schema_version: 1 and workflow: posterior')
    _keys(spec.get('model'), {'stan'}, 'model')
    stan = (path.parent / spec['model']['stan']).resolve()
    if not stan.is_file() or stan.suffix != '.stan':
        raise DefinitionError('model.stan must name an existing .stan file')
    # A self-contained model makes the frozen source unambiguous.
    if re.search(r'^\s*#include\b', stan.read_text(), re.MULTILINE):
        raise DefinitionError('The posterior entry accepts a self-contained Stan file; inline its includes')
    data = spec.get('data')
    if isinstance(data, str):
        data = json.loads((path.parent / data).read_text())
    if not isinstance(data, dict):
        raise DefinitionError('data must be a JSON filename or a Stan-data mapping')
    _keys(spec.get('priors'), {'baseline', 'interest'}, 'priors')
    if set(spec['priors']) != {'baseline', 'interest'}:
        raise DefinitionError('Provide baseline and interest Stan hyperparameter mappings')
    baseline, interest = spec['priors']['baseline'], spec['priors']['interest']
    if not isinstance(baseline, dict) or not baseline or not isinstance(interest, dict) or set(baseline) != set(interest):
        raise DefinitionError('baseline and interest must supply the same nonempty set of Stan input fields')
    if set(data) & set(baseline):
        raise DefinitionError('Keep prior hyperparameters out of the observation data mapping')
    grid = spec.get('prior_grid', [{'prior_id': 'interest'}])
    if not isinstance(grid, list) or not grid:
        raise DefinitionError('prior_grid must be a nonempty list')
    ids = []
    for row in grid:
        _keys(row, set(interest) | {'prior_id'}, 'prior_grid row')
        if not isinstance(row.get('prior_id'), str) or not row['prior_id']:
            raise DefinitionError('Each prior_grid row requires a nonempty string prior_id')
        ids.append(row['prior_id'])
    if len(ids) != len(set(ids)) or 'baseline' in ids:
        raise DefinitionError('prior_id values must be unique and different from baseline')
    targets = spec.get('targets')
    if not isinstance(targets, list) or not targets:
        raise DefinitionError('Provide one or more scalar/joint targets')
    parsed = []
    for t in targets:
        _keys(t, {'id', 'coordinates'}, 'target')
        parsed.append(TargetSpec.parse(t))
    if len({t.id for t in parsed}) != len(parsed):
        raise DefinitionError('Target IDs must be unique')
    inference = spec.get('inference', {})
    _keys(inference, {'backend', 'sobol_power', 'is_min_ess', 'initial_unconstrained',
                     'chains', 'parallel_chains', 'warmup', 'sampling', 'adapt_delta',
                     'max_treedepth', 'stan_min_ess', 'cmdstan', 'bridgestan'}, 'inference')
    if inference.get('backend') not in {'importance', 'general'}:
        raise DefinitionError('Stan posterior backend must be importance or general (NUTS)')
    for key in ('sobol_power', 'is_min_ess', 'chains', 'parallel_chains', 'warmup',
                'sampling', 'max_treedepth', 'stan_min_ess'):
        if key in inference:
            _positive_int(inference[key], key)
    if 'sobol_power' in inference and not 5 <= inference['sobol_power'] <= 24:
        raise DefinitionError('sobol_power must lie in 5:24')
    if 'adapt_delta' in inference and (isinstance(inference['adapt_delta'], bool) or
            not isinstance(inference['adapt_delta'], (int, float)) or not 0 < inference['adapt_delta'] < 1):
        raise DefinitionError('adapt_delta must lie in (0,1)')
    if 'initial_unconstrained' in inference:
        z = np.asarray(inference['initial_unconstrained'], float)
        if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
            raise DefinitionError('initial_unconstrained must be a finite vector')
    for key in ('cmdstan', 'bridgestan'):
        if key in inference:
            inference[key] = str((path.parent / inference[key]).resolve())
    seed = spec.get('seed', 20260312)
    _positive_int(seed, 'seed')
    if seed >= 2**32:
        raise DefinitionError('seed must be smaller than 2^32')
    spec.update(model={'stan': str(stan)}, data=data, prior_grid=grid, seed=seed)
    # Reject nonfinite values before they can be silently converted in JSON output.
    json.dumps(spec, allow_nan=False)
    return spec


def validate_posterior(spec):
    settings = load_posterior_spec(spec)
    return {'status': 'valid', 'workflow': 'posterior',
            'posterior_fits': 1 + len(settings['prior_grid']),
            'targets_per_fit': len(settings['targets']), 'backend': settings['inference']['backend']}


def indexed_name(name):
    """BridgeStan beta.1.2 -> public Stan notation beta[1,2]."""
    parts = name.split('.')
    return parts[0] + '[' + ','.join(parts[1:]) + ']' if len(parts) > 1 else name


def _check_target_names(targets, names):
    missing = {c.variable for t in targets for c in t.coordinates} - set(names)
    if missing:
        raise DefinitionError(f'Targets not found in Stan parameters: {sorted(missing)}; available: {names}')


def _proposal(model, settings):
    d = model.param_unc_num()
    if d < 1:
        raise DefinitionError('Stan model must contain continuous parameters')
    start = np.asarray(settings.get('initial_unconstrained', np.zeros(d)), float)
    if start.shape != (d,):
        raise DefinitionError(f'initial_unconstrained must have length {d}')

    def objective(z):
        lp, grad = model.log_density_gradient(z, propto=True, jacobian=True)
        return -lp, -grad

    fit = optimize.minimize(objective, start, jac=True, method='BFGS',
                            options={'gtol': 1e-7, 'maxiter': 2000})
    lp, grad, hess = model.log_density_hessian(fit.x, propto=True, jacobian=True)
    norm = float(np.max(np.abs(grad)))
    if not np.isfinite(lp) or not np.isfinite(norm) or norm > 1e-5:
        raise NumericalFailure(f'Stan proposal mode did not converge: gradient={norm}, {fit.message}')
    precision = -(hess + hess.T) / 2
    eig = np.linalg.eigvalsh(precision)
    if not np.isfinite(eig).all() or eig.min() <= 0:
        raise NumericalFailure('Stan proposal requires positive definite negative log-density curvature')
    covariance = np.linalg.inv(precision)
    return fit.x, covariance, {'center': fit.x, 'local_covariance': covariance,
                              'mode_gradient_max': norm, 'optimizer_success': bool(fit.success),
                              'optimizer_message': str(fit.message), 'jacobian': True,
                              'family': '0.75*t5(mode,4cov)+0.25*t5(mode,16cov)',
                              'matrix_convention': 'Student-t shape, not covariance'}


def importance_draws(model, settings, seed):
    """Universal proposal on Stan unconstrained coordinates; raw IS weights."""
    center, covariance, proposal_info = _proposal(model, settings)
    power = settings['sobol_power']
    d = len(center)
    u = np.clip(qmc.Sobol(d=d + 2, scramble=True, seed=seed).random_base2(power), 1e-12, 1-1e-12)
    component = (u[:, 0] >= .75).astype(int)
    z = stats.norm.ppf(u[:, 1:d+1]) / np.sqrt(stats.chi2.ppf(u[:, -1], 5)[:, None] / 5)
    proposal = np.empty_like(z)
    shapes = [4*covariance, 16*covariance]
    for k in (0, 1):
        ix = component == k
        proposal[ix] = center + z[ix] @ np.linalg.cholesky(shapes[k]).T
    log_q = special.logsumexp(np.column_stack([
        np.log(p) + stats.multivariate_t.logpdf(proposal, center, shape, df=5)
        for p, shape in zip((.75, .25), shapes)]), axis=1)
    log_target = np.array([model.log_density(z, propto=True, jacobian=True) for z in proposal])
    weights, diagnostics = weight_diagnostics(log_target - log_q, proposal,
                                               min_ess=settings.get('is_min_ess', 500))
    natural = np.array([model.param_constrain(z, include_tp=False, include_gq=False) for z in proposal])
    if not np.isfinite(natural).all():
        raise NumericalFailure('Nonfinite natural-coordinate Stan parameter values')
    diagnostics.update(engine='stan_bridgestan_importance', sobol_power=power,
                       seed=seed, proposal=proposal_info)
    return natural, weights, diagnostics, {'unconstrained': proposal,
                                          'log_weight_raw': log_target-log_q}


def target_summaries(points, weights, names, targets, prior_id):
    _check_target_names(targets, names)
    draws = dict(zip(names, points.T))
    rows, covariance = [], {}
    for target in targets:
        values = target.values(draws)
        if not np.isfinite(values).all():
            raise NumericalFailure(f'Nonfinite target values: {target.id}')
        mean = weights @ values
        centered = values-mean
        cov = (centered.T * weights) @ centered
        covariance[target.id] = cov.tolist()
        for j, c in enumerate(target.coordinates):
            quantiles = weighted_quantile(values[:, j], [.025, .5, .975], weights)
            rows.append(dict(prior_id=prior_id, target_id=target.id, variable=c.variable,
                             transform=c.transform, scale=c.scale, reference=c.reference,
                             mean=float(mean[j]), sd=float(np.sqrt(max(cov[j, j], 0))),
                             q025=float(quantiles[0]), median=float(quantiles[1]), q975=float(quantiles[2])))
    return rows, covariance


def _nuts_draws(stan, payload, settings, seed, directory, targets):
    info = stan.src_info()
    parameters = set(info.get('parameters', {}))
    if not parameters:
        raise DefinitionError('Stan compiler parameter metadata is required (CmdStan >=2.27)')
    directory.mkdir()
    fit = stan.sample(data=payload, seed=seed, chains=settings['chains'],
                      parallel_chains=settings.get('parallel_chains', 1),
                      iter_warmup=settings['warmup'], iter_sampling=settings['sampling'],
                      adapt_delta=settings.get('adapt_delta', .95),
                      max_treedepth=settings.get('max_treedepth', 12),
                      output_dir=str(directory), show_progress=False, show_console=False)
    names = [n for n in fit.column_names if n.split('[')[0] in parameters]
    _check_target_names(targets, names)
    ix = [fit.column_names.index(n) for n in names]
    chain_draws = fit.draws()[:, :, ix]
    points = chain_draws.reshape(-1, len(names), order='F')
    summary = fit.summary().loc[names]
    summary.to_csv(directory / 'parameter_diagnostics.csv')
    rhat, bulk, tail = (float(summary[k].max() if k == 'R_hat' else summary[k].min())
                        for k in ('R_hat', 'ESS_bulk', 'ESS_tail'))
    divergences = int(fit.method_variables()['divergent__'].sum())
    treedepths = int((fit.method_variables()['treedepth__'] >= settings.get('max_treedepth', 12)).sum())
    converged = (np.isfinite([rhat, bulk, tail]).all() and rhat <= 1.01 and
                 min(bulk, tail) >= settings.get('stan_min_ess', 400) and divergences == 0
                 and np.isfinite(points).all())
    accuracy = dict(engine='stan_nuts', converged=bool(converged), rhat=rhat,
                    ess_bulk=bulk, ess_tail=tail, divergences=divergences,
                    max_treedepth_hits=treedepths, seed=seed)
    return points, np.full(len(points), 1/len(points)), names, accuracy, {'chain_draws': chain_draws}


def posterior(spec, output, backend=None):
    """Fit baseline once plus each interest-prior cell, sharing all targets."""
    settings = load_posterior_spec(spec)
    inference = settings['inference']
    if backend is not None:
        if backend not in {'importance', 'general'}:
            raise DefinitionError('backend must be importance or general')
        inference['backend'] = backend
    engine = inference['backend']
    required = ('sobol_power',) if engine == 'importance' else ('chains', 'warmup', 'sampling')
    if any(k not in inference for k in required):
        raise DefinitionError(f'Explicit inference settings required: {required}')
    if engine == 'general' and inference['chains'] < 2:
        raise DefinitionError('NUTS diagnostics require at least two chains')
    if inference.get('parallel_chains', 1) > inference.get('chains', 4):
        raise DefinitionError('parallel_chains cannot exceed chains')
    if engine == 'importance':
        try:
            import bridgestan as bs
        except ImportError as e:
            raise DefinitionError("Install priorlens-science[bridge] for Stan-only IS") from e
        if 'bridgestan' in inference:
            bs.set_bridgestan_path(inference['bridgestan'])
    else:
        try:
            import cmdstanpy
        except ImportError as e:
            raise DefinitionError("Install priorlens-science[stan] for NUTS") from e
        if 'cmdstan' in inference:
            cmdstanpy.set_cmdstan_path(inference['cmdstan'])
        cmdstanpy.cmdstan_path()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs, build = out/'inputs', out/'build'
    inputs.mkdir(); build.mkdir()
    source = Path(settings['model']['stan'])
    shutil.copyfile(source, inputs/'model.stan')
    shutil.copyfile(source, build/'model.stan')
    write_json(inputs/'data.json', settings['data'])
    write_json(out/'resolved_spec.json', settings)
    package = 'bridgestan' if engine == 'importance' else 'cmdstanpy'
    provenance = dict(stan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      settings_sha256=digest(settings), python=platform.python_version(),
                      numpy=np.__version__, scipy=scipy.__version__,
                      backend_version=importlib.metadata.version(package),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/'provenance.json', provenance)
    targets = [TargetSpec.parse(t) for t in settings['targets']]
    fits = [('baseline', settings['priors']['baseline'])]
    for cell in settings['prior_grid']:
        prior = {**settings['priors']['interest'], **{k:v for k,v in cell.items() if k != 'prior_id'}}
        fits.append((cell['prior_id'], prior))
    rows, all_diagnostics = [], []
    current = 'compile'
    try:
        stan = None if engine == 'importance' else cmdstanpy.CmdStanModel(stan_file=str(build/'model.stan'))
        library = str(bs.compile_model(str(build/'model.stan'))) if engine == 'importance' else None
        for index, (prior_id, prior) in enumerate(fits):
            current = prior_id
            leaf = out/f'fit_{index:04d}'
            leaf.mkdir()
            payload = {**settings['data'], **prior}
            write_json(leaf/'stan_data.json', payload)
            seed = int(digest([settings['seed'], 'stan_posterior_v1', prior_id, payload])[:16], 16) % (2**32-1) + 1
            if engine == 'importance':
                model = bs.StanModel(library, payload, seed=seed)
                names = [indexed_name(n) for n in model.param_names(include_tp=False, include_gq=False)]
                _check_target_names(targets, names)
                points, weights, accuracy, extra = importance_draws(model, inference, seed)
            else:
                points, weights, names, accuracy, extra = _nuts_draws(stan, payload, inference, seed, leaf/'chains', targets)
            np.savez_compressed(leaf/'posterior.npz', points=points, weights=weights,
                                parameter_names=np.asarray(names), **extra)
            accuracy.update(prior_id=prior_id, n_points=len(points), parameter_names=names)
            write_json(leaf/'diagnostics.json', accuracy)
            all_diagnostics.append(accuracy)
            if not accuracy['converged']:
                raise NumericalFailure(f'Posterior diagnostics failed at {prior_id}; draws and diagnostics saved')
            summaries, covariance = target_summaries(points, weights, names, targets, prior_id)
            rows.extend(summaries)
            write_json(leaf/'target_covariance.json', covariance)
            write_json(leaf/'done.json', {'status': 'pass', 'prior_id': prior_id,
                       'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(leaf.iterdir()) if p.is_file()}})
        pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
        write_json(out/'diagnostics.json', all_diagnostics)
        result = {'status': 'pass', 'workflow': 'posterior', 'backend': engine,
                  'posterior_fits': len(fits), 'targets_per_fit': len(targets), 'summary_rows': len(rows)}
        write_json(out/'done.json', {**result, 'artifacts': {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [out/'summary.csv', out/'diagnostics.json', out/'resolved_spec.json', out/'provenance.json']
            + sorted(out.glob('fit_*/done.json'))}})
        return result
    except Exception as e:
        write_json(out/'failure.json', {'fit': current, 'error_type': type(e).__name__, 'error': str(e)})
        raise
