"""Clarke KL conditional on one baseline-posterior central region.

Region construction is shared with CP. Integration is not a CP cross-coverage
loss: both densities are integrated/normalized on the SAME fixed rectangle.
Refining tensor Gauss-Legendre rules avoids mixing analytic region masses with
a different finite-support approximation to the log-density expectation.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from .core import DefinitionError, NumericalFailure
from .losses import density
from .regions import central_region


def validate_controls(orders, tolerance):
    if (not isinstance(orders, (list, tuple)) or len(orders) < 2 or
        any(not isinstance(n, int) or isinstance(n, bool) or n < 2 or n > 512 for n in orders) or
        any(a >= b for a, b in zip(orders, orders[1:]))):
        raise DefinitionError("central_tkl_orders needs at least two strictly increasing integer orders in [2,512]")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise DefinitionError("central_tkl_tolerance must be finite and positive")


def central_truncated_kl(p, q, alpha=.05, density_engine="exact_density",
                         quantile_rule="weighted_cdf", orders=(16, 32, 64, 128, 256, 512), tolerance=1e-6):
    """Return KL(P(.|R_P) || Q(.|R_P)) and region/integration diagnostics.

    P is the candidate baseline posterior; Q is the informative prior. A
    normalized target density or an explicitly selected KDE is required, as
    in the existing KL API. Unknown-normalizer kernels are not accepted.
    All integrals share quadrature nodes and weights; masses are evaluated in
    log space, including very small Q mass. Failed refinement is not a result.
    """
    validate_controls(orders, tolerance)
    if p.values.shape[1] != q.values.shape[1]:
        raise DefinitionError("Central TKL target dimensions differ")
    region = central_region(p, alpha, quantile_rule)
    lower, upper = region.lower, region.upper
    width = upper - lower
    if not np.all(np.isfinite(width)) or np.any(width <= 0):
        raise NumericalFailure("Central TKL requires a finite positive-volume region")
    lp, lq = density(p, density_engine, normalized=True), density(q, density_engine, normalized=True)
    previous, history = None, []
    for order in orders:
        nodes, weights = leggauss(order)
        axes = [lower[j] + width[j] * (nodes + 1) / 2 for j in range(len(width))]
        axis_weights = [np.log(weights * width[j] / 2) for j in range(len(width))]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([x.ravel() for x in mesh])
        logw = sum(np.meshgrid(*axis_weights, indexing="ij")).ravel()
        # Bound temporary density-evaluation arrays for hierarchical mixtures/KDE.
        def evaluate(function):
            return np.concatenate([np.asarray(function(points[i:i + 256])).reshape(-1)
                                   for i in range(0, len(points), 256)])
        logp, logq = evaluate(lp), evaluate(lq)
        if logp.shape != logw.shape or logq.shape != logw.shape or not np.all(np.isfinite(logp)) or not np.all(np.isfinite(logq)):
            raise NumericalFailure("Central TKL density is nonfinite or unresolved inside the region")
        ap, aq = logw + logp, logw + logq
        logpa, logqa = float(logsumexp(ap)), float(logsumexp(aq))
        wp = np.exp(ap - logpa)
        raw_loss = float(wp @ ((ap - logpa) - (aq - logqa)))
        if not np.isfinite(raw_loss) or raw_loss < -1e-10:
            raise NumericalFailure("Central TKL quadrature produced an invalid conditional KL")
        value = max(0., raw_loss)  # Only roundoff-scale negativity is clipped.
        change = None if previous is None else max(abs(logpa - previous[0]), abs(logqa - previous[1]),
                                                   abs(value - previous[2]) / max(1., abs(value)))
        history.append({"nodes_per_axis": order, "loss": value, "log_p_mass": logpa,
                        "log_q_mass": logqa, "refinement_change": change})
        if change is not None and change <= tolerance and max(logpa, logqa) <= tolerance:
            diagnostics = {**region.diagnostics(), "region_source": "candidate_baseline_posterior",
                "density_engine": density_engine, "quantile_rule": quantile_rule,
                "p_mass": float(np.exp(logpa)), "q_mass": float(np.exp(logqa)),
                "log_p_mass": logpa, "log_q_mass": logqa,
                "p_mass_minus_nominal": float(np.exp(logpa)) - (1 - alpha),
                "p_mass_minus_support_mass": float(np.exp(logpa)) - region.own_mass,
                "integration": "refined_tensor_gauss_legendre", "nodes_per_axis": order,
                "converged": True, "tolerance": tolerance, "history": history, "raw_loss": raw_loss}
            return {"loss": value, "region_diagnostics": diagnostics}
        previous = logpa, logqa, value
    error = NumericalFailure("Central TKL rectangle integration did not converge; refine central_tkl_orders")
    error.context = {"region": region.diagnostics(), "integration_history": history}
    raise error
