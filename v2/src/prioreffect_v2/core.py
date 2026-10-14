"""Small, explicit numerical and scientific contracts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


class DefinitionError(ValueError):
    """An unsupported or ambiguous scientific request."""


class NumericalFailure(RuntimeError):
    """No certified numerical answer; never a CP unmatched outcome."""


class InadmissiblePosterior(DefinitionError):
    """The requested posterior is mathematically improper, not a solver failure."""


class UndefinedMoment(DefinitionError):
    """A proper distribution lacks a finite absolute moment required by a loss."""


class UncertifiedMoment(DefinitionError):
    """Moment existence has not been certified; finite samples are not a proof."""


def native(x):
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def digest(x):
    return hashlib.sha256(json.dumps(native(x), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sampling_data(data):
    """Identity validation metadata must not perturb historical random streams."""
    return {k: v for k, v in data.items() if k != "original_unit_ids"}


SEED_SCHEMES = {"sha256_32_v1": 8, "sha256_64_v2": 16}
_seed_scheme = ContextVar("prioreffect_seed_scheme", default="sha256_32_v1")


@contextmanager
def seed_scope(scheme):
    if scheme not in SEED_SCHEMES:
        raise DefinitionError(f"Unknown seed_scheme {scheme}")
    token = _seed_scheme.set(scheme)
    try:
        yield
    finally:
        _seed_scheme.reset(token)


def seed_for(seed, *keys, scheme=None):
    # Direct legacy APIs retain their RNG streams. New planned experiments
    # explicitly select v2 and propagate the scope to every derived fit seed.
    scheme = scheme or _seed_scheme.get()
    if scheme not in SEED_SCHEMES:
        raise DefinitionError(f"Unknown seed_scheme {scheme}")
    return int(digest([int(seed), *keys])[:SEED_SCHEMES[scheme]], 16)


def write_json(path, obj):
    """Atomic publication of a single artifact, on its own filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(native(obj), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def weighted_quantile(values, probabilities, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return np.interp(probabilities, cdf / cdf[-1], values[order])


@dataclass(frozen=True)
class Coordinate:
    variable: str
    transform: str = "identity"
    scale: float = 1.0
    reference: float = 1.0

    def __post_init__(self):
        if self.transform not in {"identity", "log", "square", "logit"}:
            raise DefinitionError(f"Unknown transform {self.transform}")
        if not np.isfinite(self.scale) or self.scale <= 0 or not np.isfinite(self.reference) or self.reference <= 0:
            raise DefinitionError("Coordinate scale/reference must be finite and positive")

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if self.transform == "log":
            if np.any(x <= 0):
                raise DefinitionError("log coordinate requires positive support")
            x = np.log(x / self.reference)
        elif self.transform == "square":
            if np.any(x < 0):
                raise DefinitionError("square coordinate is supported only on nonnegative support")
            x = x ** 2
        elif self.transform == "logit":
            if np.any((x <= 0) | (x >= 1)):
                raise DefinitionError("logit requires open unit support")
            x = np.log(x) - np.log1p(-x)
        return x / self.scale

    def inverse(self, z):
        from scipy.special import expit
        z = np.asarray(z) * self.scale
        if self.transform == "log":
            return self.reference * np.exp(z)
        if self.transform == "square":
            return np.sqrt(z)
        if self.transform == "logit":
            return expit(z)
        return z

    def log_jacobian(self, x):
        x = np.asarray(x)
        if self.transform == "log":
            j = -np.log(x)
        elif self.transform == "square":
            j = np.log(2 * x)
        elif self.transform == "logit":
            j = -np.log(x) - np.log1p(-x)
        else:
            j = np.zeros_like(x)
        return j - np.log(self.scale)


@dataclass(frozen=True)
class TargetSpec:
    id: str
    coordinates: tuple[Coordinate, ...]
    focal_unit_id: str | None = None

    def __post_init__(self):
        if len(self.coordinates) not in (1, 2):
            raise DefinitionError("EPSS targets must have dimension 1 or 2 (model dimension is unrestricted)")
        if len({c.variable for c in self.coordinates}) != len(self.coordinates):
            raise DefinitionError("A joint target must not duplicate a variable")

    @classmethod
    def parse(cls, d):
        return cls(str(d["id"]), tuple(Coordinate(**c) for c in d["coordinates"]), d.get("focal_unit_id"))

    def values(self, draws):
        return np.column_stack([c.forward(draws[c.variable]) for c in self.coordinates])

    @property
    def dimension(self):
        return len(self.coordinates)


@dataclass
class Measure:
    """A target probability measure; integration weights are not density ranks."""
    values: np.ndarray
    weights: np.ndarray
    logpdf: Callable | None = None
    ppf: Callable | None = None
    cdf: Callable | None = None
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    family: str | None = None
    parameters: dict = field(default_factory=dict)
    log_kernel: Callable | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        self.weights = np.asarray(self.weights, dtype=float)
        if self.values.ndim != 2 or self.weights.shape != (len(self.values),) or not len(self.values):
            raise NumericalFailure("Invalid posterior measure dimensions")
        if not np.all(np.isfinite(self.values)) or not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise NumericalFailure("Invalid posterior points/weights")
        self.weights = self.weights / self.weights.sum()
        if self.mean is None:
            self.mean = self.weights @ self.values
        if self.covariance is None:
            delta = self.values - self.mean
            self.covariance = (delta.T * self.weights) @ delta

    def quantile(self, p):
        if self.values.shape[1] != 1:
            raise DefinitionError("Quantiles require scalar target")
        return self.ppf(p) if self.ppf else weighted_quantile(self.values[:, 0], p, self.weights)

    def interval_mass(self, lo, hi):
        if self.cdf:
            return float(self.cdf(hi) - self.cdf(lo))
        return float(self.weights @ ((self.values[:, 0] >= lo) & (self.values[:, 0] <= hi)))


@dataclass
class Posterior:
    model: Any
    data: dict
    prior: dict
    engine: str
    accuracy: dict

    def sample(self, size, rng):
        raise NotImplementedError

    def target(self, target, size=1024, seed=123):
        draws = self.sample(size, np.random.default_rng(seed))
        return Measure(target.values(draws), np.ones(size))


def validate_grid(values, meaning, n, min_retained=1):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or np.any(x != np.floor(x)):
        raise DefinitionError("Candidate values must be a nonempty finite integer list")
    if len(np.unique(x)) != len(x) or np.any(np.diff(x) <= 0):
        raise DefinitionError("Candidate values must be sorted and unique; no silent coercion")
    if meaning not in {"number_augmented", "number_deleted", "pseudo_sample_size", "baseline_size_offset", "informative_size_reduction"}:
        raise DefinitionError(f"Unknown candidate meaning {meaning}")
    if meaning in {"number_augmented", "number_deleted", "pseudo_sample_size"} and np.any(x < 0):
        raise DefinitionError("Counts must be nonnegative")
    sizes = n - x if meaning in {"number_deleted", "informative_size_reduction"} else n + x
    if meaning == "pseudo_sample_size":
        sizes = x
    if meaning != "pseudo_sample_size" and np.any(sizes < min_retained):
        raise DefinitionError("Candidate leaves fewer than min_retained units")
    return x.astype(int)
