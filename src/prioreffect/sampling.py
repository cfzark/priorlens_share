"""Shared pseudo-data paths for Jones and CP matching variants."""

from __future__ import annotations

import numpy as np


SAMPLING_METHODS = (
    "ppd_augmentation",
    "bootstrap_augmentation",
    "bootstrap_deletion",
    "subsample_deletion",
)


def validate_sampling_method(sampling_method):
    method = str(sampling_method).strip().lower()
    if method not in SAMPLING_METHODS:
        choices = ", ".join(SAMPLING_METHODS)
        raise ValueError(
            f"sampling_method must be one of: {choices}; got {sampling_method!r}."
        )
    return method


def is_deletion_sampling(sampling_method):
    return validate_sampling_method(sampling_method).endswith("_deletion")


def is_empirical_augmentation(sampling_method):
    return validate_sampling_method(sampling_method) == "bootstrap_augmentation"


def is_exploratory_sampling(sampling_method):
    return validate_sampling_method(sampling_method) != "ppd_augmentation"


def validate_augmentation_grid(m_grid):
    """Validate augmented-sample counts and return sorted unique integers."""
    grid = np.unique(np.asarray(m_grid, dtype=int))
    if len(grid) == 0:
        raise ValueError("m_grid must not be empty.")
    if np.any(grid < 0):
        raise ValueError("Augmentation counts in m_grid must be nonnegative.")
    return grid


def empirical_bootstrap_augmentation_path(y, m_grid, rng):
    """Generate one nested empirical-bootstrap augmentation path."""
    values = np.asarray(y).reshape(-1)
    grid = validate_augmentation_grid(m_grid)
    max_m = int(np.max(grid))
    additions = (
        rng.choice(values, size=max_m, replace=True)
        if max_m > 0
        else np.empty(0, dtype=values.dtype)
    )
    return grid, {int(m): additions[: int(m)] for m in grid}


def empirical_bootstrap_augmentation_index_path(n, m_grid, rng):
    """Generate a nested bootstrap-augmentation path of unit indices.

    Sampling indices, rather than a single value array, keeps all fields of a
    sampling unit aligned.  For example, an Eight Schools unit is the observed
    pair ``(y_j, s_j)``.  The returned path contains only the newly appended
    indices; consequently the path at ``m=0`` is empty and ``m`` may exceed
    the number of observed units.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    grid = validate_augmentation_grid(m_grid)
    max_m = int(np.max(grid))
    additions = (
        rng.choice(np.arange(n, dtype=int), size=max_m, replace=True)
        if max_m > 0
        else np.empty(0, dtype=int)
    )
    return grid, {
        int(m): additions[: int(m)].copy() for m in grid
    }


def empirical_bootstrap_augmentation_matrix(y, repetitions, max_m, rng):
    """Generate vectorized empirical-bootstrap future-data paths."""
    values = np.asarray(y).reshape(-1)
    repetitions = int(repetitions)
    max_m = int(max_m)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if max_m < 0:
        raise ValueError("max_m must be nonnegative.")
    if max_m == 0:
        return np.empty((repetitions, 0), dtype=values.dtype)
    return rng.choice(values, size=(repetitions, max_m), replace=True)


def validate_deletion_grid(m_grid, n, min_retained=1):
    """Validate deletion counts and return a sorted unique integer grid."""
    grid = np.unique(np.asarray(m_grid, dtype=int))
    n = int(n)
    min_retained = int(min_retained)
    if min_retained < 0:
        raise ValueError("min_retained must be nonnegative.")
    if len(grid) == 0:
        raise ValueError("m_grid must not be empty.")
    if np.any(grid < 0):
        raise ValueError("Deletion counts in m_grid must be nonnegative.")
    if np.any(n - grid < min_retained):
        raise ValueError(
            "Deletion grid leaves too few observations: require "
            f"n - m >= {min_retained}, with n={n}; got max(m)={grid.max()}."
        )
    return grid


def _positive_prefix_variance(sequence, prefix_size, tolerance):
    prefix = np.asarray(sequence, dtype=float)[..., : int(prefix_size)]
    sums = np.sum(prefix, axis=-1)
    sums_sq = np.sum(np.square(prefix), axis=-1)
    sse = sums_sq - np.square(sums) / int(prefix_size)
    return sse > float(tolerance)


def empirical_deletion_path(
    y,
    m_grid,
    rng,
    sampling_method,
    min_retained=1,
    require_positive_variance=False,
    variance_tolerance=1e-12,
):
    """Generate one nested empirical path indexed by the number deleted.

    ``bootstrap_deletion`` draws a length-n sample with replacement from the
    empirical distribution. ``subsample_deletion`` randomly permutes the
    observed data, so m=0 is exactly the observed multiset. In either case,
    the size n-m sample is a prefix of the same path, providing common random
    numbers across candidate m values and matching directions.
    """
    method = validate_sampling_method(sampling_method)
    if not is_deletion_sampling(method):
        raise ValueError(
            "empirical_deletion_path requires a *_deletion sampling method."
        )

    values = np.asarray(y).reshape(-1)
    n = len(values)
    grid = validate_deletion_grid(m_grid, n, min_retained=min_retained)
    if require_positive_variance and np.ptp(values.astype(float)) <= 0.0:
        raise ValueError(
            "Positive-variance deletion paths cannot be generated from "
            "constant observed data."
        )

    smallest_retained = int(n - np.max(grid))
    if method == "bootstrap_deletion":
        sequence = rng.choice(values, size=n, replace=True)
        if require_positive_variance:
            while not bool(
                _positive_prefix_variance(
                    sequence, smallest_retained, variance_tolerance
                )
            ):
                sequence = rng.choice(values, size=n, replace=True)
    else:
        sequence = rng.permutation(values)
        if require_positive_variance and not bool(
            _positive_prefix_variance(
                sequence, smallest_retained, variance_tolerance
            )
        ):
            raise ValueError(
                "The subsample deletion path has zero variance at the "
                f"smallest retained size ({smallest_retained})."
            )

    return grid, {int(m): sequence[: n - int(m)] for m in grid}


def empirical_deletion_matrix(
    y,
    repetitions,
    rng,
    sampling_method,
    min_retained=1,
    require_positive_variance=False,
    variance_tolerance=1e-12,
):
    """Return independent length-n empirical sequences for vectorized paths."""
    method = validate_sampling_method(sampling_method)
    if not is_deletion_sampling(method):
        raise ValueError(
            "empirical_deletion_matrix requires a *_deletion sampling method."
        )
    values = np.asarray(y).reshape(-1)
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    min_retained = int(min_retained)
    if min_retained < 2 and require_positive_variance:
        raise ValueError(
            "Positive sample variance requires min_retained >= 2."
        )
    if require_positive_variance and np.ptp(values.astype(float)) <= 0.0:
        raise ValueError(
            "Positive-variance deletion paths cannot be generated from "
            "constant observed data."
        )
    if method == "bootstrap_deletion":
        sequences = rng.choice(
            values, size=(repetitions, len(values)), replace=True
        )
        if require_positive_variance:
            valid = _positive_prefix_variance(
                sequences, min_retained, variance_tolerance
            )
            while not np.all(valid):
                sequences[~valid] = rng.choice(
                    values,
                    size=(int(np.sum(~valid)), len(values)),
                    replace=True,
                )
                valid = _positive_prefix_variance(
                    sequences, min_retained, variance_tolerance
                )
        return sequences
    order = np.argsort(rng.random((repetitions, len(values))), axis=1)
    sequences = values[order]
    if require_positive_variance and not np.all(
        _positive_prefix_variance(
            sequences, min_retained, variance_tolerance
        )
    ):
        raise ValueError(
            "At least one subsample deletion path has zero variance at the "
            f"smallest retained size ({min_retained})."
        )
    return sequences


def empirical_deletion_index_path(
    n,
    m_grid,
    rng,
    sampling_method,
    min_retained=1,
    required_index=None,
):
    """Generate a nested deletion path of observation indices.

    This is the paired-observation analogue of :func:`empirical_deletion_path`.
    It is useful when one sampling unit contains several aligned fields, such
    as an Eight Schools ``(y_j, s_j)`` pair.

    If ``required_index`` is supplied, that observation is placed first and
    retained for every candidate deletion count.  The remaining ``n-1``
    positions are either a permutation of the other observations
    (``subsample_deletion``) or a bootstrap sample from the other observations
    (``bootstrap_deletion``).  This supports a formal school-specific
    posterior target while deleting only auxiliary schools.
    """
    method = validate_sampling_method(sampling_method)
    if not is_deletion_sampling(method):
        raise ValueError(
            "empirical_deletion_index_path requires a *_deletion sampling method."
        )
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    grid = validate_deletion_grid(
        m_grid, n, min_retained=min_retained
    )
    indices = np.arange(n, dtype=int)

    if required_index is None:
        if method == "bootstrap_deletion":
            sequence = rng.choice(indices, size=n, replace=True)
        else:
            sequence = rng.permutation(indices)
    else:
        required_index = int(required_index)
        if not 0 <= required_index < n:
            raise ValueError(
                f"required_index must lie in [0, {n}); got {required_index}."
            )
        if n - int(np.max(grid)) < 1:
            raise ValueError(
                "A required observation needs at least one retained unit."
            )
        auxiliary = indices[indices != required_index]
        if method == "bootstrap_deletion":
            auxiliary_sequence = rng.choice(
                auxiliary, size=n - 1, replace=True
            )
        else:
            auxiliary_sequence = rng.permutation(auxiliary)
        sequence = np.concatenate(
            [np.asarray([required_index], dtype=int), auxiliary_sequence]
        )

    return grid, {
        int(m): sequence[: n - int(m)].copy() for m in grid
    }


def select_signed_direction(
    negative_objective,
    positive_objective,
    m_grid,
    negative_wins_ties=True,
):
    """Select direction and deletion/augmentation count for one outer path."""
    negative = np.asarray(negative_objective, dtype=float)
    positive = np.asarray(positive_objective, dtype=float)
    grid = np.asarray(m_grid, dtype=int)
    if negative.shape != positive.shape or negative.shape != grid.shape:
        raise ValueError("Objective arrays and m_grid must have equal shapes.")
    if not np.any(np.isfinite(negative)) or not np.any(np.isfinite(positive)):
        raise ValueError("Each direction must contain a finite objective.")

    idx_negative = int(np.nanargmin(negative))
    idx_positive = int(np.nanargmin(positive))
    best_negative = float(negative[idx_negative])
    best_positive = float(positive[idx_positive])
    choose_negative = (
        best_negative <= best_positive
        if negative_wins_ties
        else best_negative < best_positive
    )
    if choose_negative:
        signed_m = -int(grid[idx_negative])
        direction = "negative"
    else:
        signed_m = int(grid[idx_positive])
        direction = "positive"

    return {
        "signed_m": signed_m,
        "direction": direction,
        "negative_m": int(grid[idx_negative]),
        "positive_m": int(grid[idx_positive]),
        "negative_objective": best_negative,
        "positive_objective": best_positive,
    }
