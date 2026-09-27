"""
Soft (fuzzy) threshold functions used throughout NCItools.

Non-covalent interactions are inherently fuzzy: a hydrogen bond does not
suddenly appear when a distance crosses a hard cutoff.  To reflect this,
every geometric criterion in NCItools is expressed as a smooth score in
the range [0, 1] rather than as a Boolean.

Two families are provided:

* :func:`soft_greater` / :func:`soft_less`
    Monotonic sigmoid criteria, used for statements such as
    "the distance must be below X" or "the angle must be above Y".

* :func:`soft_range`
    A Gaussian window, used for quantities that should lie inside an
    interval, such as the acceptor angle X–A–Y.

All functions accept scalars or NumPy arrays transparently.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Default shape parameters
# --------------------------------------------------------------------------- #

DEFAULT_STEEPNESS = 1.0
DEFAULT_VALUE_AT_THRESHOLD = 0.8
DEFAULT_VALUE_AT_BOUNDS = 0.92


# --------------------------------------------------------------------------- #
#  Sigmoid criteria
# --------------------------------------------------------------------------- #

def _sigmoid(
    x,
    threshold: float,
    steepness: float = DEFAULT_STEEPNESS,
    value_at_threshold: float = DEFAULT_VALUE_AT_THRESHOLD,
):
    """
    Logistic function f(x) = 1 / (1 + exp(-k (x - c))).

    The centre ``c`` is shifted so that ``f(threshold) = value_at_threshold``,
    which makes the parameter ``threshold`` directly interpretable.
    """
    p = value_at_threshold
    centre = threshold + np.log((1.0 - p) / p) / steepness
    return 1.0 / (1.0 + np.exp(-steepness * (x - centre)))


def soft_greater(x, threshold: float, steepness: float = DEFAULT_STEEPNESS):
    """Soft version of the predicate ``x >= threshold``."""
    return _sigmoid(x, threshold, steepness)


def soft_less(x, threshold: float, steepness: float = DEFAULT_STEEPNESS):
    """Soft version of the predicate ``x <= threshold``."""
    return soft_greater(threshold - x, 0.0, steepness)


# --------------------------------------------------------------------------- #
#  Gaussian window
# --------------------------------------------------------------------------- #

def _gaussian(
    x,
    lower: float,
    upper: float,
    value_at_bounds: float = DEFAULT_VALUE_AT_BOUNDS,
):
    """
    Gaussian window centred at ``(lower + upper) / 2``.

    The standard deviation is chosen so that the function equals
    ``value_at_bounds`` exactly at ``lower`` and ``upper``.
    """
    mid = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)

    if half_width == 0.0:
        return 1.0 if np.abs(x - mid) < 1e-8 else 0.0
    if value_at_bounds >= 1.0:
        return 1.0
    if value_at_bounds <= 0.0:
        return 0.0

    sigma2 = (half_width ** 2) / (2.0 * -np.log(value_at_bounds))
    return np.exp(-(x - mid) ** 2 / (2.0 * sigma2))


def soft_range(
    x,
    lower: float,
    upper: float,
    value_at_bounds: float = DEFAULT_VALUE_AT_BOUNDS,
):
    """
    Soft version of the predicate ``lower <= x <= upper``.

    Special case
    ------------
    For valence angles close to 180° (e.g. a range of 150–180°) the
    interval is treated as a one-sided sigmoid on the lower bound,
    because 180° is a periodic maximum and a Gaussian window would
    wrongly penalise values near the top of the range.
    """
    if upper >= 180.0 and lower >= 150.0:
        return soft_greater(x, lower)
    return _gaussian(x, lower, upper, value_at_bounds)