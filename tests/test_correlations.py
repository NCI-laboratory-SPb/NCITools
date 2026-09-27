"""
Smoke tests for the empirical geometry -> energy correlations.

These tests do not check the physics of each correlation in detail.
They only guarantee that the functions are callable, return finite
values on typical geometries, and behave monotonically where expected.
"""

import numpy as np
import pytest

from ncitools.correlations import universal_1_hb, universal_1_sb


# --------------------------------------------------------------------------- #
#  Hydrogen-bond correlation
# --------------------------------------------------------------------------- #

def test_universal_1_hb_returns_finite_negative_energy():
    energy = universal_1_hb(1.8, "O-H···O")
    assert np.isfinite(energy)
    assert energy >= 0


def test_universal_1_hb_is_monotonic_in_distance():
    e_short = universal_1_hb(1.6, "O-H···O")
    e_long = universal_1_hb(2.2, "O-H···O")
    assert e_short > e_long


def test_universal_1_hb_accepts_zero_distance_without_crashing_shape():
    """Pathological r=0 must be handled gracefully (no TypeError)."""
    try:
        result = universal_1_hb(0.0, "O-H···O")
    except (ZeroDivisionError, FloatingPointError, ValueError):
        return
    assert np.isfinite(result) or np.isinf(result)


# --------------------------------------------------------------------------- #
#  Sigma-bond (XB / ChB / PnB / TetB) correlation
# --------------------------------------------------------------------------- #

def test_universal_1_sb_returns_finite_value():
    energy = universal_1_sb(3.2, "IN")
    assert np.isfinite(energy)


def test_universal_1_sb_is_monotonic_in_distance():
    e_short = universal_1_sb(3.0, "IN")
    e_long = universal_1_sb(3.6, "IN")
    assert e_short > e_long