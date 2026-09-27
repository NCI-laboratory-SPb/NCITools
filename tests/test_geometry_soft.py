import numpy as np
import pytest

from ncitools.geometry.soft import (
    soft_greater, soft_less, soft_range
)


def test_soft_greater_value_at_threshold():
    assert soft_greater(0.0, 0.0) == pytest.approx(0.8)


def test_soft_greater_monotonic():
    xs = np.linspace(-5, 5, 51)
    ys = soft_greater(xs, 0.0)
    assert np.all(np.diff(ys) > 0)


def test_soft_less_is_complementary():
    assert soft_less(1.0, 0.0) == pytest.approx(soft_greater(-1.0, 0.0))


def test_soft_range_symmetric():
    assert soft_range(5.0, 0.0, 10.0) == pytest.approx(1.0)


def test_soft_range_value_at_bounds():
    assert soft_range(0.0, 0.0, 10.0) == pytest.approx(0.92)


def test_soft_range_angular_case_is_one_sided():
    """150–180° must behave as a sigmoid on the lower bound."""
    assert soft_range(180.0, 150.0, 180.0) == pytest.approx(1.0, abs=1e-2)
    assert soft_range(140.0, 150.0, 180.0) < 0.5