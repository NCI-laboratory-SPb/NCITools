import numpy as np
import pytest
from ncitools.geometry.detectors.clash import _segment_distance

#-----------------------------------------
# Lumelsky 3D segment distance mathematics
#-----------------------------------------
def test_segment_distance_intersecting_at_origin():
    p1, p2 = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q1, q2 = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert _segment_distance(p1, p2, q1, q2) == pytest.approx(0.0, abs=1e-8)


def test_segment_distance_skew_3d():
    """Orthogonal skew segments separated along the Z axis by 2.5 Å."""
    p1, p2 = np.array([-1.0, 0.0, 2.5]), np.array([1.0, 0.0, 2.5])
    q1, q2 = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert _segment_distance(p1, p2, q1, q2) == pytest.approx(2.5, abs=1e-8)

def test_segment_distance_parallel_collinear():
    """Parallel segments must not trigger ZeroDivisionError."""
    p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
    q1, q2 = np.array([0.0, 3.0, 0.0]), np.array([2.0, 3.0, 0.0])
    assert _segment_distance(p1, p2, q1, q2) == pytest.approx(3.0, abs=1e-8)

def test_segment_distance_clipping_and_recalculation():
    """
    Assymmetric segments testing boundary clipping
    and the proper direction of translation vector.
    """
    p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
    q1, q2 = np.array([4.0, 3.0, 0.0]), np.array([6.0, 3.0, 0.0])
    expected = np.sqrt((4.0 - 2.0)**2 + (3.0 - 0.0)**2)
    assert _segment_distance(p1, p2, q1, q2) == pytest.approx(expected, abs=1e-8)

def test_segment_distance_zero_length():
    """Pathological zero-length bond must fall back gracefully without crashing."""
    p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    q1, q2 = np.array([0.0, 2.0, 0.0]), np.array([0.0, 2.0, 0.0])
    assert _segment_distance(p1, p2, q1, q2) == pytest.approx(2.0, abs=1e-8)
