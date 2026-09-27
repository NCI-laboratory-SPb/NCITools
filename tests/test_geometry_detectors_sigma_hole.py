from ncitools.geometry.detectors._common import (
    acceptor_angle_range, vdw_radius,
)


def test_acceptor_angle_range_pnictogens():
    assert acceptor_angle_range("N", 1) == (150.0, 180.0)
    assert acceptor_angle_range("N", 2) == (100.0, 145.0)
    assert acceptor_angle_range("N", 3) == (80.0, 135.0)
    assert acceptor_angle_range("N", 4) is None


def test_acceptor_angle_range_chalcogens():
    assert acceptor_angle_range("O", 1) == (100.0, 145.0)
    assert acceptor_angle_range("O", 3) is None      # oxygen hypervalent
    assert acceptor_angle_range("S", 3) == (80.0, 120.0)


def test_acceptor_angle_range_isolated_halide():
    assert acceptor_angle_range("Cl", 0) == "skip"
    assert acceptor_angle_range("Cl", 1) == (80.0, 120.0)


def test_vdw_radius_fallback():
    from ncitools.constants import BONDI
    assert vdw_radius("O", BONDI) == BONDI["O"]
    # Unknown element should fall back to RADII + 0.7
    assert vdw_radius("Fe", BONDI) > 0