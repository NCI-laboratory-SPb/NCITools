import numpy as np
import pytest
from ase import Atoms

from ncitools.geometry.geom import (
    best_fit_plane,
    calculate_homa,
    canonical_bond_key,
    planarity_metrics,
)


def test_canonical_bond_key_sorts_symbols():
    assert canonical_bond_key("C", "C") == "CC"
    assert canonical_bond_key("H", "C") == "CH"
    assert canonical_bond_key("C", "H") == "CH"


def test_best_fit_plane_on_xy_plane():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    centre, normal = best_fit_plane(coords)
    assert np.allclose(centre, [0.5, 0.5, 0.0])
    assert abs(abs(normal[2]) - 1.0) < 1e-8


def test_planarity_metrics_flat_vs_distorted():
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    rms_flat, max_flat = planarity_metrics(flat)
    assert rms_flat == pytest.approx(0.0, abs=1e-10)
    assert max_flat == pytest.approx(0.0, abs=1e-10)

    distorted = flat.copy()
    distorted[0, 2] = 0.5
    rms_d, max_d = planarity_metrics(distorted)
    assert rms_d > 0
    assert max_d >= rms_d


def test_calculate_homa_returns_none_for_unknown_bond():
    """HOMA must return None when a bond type has no parametrisation."""
    # Arrange: hexagon of a fictitious element, so no XeXe parameters exist.
    atoms = Atoms(
        symbols=["Xe"] * 6,
        positions=[
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ],
    )
    assert calculate_homa(atoms, [0, 1, 2, 3, 4, 5]) is None