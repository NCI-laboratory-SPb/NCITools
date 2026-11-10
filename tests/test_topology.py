"""Tests for the topology BCP search helpers.

These are unit tests on synthetic Hessians and geometry dictionaries;
no CUBE file or density grid is required.
"""

import numpy as np
import pytest

from ncitools.topology.topology import (
    _compute_bcp_score,
    deduplicate_bcps,
    newton_stationary_point,
    pair_alignment_cosine,
    pair_geometry,
    positive_hessian_eigenvector,
    valid_bcp_pair_geometry,
)


# --------------------------------------------------------------------------- #
#  Hessian alignment
# --------------------------------------------------------------------------- #

def test_positive_eigenvector_is_along_bond_path():
    hessian = np.diag([-1.0, -2.0, 3.0])
    vector = positive_hessian_eigenvector(hessian)
    assert vector is not None
    assert np.allclose(np.abs(vector), [0.0, 0.0, 1.0])


def test_positive_eigenvector_rejects_non_bcp_hessians():
    # Two positive eigenvalues: not a (3,-1) critical point.
    assert positive_hessian_eigenvector(np.diag([-1.0, 1.0, 1.0])) is None
    # No positive eigenvalue.
    assert positive_hessian_eigenvector(np.diag([-1.0, -2.0, -3.0])) is None


def test_alignment_cosine_range_and_sign_invariance():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 2.0])

    assert pair_alignment_cosine(a, b, np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0)
    assert pair_alignment_cosine(a, b, np.array([0.0, 0.0, -1.0])) == pytest.approx(1.0)
    assert pair_alignment_cosine(a, b, np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert pair_alignment_cosine(
        a, b, np.array([1.0, 0.0, 1.0])
    ) == pytest.approx(1.0 / np.sqrt(2.0))


def test_alignment_cosine_degenerate_inputs():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 2.0])

    assert pair_alignment_cosine(a, a, np.array([0.0, 0.0, 1.0])) is None
    assert pair_alignment_cosine(a, b, np.zeros(3)) is None
    assert pair_alignment_cosine(a, b, None) is None


# --------------------------------------------------------------------------- #
#  pair_geometry
# --------------------------------------------------------------------------- #

def test_pair_geometry_without_eigenvector_is_unchanged():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 2.0])
    r = np.array([0.0, 0.0, 1.0])

    geometry = pair_geometry(r, a, b)

    assert geometry["alignment"] is None
    assert geometry["atom_distance"] == pytest.approx(2.0)
    assert geometry["distance_a"] == pytest.approx(1.0)
    assert geometry["distance_b"] == pytest.approx(1.0)
    assert geometry["projection_fraction"] == pytest.approx(0.5)
    assert geometry["line_distance"] == pytest.approx(0.0)
    assert geometry["angle"] == pytest.approx(180.0)


def test_pair_geometry_adds_alignment_when_eigenvector_given():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 2.0])
    r = np.array([0.0, 0.0, 1.0])

    geometry = pair_geometry(r, a, b, eigenvector=np.array([0.0, 0.0, 1.0]))

    assert geometry["alignment"] == pytest.approx(1.0)
    # Proximity values must not change.
    assert geometry["projection_fraction"] == pytest.approx(0.5)
    assert geometry["line_distance"] == pytest.approx(0.0)


def test_pair_geometry_returns_none_for_coincident_atoms():
    a = np.array([0.0, 0.0, 0.0])
    assert pair_geometry(np.zeros(3), a, a) is None


# --------------------------------------------------------------------------- #
#  valid_bcp_pair_geometry
# --------------------------------------------------------------------------- #

def _geometry(alignment=None, angle=180.0, fraction=0.5, line_distance=0.0,
              atom_distance=2.0):
    return {
        "distance_a": 1.0,
        "distance_b": 1.0,
        "atom_distance": atom_distance,
        "angle": angle,
        "projection": fraction * atom_distance,
        "projection_fraction": fraction,
        "line_distance": line_distance,
        "alignment": alignment,
    }


def test_proximity_checks_are_unchanged():
    assert valid_bcp_pair_geometry(_geometry())
    assert not valid_bcp_pair_geometry(_geometry(angle=120.0))
    assert not valid_bcp_pair_geometry(_geometry(fraction=-0.3))
    assert not valid_bcp_pair_geometry(_geometry(fraction=1.3))
    # max_line_fraction * atom_distance == 0.7
    assert not valid_bcp_pair_geometry(_geometry(line_distance=0.8))
    assert valid_bcp_pair_geometry(_geometry(line_distance=0.6))


def test_alignment_filter_is_optional():
    # No alignment supplied and no threshold: pure proximity.
    assert valid_bcp_pair_geometry(_geometry(alignment=None))


def test_alignment_filter_rejects_low_and_missing_values():
    assert valid_bcp_pair_geometry(_geometry(alignment=0.9), min_alignment=0.8)
    assert not valid_bcp_pair_geometry(_geometry(alignment=0.5), min_alignment=0.8)
    assert not valid_bcp_pair_geometry(_geometry(alignment=None), min_alignment=0.8)


# --------------------------------------------------------------------------- #
#  Candidate scoring and deduplication
# --------------------------------------------------------------------------- #

def _bcp(alignment, distance_a=1.0, distance_b=1.0, angle=180.0,
         line_distance=0.0, projection_fraction=0.5, atom1=0, atom2=1,
         position=(0.0, 0.0, 0.0)):
    return {
        "position": np.asarray(position, dtype=float),
        "atom1": atom1,
        "atom2": atom2,
        "pair_distance": distance_a + distance_b,
        "pair_angle": angle,
        "line_distance": line_distance,
        "projection_fraction": projection_fraction,
        "distance_a": distance_a,
        "distance_b": distance_b,
        "alignment": alignment,
    }


def test_alignment_dominates_geometry_score():
    # Aligned candidate has worse proximity, but must still rank first.
    aligned = _bcp(0.99, distance_a=3.0, distance_b=3.0, angle=170.0,
                   line_distance=0.5)
    misaligned = _bcp(0.10, distance_a=0.1, distance_b=0.1, angle=180.0,
                      line_distance=0.0)

    assert _compute_bcp_score(aligned) < _compute_bcp_score(misaligned)


def test_missing_alignment_ranks_last():
    without_alignment = _bcp(None)
    with_alignment = _bcp(0.5)

    assert _compute_bcp_score(with_alignment) < _compute_bcp_score(without_alignment)


def test_deduplicate_picks_the_aligned_pair():
    aligned = _bcp(0.99, atom1=0, atom2=1)
    misaligned = _bcp(0.10, atom1=0, atom2=2)

    result = deduplicate_bcps([misaligned, aligned], position_tol_bohr=0.05)

    assert len(result) == 1
    assert result[0]["atom2"] == 1
    assert result[0]["alignment"] == pytest.approx(0.99)


def test_deduplicate_keeps_distinct_positions():
    first = _bcp(0.9, position=(0.0, 0.0, 0.0))
    second = _bcp(0.9, position=(1.0, 0.0, 0.0))

    result = deduplicate_bcps([first, second], position_tol_bohr=0.05)

    assert len(result) == 2


# --------------------------------------------------------------------------- #
#  Newton solver returns eigenvectors
# --------------------------------------------------------------------------- #

class _QuadraticDensity:
    """rho = -0.5 x^2 - 0.5 y^2 + z^2, a (3,-1) point at the origin."""

    def compute_all(self, r):
        r = np.asarray(r, dtype=float)
        rho = -0.5 * r[0] ** 2 - 0.5 * r[1] ** 2 + r[2] ** 2
        grad = np.array([-r[0], -r[1], 2.0 * r[2]])
        hess = np.diag([-1.0, -1.0, 2.0])
        return rho, grad, hess


def test_newton_stationary_point_returns_eigenvectors():
    result = newton_stationary_point(_QuadraticDensity(), np.zeros(3))

    assert result is not None
    assert np.allclose(result["eigenvalues"], [-1.0, -1.0, 2.0])
    assert "eigenvectors" in result
    positive_vector = result["eigenvectors"][:, 2]
    assert np.allclose(np.abs(positive_vector), [0.0, 0.0, 1.0])


def test_worker_enforces_alignment_filter(monkeypatch):
    import ncitools.topology.topology as topology_module

    monkeypatch.setattr(topology_module, "_WORKER_INTERP", _QuadraticDensity())

    settings = {
        "grad_tol": 1.0e-6,
        "max_iter": 60,
        "eig_tol": 1.0e-8,
        "min_bcp_angle": 140.0,
        "projection_margin": 0.20,
        "max_line_fraction": 0.35,
        "min_alignment": 0.8,
        "max_step_bohr": 1.5,
    }

    # A-B is parallel to the positive eigenvector (z): accepted.
    aligned = topology_module._search_pair_worker((
        0, 1,
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        settings,
    ))
    assert len(aligned) == 1
    assert aligned[0]["alignment"] == pytest.approx(1.0)

    # A-B is perpendicular to the positive eigenvector: hard-filtered.
    perpendicular = topology_module._search_pair_worker((
        0, 1,
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        settings,
    ))
    assert perpendicular == []
