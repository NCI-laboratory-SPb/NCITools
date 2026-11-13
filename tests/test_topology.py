"""Tests for the topology BCP search helpers.

These are unit tests on synthetic Hessians and geometry dictionaries;
no CUBE file or density grid is required.
"""

import networkx as nx
import numpy as np
import pytest

from ncitools.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, BONDI
from ncitools.topology.detectors.dihydrogen import (
    find_dihydrogen_interactions,
)
from ncitools.topology.detectors.metallophilic import (
    find_metallophilic_interactions,
)
from ncitools.topology.detectors.sigma_hole import (
    find_sigma_hole_interactions,
)
from ncitools.topology.detectors.aromatic import (
    find_nci_with_aromatic,
)
from ncitools.topology.detectors.carbonyl import (
    find_carbonyl_interactions,
)
from ncitools.topology.detectors.unclassified import (
    find_unclassified_interactions,
)
from ncitools.topology.topology import (
    _compute_bcp_score,
    bond_cutoff_angstrom,
    classify_covalent_like,
    deduplicate_bcps,
    find_nci,
    make_seed_points,
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
    assert valid_bcp_pair_geometry(_geometry(), min_alignment=None)
    assert not valid_bcp_pair_geometry(_geometry(angle=120.0), min_alignment=None)
    assert not valid_bcp_pair_geometry(_geometry(fraction=-0.3), min_alignment=None)
    assert not valid_bcp_pair_geometry(_geometry(fraction=1.3), min_alignment=None)
    # max_line_fraction * atom_distance == 0.7
    assert not valid_bcp_pair_geometry(
        _geometry(line_distance=0.8), min_alignment=None
    )
    assert valid_bcp_pair_geometry(
        _geometry(line_distance=0.6), min_alignment=None
    )


def test_alignment_filter_can_be_disabled():
    # No alignment supplied and no threshold: pure proximity.
    assert valid_bcp_pair_geometry(_geometry(alignment=None), min_alignment=None)


def test_alignment_filter_rejects_low_and_missing_values():
    assert valid_bcp_pair_geometry(_geometry(alignment=0.95), min_alignment=0.9)
    assert not valid_bcp_pair_geometry(_geometry(alignment=0.85), min_alignment=0.9)
    assert not valid_bcp_pair_geometry(_geometry(alignment=None), min_alignment=0.9)


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


def test_geometry_score_is_alignment_independent():
    # Alignment is a hard gate, not part of the ranking key.
    well_aligned = _bcp(0.99, distance_a=1.0, distance_b=1.0)
    poorly_aligned = _bcp(0.50, distance_a=1.0, distance_b=1.0)
    without_alignment = _bcp(None, distance_a=1.0, distance_b=1.0)

    assert _compute_bcp_score(well_aligned) == pytest.approx(
        _compute_bcp_score(poorly_aligned)
    )
    assert _compute_bcp_score(well_aligned) == pytest.approx(
        _compute_bcp_score(without_alignment)
    )


def test_geometry_score_prefers_closer_bond_like_pair():
    close = _bcp(0.99, distance_a=1.0, distance_b=1.0)
    far = _bcp(0.99, distance_a=3.0, distance_b=3.0)

    assert _compute_bcp_score(close) < _compute_bcp_score(far)


def test_deduplicate_picks_best_geometry_not_highest_alignment():
    # Regression for a linear O-H...N contact: the heavy-atom N...O pair
    # is marginally better aligned (spline noise), but the BCP lies on
    # the H...N bond path, so the hydrogen pair must win.
    hydrogen_bond = _bcp(
        0.9995, distance_a=0.5, distance_b=1.5, atom1=0, atom2=1,
    )
    heavy_atom = _bcp(
        0.9999, distance_a=1.48, distance_b=1.48, atom1=2, atom2=3,
    )

    result = deduplicate_bcps([heavy_atom, hydrogen_bond], position_tol_bohr=0.05)

    assert len(result) == 1
    assert {result[0]["atom1"], result[0]["atom2"]} == {0, 1}


def test_default_min_alignment_is_0_9():
    import inspect

    import ncitools.topology.topology as topology_module

    signature = inspect.signature(topology_module.find_bcps_parallel)
    assert signature.parameters["min_alignment"].default == pytest.approx(0.9)


def test_deduplicate_keeps_distinct_positions():
    first = _bcp(0.9, position=(0.0, 0.0, 0.0))
    second = _bcp(0.9, position=(1.0, 0.0, 0.0))

    result = deduplicate_bcps([first, second], position_tol_bohr=0.05)

    assert len(result) == 2


# --------------------------------------------------------------------------- #
#  Covalent-like BCP screening
# --------------------------------------------------------------------------- #

def test_bond_cutoff_is_additive_tolerance():
    # R_C = 0.75 A in the current RADII table.
    assert bond_cutoff_angstrom("C", "C") == pytest.approx(1.70)
    assert bond_cutoff_angstrom("C", "C", bias=0.5) == pytest.approx(2.00)


def _covalent_bcp(distance_angstrom, rho=0.1, laplacian=0.0):
    return {
        "atom1": 0,
        "atom2": 1,
        "pair_distance": distance_angstrom * ANGSTROM_TO_BOHR,
        "rho": rho,
        "laplacian": laplacian,
    }


def _carbon_pair():
    return [
        (6, np.zeros(3)),
        (6, np.array([0.0, 0.0, 1.7 * ANGSTROM_TO_BOHR])),
    ]


def test_classify_covalent_like_uses_additive_tolerance():
    atoms = _carbon_pair()

    # Default tolerance 0.2 A -> cutoff C-C 1.70 A.
    assert classify_covalent_like(_covalent_bcp(1.69), atoms)
    assert not classify_covalent_like(_covalent_bcp(1.71), atoms)
    # A larger tolerance widens the window.
    assert classify_covalent_like(_covalent_bcp(1.80), atoms, tolerance=0.4)


def test_classify_covalent_like_keeps_other_gates():
    atoms = _carbon_pair()

    assert not classify_covalent_like(_covalent_bcp(1.50, rho=0.05), atoms)
    assert not classify_covalent_like(
        _covalent_bcp(1.50, laplacian=1.0),
        atoms,
        require_negative_laplacian=True,
    )
    assert classify_covalent_like(
        _covalent_bcp(1.50, laplacian=-1.0),
        atoms,
        require_negative_laplacian=True,
    )


def test_find_nci_splits_with_tolerance():
    atoms = _carbon_pair()
    covalent = _covalent_bcp(1.50, laplacian=1.0)
    nci = {
        "atom1": 0,
        "atom2": 1,
        "pair_distance": 3.0 * ANGSTROM_TO_BOHR,
        "rho": 0.01,
        "laplacian": 0.05,
    }

    bcps_nci, bcps_covalent = find_nci([covalent, nci], atoms)

    assert bcps_covalent == [covalent]
    assert bcps_nci == [nci]


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


def test_seed_points_start_at_midpoint_then_fall_back():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 4.0])

    seeds = make_seed_points(a, b)

    assert [float(seed[2]) for seed in seeds] == pytest.approx([2.0, 3.0, 1.0])


def test_worker_falls_back_when_midpoint_search_fails(monkeypatch):
    import ncitools.topology.topology as topology_module

    monkeypatch.setattr(topology_module, "_WORKER_INTERP", _QuadraticDensity())

    real_newton = topology_module.newton_stationary_point
    seeds = []

    def flaky_newton(interp, r0, **kwargs):
        seeds.append(np.asarray(r0, dtype=float).copy())
        if len(seeds) == 1:
            return None
        return real_newton(interp, r0, **kwargs)

    monkeypatch.setattr(topology_module, "newton_stationary_point", flaky_newton)

    settings = {
        "grad_tol": 1.0e-6,
        "max_iter": 60,
        "eig_tol": 1.0e-8,
        "min_bcp_angle": 140.0,
        "projection_margin": 0.20,
        "max_line_fraction": 0.35,
        "min_alignment": 0.9,
        "max_step_bohr": 1.5,
    }

    result = topology_module._search_pair_worker((
        0, 1,
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        settings,
    ))

    # The midpoint was tried first, then the 0.75 seed succeeded.
    assert len(seeds) == 2
    assert seeds[1][2] == pytest.approx(0.5)
    assert len(result) == 1
    assert result[0]["alignment"] == pytest.approx(1.0)


def test_worker_returns_empty_when_all_seeds_fail(monkeypatch):
    import ncitools.topology.topology as topology_module

    monkeypatch.setattr(topology_module, "_WORKER_INTERP", _QuadraticDensity())
    monkeypatch.setattr(
        topology_module, "newton_stationary_point", lambda *args, **kwargs: None
    )

    settings = {
        "grad_tol": 1.0e-6,
        "max_iter": 60,
        "eig_tol": 1.0e-8,
        "min_bcp_angle": 140.0,
        "projection_margin": 0.20,
        "max_line_fraction": 0.35,
        "min_alignment": 0.9,
        "max_step_bohr": 1.5,
    }

    result = topology_module._search_pair_worker((
        0, 1,
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        settings,
    ))

    assert result == []


# --------------------------------------------------------------------------- #
#  Dihydrogen detector
# --------------------------------------------------------------------------- #

def _covalent_graph(atoms, bonds):
    G = nx.Graph()
    for idx, (atomic_number, _) in enumerate(atoms):
        G.add_node(idx, element=int(atomic_number), node_type="atom")
    for a, b in bonds:
        G.add_edge(a, b, bond_type="covalent")
    _assign_num_neighbours(G)
    return G


def _assign_num_neighbours(G):
    """Mirror geometry.graph_builder: count covalent+coordination partners."""
    for idx in G.nodes:
        if not isinstance(idx, int):
            continue
        G.nodes[idx]["num_neighbours"] = sum(
            1 for n in G.adj[idx]
            if G.edges[idx, n].get("bond_type") in ("covalent", "coordination")
        )


def _angstrom_atoms(specs):
    return [
        (z, np.asarray(pos, dtype=float) * ANGSTROM_TO_BOHR)
        for z, pos in specs
    ]


def _hh_bcp(i, j, position_angstrom, rho=0.008):
    return {
        "atom1": i,
        "atom2": j,
        "position": np.asarray(position_angstrom, dtype=float) * ANGSTROM_TO_BOHR,
        "rho": rho,
        "laplacian": 0.0327,
        "ellipticity": 0.74,
        "eigenvalues": np.array([-0.0100, -0.0049, 0.0476]),
        "grad_norm": 1.0e-7,
    }


def _two_methyls(hh_distance=2.2):
    """H1 and H3 belong to different carbons; the H···H gap is adjustable."""
    atoms = _angstrom_atoms([
        (6, (0.0, 0.0, -1.09)),
        (1, (0.0, 0.0, 0.0)),
        (6, (0.0, 0.0, hh_distance + 1.09)),
        (1, (0.0, 0.0, hh_distance)),
    ])
    G = _covalent_graph(atoms, [(0, 1), (2, 3)])
    bcp = _hh_bcp(1, 3, (0.0, 0.0, hh_distance / 2.0))
    return atoms, G, bcp


def test_dihydrogen_detects_h_h_contact():
    atoms, G, bcp = _two_methyls(2.2)

    G = find_dihydrogen_interactions([bcp], G, atoms)

    assert G.has_edge(1, 3)
    edge = G.edges[1, 3]
    assert edge["bond_type"] == "dihydrogen"
    assert edge["distance"] == pytest.approx(2.2)
    assert (edge["donor1_num"], edge["donor2_num"]) == (0, 2)
    assert edge["donor1_symbol"] == "C"
    assert edge["donor2_symbol"] == "C"
    assert edge["rho"] == pytest.approx(0.008)


def test_dihydrogen_keeps_weak_long_h_h_contact():
    # The BCP is the only requirement: weak contacts beyond the vdW sum
    # are reported with a low confidence score instead of being dropped.
    atoms, G, bcp = _two_methyls(2.6)

    G = find_dihydrogen_interactions([bcp], G, atoms)

    assert G.has_edge(1, 3)
    edge = G.edges[1, 3]
    assert edge["bond_type"] == "dihydrogen"
    assert edge["distance"] == pytest.approx(2.6)
    assert edge["confidence_score"] < 0.8


def test_dihydrogen_requires_a_bcp():
    atoms, G, _ = _two_methyls(2.2)

    G = find_dihydrogen_interactions([], G, atoms)

    assert G.number_of_edges() == 2  # only the two C-H bonds


def test_dihydrogen_ignores_geminal_hydrogens():
    atoms = _angstrom_atoms([
        (6, (0.0, 0.0, 0.0)),
        (1, (0.89, 0.0, -0.63)),
        (1, (-0.89, 0.0, -0.63)),
    ])
    G = _covalent_graph(atoms, [(0, 1), (0, 2)])
    bcp = _hh_bcp(1, 2, (0.0, 0.0, -0.63))

    assert not find_dihydrogen_interactions([bcp], G.copy(), atoms).has_edge(1, 2)

    relaxed = find_dihydrogen_interactions(
        [bcp], G.copy(), atoms, exclude_geminal=False
    )
    assert relaxed.edges[1, 2]["bond_type"] == "dihydrogen"


def test_dihydrogen_ignores_non_hydrogen_pairs():
    atoms = _angstrom_atoms([(8, (0.0, 0.0, 0.0)), (8, (0.0, 0.0, 2.6))])
    G = _covalent_graph(atoms, [])
    bcp = _hh_bcp(0, 1, (0.0, 0.0, 1.3))

    G = find_dihydrogen_interactions([bcp], G, atoms)

    assert G.number_of_edges() == 0


def test_dihydrogen_leaves_already_classified_bcps_alone():
    atoms, G, bcp = _two_methyls(2.2)
    G.add_edge(1, 3, bond_type="HB", position=np.asarray(bcp["position"]))

    G = find_dihydrogen_interactions([bcp], G, atoms)

    assert G.edges[1, 3]["bond_type"] == "HB"


def test_dihydrogen_prevents_unclassified_fallback():
    atoms, G, bcp = _two_methyls(2.2)

    G = find_dihydrogen_interactions([bcp], G, atoms)
    G = find_unclassified_interactions([bcp], G, atoms)

    assert G.edges[1, 3]["bond_type"] == "dihydrogen"
    assert all(
        data.get("bond_type") != "unclassified"
        for _, _, data in G.edges(data=True)
    )


# --------------------------------------------------------------------------- #
#  Confidence score / new families
# --------------------------------------------------------------------------- #

def _bcp_dict(atom1, atom2, position_angstrom, rho=0.01, laplacian=0.02):
    return {
        "atom1": atom1,
        "atom2": atom2,
        "position": np.asarray(position_angstrom, dtype=float) * ANGSTROM_TO_BOHR,
        "rho": rho,
        "laplacian": laplacian,
        "ellipticity": 0.1,
        "eigenvalues": np.array([-0.01, -0.005, 0.02]),
        "grad_norm": 1.0e-7,
    }


def _aromatic_system(extra_specs, ring_radius=1.39):
    """Benzene ring in the xy plane plus extra atoms at indices >= 6."""
    ring_positions = [
        (
            ring_radius * np.cos(2.0 * np.pi * k / 6.0),
            ring_radius * np.sin(2.0 * np.pi * k / 6.0),
            0.0,
        )
        for k in range(6)
    ]
    atoms = _angstrom_atoms(
        [(6, position) for position in ring_positions] + list(extra_specs)
    )

    G = nx.Graph()
    for idx, (atomic_number, _) in enumerate(atoms):
        G.add_node(idx, element=int(atomic_number), node_type="atom")
    for k in range(6):
        G.add_edge(k, (k + 1) % 6, bond_type="covalent")

    G.add_node(
        "ring_0",
        node_type="aromatic_center",
        aromatic=True,
        cycle=tuple(range(6)),
        position=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        radius=ring_radius * ANGSTROM_TO_BOHR,
        size=6,
    )
    for k in range(6):
        G.add_edge(k, "ring_0", bond_type="aromatic")

    _assign_num_neighbours(G)
    return atoms, G


def _ase_atoms(atoms):
    from ase import Atoms as AseAtoms

    from ncitools.utils import get_symbol

    return AseAtoms(
        symbols=[get_symbol(atomic_number) for atomic_number, _ in atoms],
        positions=[
            np.asarray(position) * BOHR_TO_ANGSTROM for _, position in atoms
        ],
    )


def _to_angstrom_graph(G):
    """Copy a topology graph and convert aromatic metadata to Angstrom."""
    H = G.copy()
    for _, data in H.nodes(data=True):
        if data.get("node_type") != "aromatic_center":
            continue
        if data.get("position") is not None:
            data["position"] = (
                np.asarray(data["position"]) * BOHR_TO_ANGSTROM
            )
        if data.get("radius") is not None:
            data["radius"] = float(data["radius"]) * BOHR_TO_ANGSTROM
    return H


def _tetrel_system():
    atoms = _angstrom_atoms([
        (14, (0.0, 0.0, 0.0)),        # Si (electrophile)
        (6, (-1.9, 0.0, 0.0)),        # D
        (8, (2.2, 0.0, 0.0)),         # A
        (1, (1.72, -0.831, 0.0)),     # Y on A
    ])
    G = _covalent_graph(atoms, [(0, 1), (2, 3)])
    bcp = _bcp_dict(0, 2, (1.1, 0.0, 0.0))
    return atoms, G, bcp


def test_sigma_hole_detects_tetrel_bond():
    atoms, G, bcp = _tetrel_system()

    G = find_sigma_hole_interactions([bcp], G, atoms)

    assert G.edges[0, 2]["bond_type"] == "TetB"
    assert 0.0 < G.edges[0, 2]["confidence_score"] <= 1.0


def test_sigma_hole_tetrel_confidence_matches_geometry():
    from ncitools.geometry.detectors.sigma_hole import find_nci_bonds

    atoms, G, bcp = _tetrel_system()
    G_topo = find_sigma_hole_interactions([bcp], G.copy(), atoms)

    G_geom = find_nci_bonds(
        _ase_atoms(atoms), G.copy(), ["Si"], 150.0, "TetB",
        radii=BONDI, confidence_threshold=0.0,
    )

    assert (
        G_topo.edges[0, 2]["confidence_score"]
        == pytest.approx(G_geom.edges[0, 2]["confidence_score"])
    )
    assert (
        G_topo.edges[0, 2]["angle_xay"]
        == pytest.approx(G_geom.edges[0, 2]["angle_xay"])
    )


def test_sigma_hole_skip_tetrel_family():
    atoms, G, bcp = _tetrel_system()

    G = find_sigma_hole_interactions(
        [bcp], G, atoms, skipped=frozenset({"TetB"}),
    )

    assert not G.has_edge(0, 2)


def _x_pi_system(angle=180.0):
    theta = np.radians(180.0 - angle)
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
    chlorine = np.array([0.0, 0.0, 3.0])
    carbon = chlorine + 1.8 * direction

    atoms, G = _aromatic_system([
        (17, chlorine),
        (6, carbon),
    ])
    G.add_edge(6, 7, bond_type="covalent")
    _assign_num_neighbours(G)

    bcp = _bcp_dict(6, 0, (0.695, 0.0, 1.5))
    return atoms, G, bcp


def test_x_pi_requires_sigma_hole_angle():
    atoms, G, bcp = _x_pi_system(180.0)
    G = find_nci_with_aromatic([bcp], G, atoms)

    assert G.edges[6, "ring_0"]["bond_type"] == "X-pi"
    assert G.edges[6, "ring_0"]["confidence_score"] > 0.8

    atoms, G, bcp = _x_pi_system(120.0)
    G = find_nci_with_aromatic([bcp], G, atoms)

    assert not G.has_edge(6, "ring_0")


def test_x_pi_confidence_matches_geometry():
    from ncitools.geometry.detectors.aromatic import (
        find_nci_with_aromatic as geometry_aromatic,
    )

    atoms, G, bcp = _x_pi_system(180.0)
    G_topo = find_nci_with_aromatic([bcp], G.copy(), atoms)

    G_geom = geometry_aromatic(
        _ase_atoms(atoms), _to_angstrom_graph(G), confidence_threshold=0.0,
    )

    assert (
        G_topo.edges[6, "ring_0"]["confidence_score"]
        == pytest.approx(G_geom.edges[6, "ring_0"]["confidence_score"])
    )


def test_ion_pi_requires_a_bare_ion():
    atoms, G = _aromatic_system([(11, (0.0, 0.0, 2.5))])
    bcp = _bcp_dict(6, 0, (0.695, 0.0, 1.25))

    G = find_nci_with_aromatic([bcp], G, atoms)

    assert G.edges[6, "ring_0"]["bond_type"] == "ion-pi"
    assert G.edges[6, "ring_0"]["angle_to_normal"] == pytest.approx(0.0)

    # A metal with a covalent bond is not a bare ion.
    atoms, G = _aromatic_system([
        (11, (0.0, 0.0, 2.5)),
        (6, (0.0, 0.0, 5.0)),
    ])
    G.add_edge(6, 7, bond_type="covalent")
    _assign_num_neighbours(G)
    bcp = _bcp_dict(6, 0, (0.695, 0.0, 1.25))

    G = find_nci_with_aromatic([bcp], G, atoms)

    assert not G.has_edge(6, "ring_0")


def test_ion_pi_confidence_matches_geometry():
    from ncitools.geometry.detectors.aromatic import (
        find_nci_with_aromatic as geometry_aromatic,
    )

    atoms, G = _aromatic_system([(11, (0.0, 0.0, 2.5))])
    bcp = _bcp_dict(6, 0, (0.695, 0.0, 1.25))
    G_topo = find_nci_with_aromatic([bcp], G.copy(), atoms)

    G_geom = geometry_aromatic(
        _ase_atoms(atoms), _to_angstrom_graph(G), confidence_threshold=0.0,
    )

    assert (
        G_topo.edges[6, "ring_0"]["confidence_score"]
        == pytest.approx(G_geom.edges[6, "ring_0"]["confidence_score"])
    )


def test_n_pi_confidence_matches_geometry():
    from ncitools.geometry.detectors.aromatic import (
        find_nci_with_aromatic as geometry_aromatic,
    )

    atoms, G = _aromatic_system([
        (7, (0.0, 0.0, 3.0)),    # lone-pair donor
        (6, (0.0, 0.0, 4.4)),    # donor substituent
    ])
    G.add_edge(6, 7, bond_type="covalent")
    _assign_num_neighbours(G)
    bcp = _bcp_dict(6, 0, (0.695, 0.0, 1.5))

    G_topo = find_nci_with_aromatic([bcp], G.copy(), atoms)
    G_geom = geometry_aromatic(
        _ase_atoms(atoms), _to_angstrom_graph(G), confidence_threshold=0.0,
    )

    assert G_topo.edges[6, "ring_0"]["bond_type"] == "n-pi"
    assert (
        G_topo.edges[6, "ring_0"]["confidence_score"]
        == pytest.approx(G_geom.edges[6, "ring_0"]["confidence_score"])
    )
    # The donor→centroid vs donor→Y convention must match geometry.
    assert (
        G_topo.edges[6, "ring_0"]["angle"]
        == pytest.approx(G_geom.edges[6, "ring_0"]["angle"])
    )


def test_hb_confidence_matches_geometry():
    from ncitools.geometry.detectors.sigma_hole import find_nci_bonds

    atoms = _angstrom_atoms([
        (8, (0.0, 0.0, 0.0)),      # D
        (1, (0.96, 0.0, 0.0)),     # H
        (7, (2.8, 0.0, 0.0)),      # A
        (6, (4.3, 0.0, 0.0)),      # Y on A
    ])
    G = _covalent_graph(atoms, [(0, 1), (2, 3)])
    bcp = _bcp_dict(1, 2, (1.88, 0.0, 0.0))

    G_topo = find_sigma_hole_interactions([bcp], G.copy(), atoms)
    G_geom = find_nci_bonds(
        _ase_atoms(atoms), G.copy(), ["H"], 110.0, "HB",
        radii=BONDI, confidence_threshold=0.0,
    )

    assert G_topo.edges[1, 2]["bond_type"] == "HB"
    assert (
        G_topo.edges[1, 2]["confidence_score"]
        == pytest.approx(G_geom.edges[1, 2]["confidence_score"])
    )
    assert (
        G_topo.edges[1, 2]["angle_xay"]
        == pytest.approx(G_geom.edges[1, 2]["angle_xay"])
    )


def _au_system(distance=3.5, bonded=False):
    atoms = _angstrom_atoms([
        (79, (0.0, 0.0, 0.0)),
        (79, (0.0, 0.0, distance)),
    ])
    bonds = [(0, 1)] if bonded else []
    G = _covalent_graph(atoms, bonds)
    bcp = _bcp_dict(0, 1, (0.0, 0.0, distance / 2.0))
    return atoms, G, bcp


def test_metallophilic_requires_bcp_and_no_bond():
    atoms, G, bcp = _au_system()
    G = find_metallophilic_interactions([bcp], G, atoms)

    assert G.edges[0, 1]["bond_type"] == "metallophilic"
    assert G.edges[0, 1]["distance"] == pytest.approx(3.5)
    assert 0.0 < G.edges[0, 1]["confidence_score"] <= 1.0

    # No BCP -> no contact.
    atoms, G, _ = _au_system()
    assert not find_metallophilic_interactions([], G, atoms).has_edge(0, 1)

    # Covalent bond -> not metallophilic.
    atoms, G, bcp = _au_system(bonded=True)
    G = find_metallophilic_interactions([bcp], G, atoms)
    assert G.edges[0, 1]["bond_type"] == "covalent"


def test_metallophilic_confidence_matches_geometry():
    from ncitools.geometry.detectors.metallophilic import (
        find_intermetalls_contact,
    )

    atoms, G, bcp = _au_system()
    G_topo = find_metallophilic_interactions([bcp], G.copy(), atoms)
    G_geom = find_intermetalls_contact(_ase_atoms(atoms), G.copy())

    assert (
        G_topo.edges[0, 1]["confidence_score"]
        == pytest.approx(G_geom.edges[0, 1]["confidence_score"])
    )


def test_output_header_uses_input_file(tmp_path):
    from ncitools.topology.output import output

    atoms, G, bcp = _two_methyls(2.2)
    G = find_dihydrogen_interactions([bcp], G, atoms)

    report = output(
        G, filename=str(tmp_path / "PnB_1_cube"), ext=".cub",
        input_file=r"C:\data\PnB_1.cub",
    )
    header = open(report, encoding="utf-8").read().splitlines()
    assert header[2] == r"Input file: C:\data\PnB_1.cub"

    # Without input_file the header keeps the old basename+ext behaviour.
    fallback = output(G, filename=str(tmp_path / "dih"), ext=".cub")
    header = open(fallback, encoding="utf-8").read().splitlines()
    assert header[2] == f"Input file: {tmp_path / 'dih'}.cub"


def _carbonyl_system():
    """Donor O···C=O with a Bürgi–Dunitz angle of ~115 degrees."""
    bd = np.radians(115.0)
    donor = 2.30 * np.array([np.cos(bd), np.sin(bd), 0.0])
    # R-O···C = 120 degrees.
    to_carbon = -donor / np.linalg.norm(donor)
    phi = np.arctan2(to_carbon[1], to_carbon[0]) + np.radians(120.0)
    substituent = donor + 1.20 * np.array([np.cos(phi), np.sin(phi), 0.0])

    atoms = _angstrom_atoms([
        (6, (0.0, 0.0, 0.0)),        # carbonyl carbon
        (8, (1.20, 0.0, 0.0)),       # carbonyl oxygen
        (6, (0.0, 1.30, 0.0)),       # substituent
        (6, (0.0, -1.30, 0.0)),      # substituent
        (8, tuple(donor)),           # lone-pair donor
        (7, tuple(substituent)),     # donor substituent
    ])
    G = _covalent_graph(atoms, [(0, 1), (0, 2), (0, 3), (4, 5)])
    bcp = _bcp_dict(4, 0, tuple(donor / 2.0))
    return atoms, G, bcp


def test_carbonyl_detects_burgi_dunitz_contact():
    atoms, G, bcp = _carbonyl_system()

    G = find_carbonyl_interactions([bcp], G, atoms)

    assert G.edges[4, 0]["bond_type"] == "carbonyl"
    assert G.edges[4, 0]["distance"] == pytest.approx(2.30)
    assert G.edges[4, 0]["burgi_dunitz_angle"] == pytest.approx(115.0)
    assert 0.0 < G.edges[4, 0]["confidence_score"] <= 1.0


def test_carbonyl_confidence_matches_geometry():
    from ncitools.geometry.detectors.carbonyl import (
        find_carbonyl_interactions as geometry_carbonyl,
    )

    atoms, G, bcp = _carbonyl_system()
    G_topo = find_carbonyl_interactions([bcp], G.copy(), atoms)
    G_geom = geometry_carbonyl(
        _ase_atoms(atoms), G.copy(), confidence_threshold=0.0,
    )

    assert (
        G_topo.edges[4, 0]["confidence_score"]
        == pytest.approx(G_geom.edges[4, 0]["confidence_score"])
    )
    assert (
        G_topo.edges[4, 0]["burgi_dunitz_angle"]
        == pytest.approx(G_geom.edges[4, 0]["burgi_dunitz_angle"])
    )
    assert (
        G_topo.edges[4, 0]["angle_rac"]
        == pytest.approx(G_geom.edges[4, 0]["angle"])
    )


def test_output_renders_dihydrogen_section(tmp_path):
    from ncitools.topology.output import output

    atoms, G, bcp = _two_methyls(2.2)
    G = find_dihydrogen_interactions([bcp], G, atoms)

    report = output(G, filename=str(tmp_path / "dih"))

    text = open(report, encoding="utf-8").read()
    assert "DIHYDROGEN CONTACTS" in text
    assert "Confidence score" in text
    assert "C-H···H-C" in text
    assert "2.2" in text


def test_output_renders_new_sections_with_confidence(tmp_path):
    from ncitools.topology.output import output

    atoms, G, bcp = _tetrel_system()
    tetrel = find_sigma_hole_interactions([bcp], G, atoms)

    atoms, G, bcp = _x_pi_system(180.0)
    x_pi = find_nci_with_aromatic([bcp], G, atoms)

    atoms, G = _aromatic_system([(11, (0.0, 0.0, 2.5))])
    bcp = _bcp_dict(6, 0, (0.695, 0.0, 1.25))
    ion_pi = find_nci_with_aromatic([bcp], G, atoms)

    atoms, G, bcp = _au_system()
    metallophilic = find_metallophilic_interactions([bcp], G, atoms)

    cases = [
        ("tetb", "TETREL BONDS", tetrel),
        ("xpi", "R-X···π INTERACTIONS", x_pi),
        ("ionpi", "ION···π INTERACTIONS", ion_pi),
        ("met", "METALLOPHILIC INTERACTIONS", metallophilic),
    ]

    for name, title, graph in cases:
        report = output(graph, filename=str(tmp_path / name))
        text = open(report, encoding="utf-8").read()
        assert title in text
        assert "Confidence score" in text

        confidence = next(
            data["confidence_score"]
            for _, _, data in graph.edges(data=True)
            if "confidence_score" in data
        )
        expected = str(round(float(confidence), 2))
        if expected.endswith(".0"):
            expected = expected[:-2]
        assert expected in text


def test_output_unclassified_has_no_confidence_column(tmp_path):
    from ncitools.topology.output import output

    G = nx.Graph()
    G.add_edge(
        0, 1,
        bond_type="unclassified",
        atom1_num=0, atom2_num=1,
        atom1_symbol="C", atom2_symbol="O",
        distance=3.0, rho=0.004, laplacian=0.01, ellipticity=0.2,
    )

    report = output(G, filename=str(tmp_path / "unclassified"))

    text = open(report, encoding="utf-8").read()
    assert "UNCLASSIFIED NON-COVALENT CONTACTS" in text
    assert "Confidence score" not in text
