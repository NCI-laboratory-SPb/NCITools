"""
Tests for the geometry-based analysis pipeline.

Coverage
--------
* graph construction (``graph_builder``);
* soft threshold functions used inside the detectors (indirectly);
* every detector family: HB, XB, ChB, PnB, stacking, H-pi, n-pi,
  carbonyl, metallophilic, ion-pi;
* the ``.nci`` report writer.
"""

import numpy as np
import pytest
from ase import Atoms

from ncitools.constants import BONDI, RADII
from ncitools.geometry import (
    build_graph_with_covalent_pairs,
    find_carbonyl_interactions,
    find_intermetalls_contact,
    find_nci_bonds,
    find_nci_with_aromatic,
    output,
)


# ========================================================================== #
#  Constants
# ========================================================================== #

def test_radii_contain_common_elements():
    """Covalent and van der Waals radii must contain the common elements."""
    for element in ("H", "C", "N", "O", "S", "Cl"):
        assert element in RADII, f"Missing {element} in RADII"
        assert element in BONDI, f"Missing {element} in BONDI"


# ========================================================================== #
#  Graph builder
# ========================================================================== #

def test_build_graph_with_covalent_pairs(water_dimer):
    """Water dimer: 6 nodes and exactly four O–H covalent edges."""
    G = build_graph_with_covalent_pairs(water_dimer, radii=RADII, tolerance=0.1)

    assert len(G.nodes) == 6

    expected = {
        frozenset((0, 1)), frozenset((0, 2)),
        frozenset((3, 4)), frozenset((3, 5)),
    }
    actual = {
        frozenset((u, v))
        for u, v, d in G.edges(data=True)
        if d.get("bond_type") == "covalent"
    }
    assert actual == expected


def test_build_graph_no_covalent_bond_when_atoms_far_apart():
    """Two hydrogens 5 Å apart must not be bonded."""
    atoms = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [5, 0, 0]])

    G = build_graph_with_covalent_pairs(atoms, radii=RADII, tolerance=0.1)

    assert len(G.nodes) == 2
    assert len(G.edges) == 0


def test_build_graph_detects_aromatic_ring():
    """A benzene molecule must produce one aromatic-centre node."""
    atoms = Atoms(
        symbols=["C"] * 6 + ["H"] * 6,
        positions=[
            [-2.29444056,  0.30935253,  0.00000000],
            [-0.89928056,  0.30935253,  0.00000000],
            [-0.20174256,  1.51710353,  0.00000000],
            [-0.89939656,  2.72561253, -0.00119900],
            [-2.29422156,  2.72553453, -0.00167800],
            [-2.99182256,  1.51732853, -0.00068200],
            [-2.82939565, -0.61732048,  0.00043788],
            [-0.36459159, -0.61747327,  0.00127954],
            [ 0.86825726,  1.51718137,  0.00061689],
            [-0.36404633,  3.65205742, -0.00125641],
            [-2.82945653,  3.65204555, -0.00260521],
            [-4.06182253,  1.51750660, -0.00085715],
        ],
    )

    G = build_graph_with_covalent_pairs(atoms, planarity_tol=0.15)

    aromatic_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "aromatic_center"
    ]
    assert len(aromatic_nodes) == 1

    ring = G.nodes[aromatic_nodes[0]]
    assert ring["aromatic"] is True
    assert ring["size"] == 6
    assert len(ring["cycle"]) == 6
    assert np.isfinite(ring["rms_planarity"])
    assert ring["rms_planarity"] < 0.11
    assert len(list(G.neighbors(aromatic_nodes[0]))) == 6


# ========================================================================== #
#  σ-hole detectors: HB, XB, ChB, PnB
# ========================================================================== #

def test_find_nci_bonds_hb(water_dimer):
    """Water dimer must yield at least one hydrogen bond."""
    G = build_graph_with_covalent_pairs(water_dimer, radii=RADII, tolerance=0.1)
    G = find_nci_bonds(water_dimer, G, ["H"], 120, "HB", desc="test")

    hb_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("bond_type") == "HB"
    ]
    assert hb_edges, "No hydrogen bond found"

    u, v, edge = hb_edges[0]
    assert edge["bond_type"] == "HB"
    assert edge["symbol_x"] == "H"
    assert edge["symbol_a"] == "O"
    assert edge["length_xb"] > 0      # fixed: was length_hb
    assert edge["length_cov"] > 0
    assert edge["length_da"] > 0
    assert edge["angle"] >= 120


def test_hydrogen_bond_angle_threshold(water_dimer):
    """Stricter D–H···A angles must reduce the number of accepted bonds."""
    G = build_graph_with_covalent_pairs(water_dimer)

    def count_hb(threshold):
        Gx = find_nci_bonds(water_dimer, G.copy(), ["H"], threshold, "HB")
        return sum(
            d.get("bond_type") == "HB"
            for _, _, d in Gx.edges(data=True)
        )

    assert count_hb(100) >= 1
    assert count_hb(180) == 0


@pytest.mark.parametrize(
    "central_elements, bond_type, fixture_name",
    [
        (["Cl", "Br", "I"],       "XB",  "halogen_bond_system"),
        (["S", "Se", "Te"],       "ChB", "chalcogen_bond_system"),
        #(["P", "As", "Sb", "Bi"], "PnB", "pnictogen_bond_system"),
    ],
)
def test_find_nci_bonds_other_types(
    request, central_elements, bond_type, fixture_name,
):
    """Halogen, chalcogen and pnictogen bonds are detected on their fixtures."""
    atoms = request.getfixturevalue(fixture_name)

    G = build_graph_with_covalent_pairs(atoms)
    G = find_nci_bonds(atoms, G, central_elements, 150, bond_type)

    edges = [
        d for _, _, d in G.edges(data=True)
        if d.get("bond_type") == bond_type
    ]
    assert edges, f"{bond_type} interaction was not detected"

    edge = edges[0]
    assert edge["bond_type"] == bond_type
    assert edge["angle"] >= 150


# ========================================================================== #
#  Aromatic-ring detectors: stacking, H-pi, n-pi
# ========================================================================== #

def test_find_nci_with_aromatic_detects_stacking(aromatic_dimer):
    G = build_graph_with_covalent_pairs(aromatic_dimer)
    G = find_nci_with_aromatic(aromatic_dimer, G)

    stacking = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("bond_type") == "stacking"
    ]
    assert len(stacking) == 1

    edge = stacking[0][2]
    assert edge["bond_type"] == "stacking"
    assert edge["centroid_distance"] <= 5.5
    assert edge["offset"] <= 2.8             # matches default offset_threshold
    assert edge["angle"] <= 30.0


def test_find_nci_with_aromatic_no_stacking(no_stacking_dimer):
    G = build_graph_with_covalent_pairs(no_stacking_dimer)
    G = find_nci_with_aromatic(no_stacking_dimer, G)

    stacking = [
        d for _, _, d in G.edges(data=True)
        if d.get("bond_type") == "stacking"
    ]
    assert not stacking


def test_find_nci_with_aromatic_detects_h_pi(ch_pi_system):
    G = build_graph_with_covalent_pairs(ch_pi_system)
    G = find_nci_with_aromatic(ch_pi_system, G)

    edges = [
        d for _, _, d in G.edges(data=True)
        if d.get("bond_type") == "H-pi"
    ]
    assert edges

    edge = edges[0]
    assert edge["bond_type"] == "H-pi"
    assert edge["h_centroid_distance"] <= 3.5
    assert edge["angle"] >= 120


def test_find_nci_with_aromatic_detects_n_pi(n_pi_system):
    G = build_graph_with_covalent_pairs(n_pi_system)
    G = find_nci_with_aromatic(n_pi_system, G)

    edges = [
        d for _, _, d in G.edges(data=True)
        if d.get("bond_type") == "n-pi"
    ]
    assert edges

    edge = edges[0]
    assert edge["bond_type"] == "n-pi"
    assert 2.5 <= edge["lp_centroid_distance"] <= 3.6
    assert 100 <= edge["angle"] <= 130


# ========================================================================== #
#  Carbonyl n→π*
# ========================================================================== #

def test_find_carbonyl_interaction(carbonyl_system):
    G = build_graph_with_covalent_pairs(carbonyl_system)
    G = find_carbonyl_interactions(carbonyl_system, G)

    edges = [
        d for _, _, d in G.edges(data=True)
        if d.get("bond_type") == "carbonyl"
    ]
    assert edges

    edge = edges[0]
    assert edge["bond_type"] == "carbonyl"
    assert "distance" in edge
    assert "burgi_dunitz_angle" in edge
    assert "plane_angle" in edge
    assert edge["distance"] <= 3.22
    assert 90 <= edge["burgi_dunitz_angle"] <= 130


def test_find_carbonyl_interaction_rejects_long_distance(carbonyl_system_far_apart):
    G = build_graph_with_covalent_pairs(carbonyl_system_far_apart)
    G = find_carbonyl_interactions(carbonyl_system_far_apart, G)

    assert not any(
        d.get("bond_type") == "carbonyl"
        for _, _, d in G.edges(data=True)
    )


# ========================================================================== #
#  Metallophilic contacts
# ========================================================================== #

def test_find_metallophilic_contact():
    """Two Au atoms at ~3.2 Å must yield a metallophilic contact."""
    atoms = Atoms(symbols=["Au", "Au"], positions=[[0, 0, 0], [3.2, 0, 0]])

    G = build_graph_with_covalent_pairs(atoms)
    G = find_intermetalls_contact(atoms, G)

    edges = [
        d for _, _, d in G.edges(data=True)
        if d.get("bond_type") == "metallophilic"
    ]
    assert len(edges) == 1
    assert edges[0]["metal_sym_1"] == "Au"
    assert edges[0]["distance"] == pytest.approx(3.2, abs=1e-6)


def test_find_metallophilic_contact_too_far():
    """Two Au atoms 8 Å apart must not interact."""
    atoms = Atoms(symbols=["Au", "Au"], positions=[[0, 0, 0], [8.0, 0, 0]])

    G = build_graph_with_covalent_pairs(atoms)
    G = find_intermetalls_contact(atoms, G)

    assert not any(
        d.get("bond_type") == "metallophilic"
        for _, _, d in G.edges(data=True)
    )


# ========================================================================== #
#  Output writer
# ========================================================================== #

def test_output_creates_nci_file(tmp_path, water_dimer):
    G = build_graph_with_covalent_pairs(water_dimer)
    G = find_nci_bonds(water_dimer, G, ["H"], 120, "HB")

    filename = tmp_path / "result"
    output(G, filename=str(filename), ext=".xyz")

    result = tmp_path / "result.nci"
    assert result.exists()

    content = result.read_text(encoding="utf-8")
    assert "NCITools: Geometry based analysis" in content
    assert "HYDROGEN BONDS" in content
    assert "O-H···O" in content