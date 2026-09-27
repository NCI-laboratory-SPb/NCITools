"""
n→π* interactions between a lone-pair donor and a carbonyl acceptor.

Following the geometric criteria of Burgi and Dunitz, an n→π* contact
between a lone-pair donor (X=O) and a carbonyl group (C=O) is accepted
when

* the O···C distance is shorter than the sum of the van der Waals radii;
* the X–O···C angle (the "Bürgi–Dunitz angle") lies inside a characteristic
  window, typically 90°–130°;
* the angle between the donor plane and the acceptor plane is close to
  perpendicular, typically 50°–90°.

All three criteria are combined as a product of soft scores.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
from ase import Atoms
from tqdm import tqdm

from ncitools.constants import BONDI
from ncitools.geometry.geom import best_fit_plane
from ncitools.geometry.soft import soft_less, soft_range


def find_carbonyl_interactions(
    atoms: Atoms,
    G: nx.Graph,
    distance_cutoff: Optional[float] = None,
    angle_min: float = 90.0,
    angle_max: float = 130.0,
    plane_angle_min: float = 50.0,
    plane_angle_max: float = 90.0,
    confidence_threshold: float = 0.8,
) -> nx.Graph:
    """
    Add carbonyl n→π* interaction edges to ``G``.

    Donors
    ------
    Any oxygen atom with at least one covalent neighbour.  The plane of
    the donor group is defined by the first covalent neighbour and that
    neighbour's covalent partners.

    Acceptors
    ---------
    Carbonyl groups C=O, i.e. oxygen atoms with exactly one covalent
    neighbour which is a carbon.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    G : nx.Graph
        Molecular graph.  Modified in place.
    distance_cutoff : float, optional
        Maximum O···C distance in ångströms.  Defaults to
        ``BONDI['C'] + BONDI['O']``.
    angle_min, angle_max : float
        Allowed Bürgi–Dunitz angle range (degrees).
    plane_angle_min, plane_angle_max : float
        Allowed donor/acceptor plane angle range (degrees).
    confidence_threshold : float
        Minimum product of soft scores required to accept an interaction.

    Returns
    -------
    nx.Graph
        The input graph with new ``carbonyl`` edges added.
    """
    if distance_cutoff is None:
        distance_cutoff = BONDI["C"] + BONDI["O"]

    symbols = atoms.get_chemical_symbols()

    donors = _collect_donors(atoms, G, symbols)
    acceptors = _collect_acceptors(atoms, G, symbols)

    for donor in tqdm(
        donors,
        desc="Searching for n→π* interactions with carbonyls",
    ):
        _match_donor_against_acceptors(
            atoms, G, donor, acceptors, symbols,
            distance_cutoff,
            angle_min, angle_max,
            plane_angle_min, plane_angle_max,
            confidence_threshold,
        )

    return G


# --------------------------------------------------------------------------- #
#  Donor / acceptor collection
# --------------------------------------------------------------------------- #

def _covalent_neighbours(G: nx.Graph, idx: int) -> list:
    """Heavy (non-virtual) covalent neighbours of ``idx``."""
    return [
        n for n in G.adj[idx]
        if not isinstance(n, str)
        and G.edges[idx, n].get("bond_type") == "covalent"
    ]


def _collect_donors(atoms: Atoms, G: nx.Graph, symbols) -> list:
    """
    Lone-pair donors: any oxygen with at least one covalent neighbour.

    For each donor, the plane of its host group is precomputed so that
    the plane-angle criterion can be evaluated later.
    """
    donors = []

    for o, symbol in enumerate(symbols):
        if symbol != "O":
            continue

        neighbours = _covalent_neighbours(G, o)
        if not neighbours:
            continue

        center = neighbours[0]
        donors.append({
            "oxygen": o,
            "center": center,
            "normal": _group_plane_normal(atoms, G, center, o),
        })

    return donors


def _collect_acceptors(atoms: Atoms, G: nx.Graph, symbols) -> list:
    """Carbonyl acceptors: O with exactly one covalent neighbour, a carbon."""
    acceptors = []

    for o, symbol in enumerate(symbols):
        if symbol != "O":
            continue

        neighbours = _covalent_neighbours(G, o)
        if len(neighbours) != 1:
            continue

        carbon = neighbours[0]
        if symbols[carbon] != "C":
            continue

        acceptors.append({
            "oxygen": o,
            "carbon": carbon,
            "normal": _group_plane_normal(atoms, G, carbon, o),
        })

    return acceptors


def _group_plane_normal(
    atoms: Atoms,
    G: nx.Graph,
    center: int,
    oxygen: int,
) -> Optional[np.ndarray]:
    """
    Unit normal of the plane defined by ``center``, ``oxygen`` and the
    remaining covalent partners of ``center``.

    Returns ``None`` if fewer than three distinct points are available.
    """
    coords = [atoms.positions[center], atoms.positions[oxygen]]

    for nbr in _covalent_neighbours(G, center):
        if nbr != oxygen:
            coords.append(atoms.positions[nbr])

    if len(coords) < 3:
        return None

    _, normal = best_fit_plane(np.asarray(coords))
    return normal


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #

def _match_donor_against_acceptors(
    atoms: Atoms,
    G: nx.Graph,
    donor: dict,
    acceptors: list,
    symbols,
    distance_cutoff: float,
    angle_min: float,
    angle_max: float,
    plane_angle_min: float,
    plane_angle_max: float,
    confidence_threshold: float,
) -> None:
    """Test every carbonyl acceptor against a single donor."""
    o_d = donor["oxygen"]
    donor_center = donor["center"]
    n_d = donor["normal"]

    for acceptor in acceptors:
        o_a = acceptor["oxygen"]
        c_a = acceptor["carbon"]
        n_a = acceptor["normal"]

        # Donor and acceptor may not be the same oxygen.
        if o_d == o_a:
            continue

        distance = atoms.get_distance(o_d, c_a, mic=True)
        score_dist = soft_less(distance, distance_cutoff)
        if score_dist < 0.01:
            continue

        bd_angle = atoms.get_angle(o_d, c_a, o_a, mic=True)
        score_bd = soft_range(bd_angle, angle_min, angle_max)

        if n_d is not None and n_a is not None:
            plane_angle = float(np.degrees(np.arccos(np.clip(
                abs(np.dot(n_d, n_a)), 0.0, 1.0
            ))))
            score_plane = soft_range(
                plane_angle, plane_angle_min, plane_angle_max
            )
        else:
            plane_angle = None
            score_plane = 1.0

        total = score_dist * score_bd * score_plane
        if total < confidence_threshold:
            continue

        G.add_edge(
            o_d, c_a,
            bond_type="carbonyl",
            donor_oxygen=o_d,
            donor_center=donor_center,
            acceptor_carbon=c_a,
            acceptor_oxygen=o_a,
            distance=float(distance),
            burgi_dunitz_angle=float(bd_angle),
            plane_angle=plane_angle,
            confidence_score=float(total),
        )