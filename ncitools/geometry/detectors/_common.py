"""
Helpers shared between detectors.
"""

from __future__ import annotations

from typing import Union

import networkx as nx
from ase import Atoms

from ncitools.constants import *
from ncitools.geometry.soft import soft_range


PNICTOGENS = frozenset({"N", "P", "As", "Sb", "Bi"})
CHALCOGENS = frozenset({"O", "S", "Se", "Te"})
HALOGENS = frozenset({"F", "Cl", "Br", "I"})

# ``None``  -> skip the acceptor entirely;
# ``"skip"`` -> no angular criterion can be evaluated.
AngleRange = Union[tuple, None, str]


def vdw_radius(element: str, radii: dict) -> float:
    """
    Van der Waals radius for ``element``.

    Falls back to the covalent radius plus a fixed increment for elements
    that are missing from ``radii`` (typically metals).
    """
    try:
        return radii[element]
    except KeyError:
        return CHARRY_TKATCHENKO[element]


def acceptor_angle_range(element: str, num_neighbours: int) -> AngleRange:
    """
    Return the allowed X–A–Y angle range (in degrees) for an acceptor
    atom A of the given element and coordination number.

    The ranges encode the hybridisation of A:

    * pnictogens: sp (linear), sp², sp³;
    * chalcogens: sp³, sp³ with one lone pair, hypervalent;
    * halogens:  isolated anion or covalently bound.
    """
    if element in PNICTOGENS:
        if num_neighbours == 1:
            return 150.0, 180.0
        if num_neighbours == 2:
            return 100.0, 145.0
        if num_neighbours == 3:
            return 80.0, 135.0
        return None

    if element in CHALCOGENS:
        if num_neighbours == 1:
            return 100.0, 145.0
        if num_neighbours == 2:
            return 80.0, 135.0
        if num_neighbours == 3:
            if element == "O":
                return None
            return 80.0, 120.0
        return None

    if element in HALOGENS:
        if num_neighbours == 0:
            return "skip"
        return 80.0, 120.0

    return None


def soft_acceptor_angle_score(
    atoms: Atoms,
    G: nx.Graph,
    x: int,
    a: int,
    symbols,
) -> tuple:
    """
    Soft score for the X–A–Y angular criterion at acceptor ``a``.

    Returns
    -------
    (score, angle)
        ``score`` in [0, 1]; ``angle`` is the best matching Y angle in
        degrees, or ``None`` if the criterion is skipped.
    """
    a_el = symbols[a]
    num_neighbours = G.nodes[a]["num_neighbours"]
    angle_range = acceptor_angle_range(a_el, num_neighbours)

    if angle_range is None:
        return 0.0, None
    if angle_range == "skip":
        return 1.0, None

    angle_min, angle_max = angle_range
    best_score = 0.0
    best_angle = None

    for y in G.adj[a]:
        if isinstance(y, str):
            continue
        if G.edges[a, y]["bond_type"] != "covalent":
            continue

        angle_xay = atoms.get_angle(x, a, y, mic=True)
        score = soft_range(angle_xay, angle_min, angle_max)

        if score > best_score:
            best_score = score
            best_angle = angle_xay

    return best_score, best_angle