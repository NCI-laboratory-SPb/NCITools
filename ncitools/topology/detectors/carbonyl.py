"""Carbonyl n→π* interactions from BCPs."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR
from ncitools.utils import get_symbol

from ..topology import _angle, _bcp_properties
from ._common import (
    _collect_classified_bcp_positions,
    _covalent_neighbors,
    _fit_ring_plane,
)


def _identify_carbonyl_carbons(
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    carbonyl_max_distance: float = 1.35,
) -> Dict[int, int]:
    """
    Identify carbonyl carbons from the covalent graph.

    A C atom is carbonyl-like if:
        - it has a covalent O neighbour;
        - the shortest C–O distance ≤ threshold.
    """
    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    carbonyls: Dict[int, int] = {}
    for idx, symbol in enumerate(symbols):
        if symbol != "C":
            continue
        neighbours = _covalent_neighbors(G, idx)
        oxygen_neighbours = [
            n for n in neighbours
            if symbols[n] == "O"
            and norm(coordinates[idx] - coordinates[n]) * BOHR_TO_ANGSTROM
            <= carbonyl_max_distance
        ]
        if oxygen_neighbours:
            carbonyl_oxygen = min(
                oxygen_neighbours,
                key=lambda j: norm(coordinates[idx] - coordinates[j]),
            )
            carbonyls[idx] = carbonyl_oxygen
    return carbonyls


def find_carbonyl_interactions(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    distance_cutoff: float = 3.22,
    angle_min: float = 90.0,
    angle_max: float = 130.0,
    position_tolerance: float = 0.10,
) -> nx.Graph:
    """
    Detect n→π* interactions with a carbonyl C=O.

    Geometry:
        donor Y ... C=O
        Y···C ≤ 3.22 Å
        Y···C–O angle = 95–125°
    """
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]
    carbonyls = _identify_carbonyl_carbons(G, atoms)

    classified_positions = _collect_classified_bcp_positions(G)
    tree = cKDTree(np.asarray(classified_positions)) if classified_positions else None
    tolerance_bohr = position_tolerance * ANGSTROM_TO_BOHR

    for bcp in tqdm(bcps, desc="Searching n→π* interactions"):
        position = np.asarray(bcp["position"])
        if tree is not None:
            dist, _ = tree.query(position, k=1)
            if dist <= tolerance_bohr:
                continue

        i = int(bcp["atom1"])
        j = int(bcp["atom2"])

        if symbols[i] == "O" and j in carbonyls:
            donor_oxygen, acceptor_carbon = i, j
        elif symbols[j] == "O" and i in carbonyls:
            donor_oxygen, acceptor_carbon = j, i
        else:
            continue

        carbonyl_oxygen = carbonyls[acceptor_carbon]
        if donor_oxygen == carbonyl_oxygen:
            continue

        distance = norm(
            coordinates[donor_oxygen] - coordinates[acceptor_carbon]
        ) * BOHR_TO_ANGSTROM
        if distance > distance_cutoff:
            continue

        angle = _angle(
            coordinates[donor_oxygen],
            coordinates[acceptor_carbon],
            coordinates[carbonyl_oxygen],
        )
        if angle is None or not (angle_min <= angle <= angle_max):
            continue

        carbon_neighbours = _covalent_neighbors(G, acceptor_carbon)
        substituents = [
            n for n in carbon_neighbours if n != carbonyl_oxygen
        ]
        plane_angle = None
        if len(substituents) >= 2:
            coords = np.asarray([coordinates[n] for n in substituents])
            plane = _fit_ring_plane(coords)
            if plane is not None:
                _, normal, _, _ = plane
                vector = coordinates[donor_oxygen] - coordinates[acceptor_carbon]
                vnorm = norm(vector)
                if vnorm > 1.0e-12:
                    plane_angle = np.degrees(
                        np.arcsin(
                            np.clip(
                                abs(np.dot(vector, normal)) / vnorm,
                                0.0, 1.0,
                            )
                        )
                    )

        attrs = {
            "bond_type": "carbonyl",
            "donor_oxygen": int(donor_oxygen),
            "acceptor_carbon": int(acceptor_carbon),
            "carbonyl_oxygen": int(carbonyl_oxygen),
            "donor_symbol": symbols[donor_oxygen],
            "acceptor_symbol": symbols[acceptor_carbon],
            "distance": float(distance),
            "burgi_dunitz_angle": float(angle),
            "plane_angle": None if plane_angle is None else float(plane_angle),
        }
        attrs.update(_bcp_properties(bcp))
        G.add_edge(donor_oxygen, acceptor_carbon, **attrs)

        classified_positions.append(position.copy())
        tree = cKDTree(np.asarray(classified_positions))

    return G