"""Fallback: represent remaining NCI BCPs as unclassified contacts."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR
from ncitools.utils import get_symbol

from ..topology import _bcp_properties
from ._common import (
    _aromatic_membership,
    _collect_classified_bcp_positions,
    _edge_has_type,
    _nearest_pair_from_bcp,
)


def find_unclassified_interactions(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    position_tolerance: float = 0.10,
) -> nx.Graph:
    """Add remaining NCI BCPs as unclassified contacts."""
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    classified_positions = _collect_classified_bcp_positions(G)
    tree = cKDTree(np.asarray(classified_positions)) if classified_positions else None
    tolerance_bohr = position_tolerance * ANGSTROM_TO_BOHR

    for bcp in tqdm(bcps, desc="Collecting unclassified NCI"):
        position = np.asarray(bcp["position"])
        if tree is not None:
            dist, _ = tree.query(position, k=1)
            if dist <= tolerance_bohr:
                continue

        i, j = _nearest_pair_from_bcp(bcp, atoms)
        if i == j:
            continue

        if _edge_has_type(
            G, i, j,
            {"HB", "XB", "ChB", "PnB", "TetB", "stacking",
             "H-pi", "n-pi", "X-pi", "ion-pi", "metallophilic",
             "carbonyl", "dihydrogen"},
        ):
            continue

        aromatic1, rings1 = _aromatic_membership(G, i)
        aromatic2, rings2 = _aromatic_membership(G, j)
        if aromatic1 and aromatic2 and set(rings1) & set(rings2):
            continue

        if aromatic1 and aromatic2:
            skip = any(
                _edge_has_type(G, ri, rj, {"stacking"})
                for ri in rings1 for rj in rings2
            )
            if skip:
                continue

        distance = norm(coordinates[i] - coordinates[j]) * BOHR_TO_ANGSTROM
        attrs = {
            "bond_type": "unclassified",
            "atom1_num": int(i),
            "atom2_num": int(j),
            "atom1_symbol": symbols[i],
            "atom2_symbol": symbols[j],
            "distance": float(distance),
        }
        attrs.update(_bcp_properties(bcp))
        G.add_edge(i, j, **attrs)

        classified_positions.append(position.copy())
        tree = cKDTree(np.asarray(classified_positions))

    return G