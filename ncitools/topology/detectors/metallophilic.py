"""Metallophilic contacts from BCPs between closed-shell metal centres."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from ncitools.geometry.detectors.metallophilic import METALLOPHILIC_METALS
from ncitools.utils import get_symbol

from ..soft import metallophilic_score
from ..topology import _bcp_properties
from ._common import (
    _collect_classified_bcp_positions,
    _edge_has_type,
)


def find_metallophilic_interactions(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    position_tolerance: float = 0.10,
) -> nx.Graph:
    """
    Add metallophilic contacts from metal···metal BCPs.

    A (3,-1) BCP between two metals of
    :data:`~ncitools.geometry.detectors.metallophilic.METALLOPHILIC_METALS`
    is sufficient, provided the graph has no covalent or coordination edge
    between them.  No distance or angle gating is applied; the distance and
    the BCP properties are reported and the geometry confidence score is
    stored on the edge.
    """
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    classified_positions = _collect_classified_bcp_positions(G)
    tree = cKDTree(np.asarray(classified_positions)) if classified_positions else None
    tolerance_bohr = position_tolerance * ANGSTROM_TO_BOHR

    for bcp in tqdm(bcps, desc="Searching metallophilic contacts"):
        position = np.asarray(bcp["position"])
        if tree is not None:
            dist, _ = tree.query(position, k=1)
            if dist <= tolerance_bohr:
                continue

        i, j = int(bcp["atom1"]), int(bcp["atom2"])
        if i == j:
            continue
        if symbols[i] not in METALLOPHILIC_METALS:
            continue
        if symbols[j] not in METALLOPHILIC_METALS:
            continue
        if _edge_has_type(G, i, j, {"covalent", "coordination"}):
            continue

        distance = norm(coordinates[i] - coordinates[j]) * BOHR_TO_ANGSTROM

        attrs = {
            "bond_type": "metallophilic",
            "metal_1": int(i),
            "metal_2": int(j),
            "metal_sym_1": symbols[i],
            "metal_sym_2": symbols[j],
            "distance": float(distance),
            "confidence_score": metallophilic_score(
                float(distance), symbols[i], symbols[j],
            ),
        }
        attrs.update(_bcp_properties(bcp))
        G.add_edge(i, j, **attrs)

        classified_positions.append(position.copy())
        tree = cKDTree(np.asarray(classified_positions))

    return G
