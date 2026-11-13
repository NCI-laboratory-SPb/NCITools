"""Dihydrogen H···H contacts from BCPs."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from ncitools.utils import get_symbol

from ..soft import dihydrogen_score
from ..topology import _bcp_properties
from ._common import (
    _collect_classified_bcp_positions,
    _covalent_neighbors,
)


def _heavy_partner(
    G: nx.Graph,
    idx: int,
    symbols: List[str],
) -> Optional[int]:
    """Heavy atom a hydrogen is covalently bound to, if any."""
    for neighbour in _covalent_neighbors(G, idx, ("covalent",)):
        if symbols[neighbour] != "H":
            return neighbour
    return None


def find_dihydrogen_interactions(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    position_tolerance: float = 0.10,
    exclude_geminal: bool = True,
) -> nx.Graph:
    """
    Detect dihydrogen (H···H) contacts.

    A BCP whose two atoms are hydrogens that are not covalently bound is the
    only requirement: the contact may be very weak, so no distance cutoff is
    applied.  The confidence score is distance-only (a sigmoid centred on
    the sum of two BONDI hydrogen radii).

    With ``exclude_geminal`` the two hydrogens must not be attached to
    the same heavy atom: a geminal H-C-H pair is a covalent feature of
    one group, not a contact between two groups.
    """
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    classified_positions = _collect_classified_bcp_positions(G)
    tree = cKDTree(np.asarray(classified_positions)) if classified_positions else None
    tolerance_bohr = position_tolerance * ANGSTROM_TO_BOHR

    for bcp in tqdm(bcps, desc="Searching H···H contacts"):
        position = np.asarray(bcp["position"])
        if tree is not None:
            dist, _ = tree.query(position, k=1)
            if dist <= tolerance_bohr:
                continue

        i, j = int(bcp["atom1"]), int(bcp["atom2"])
        if i == j or symbols[i] != "H" or symbols[j] != "H":
            continue
        if i not in G or j not in G:
            continue

        distance = norm(coordinates[i] - coordinates[j]) * BOHR_TO_ANGSTROM

        donor1 = _heavy_partner(G, i, symbols)
        donor2 = _heavy_partner(G, j, symbols)
        if exclude_geminal and donor1 is not None and donor1 == donor2:
            continue

        attrs = {
            "bond_type": "dihydrogen",
            "atom1_num": int(i),
            "atom2_num": int(j),
            "atom1_symbol": symbols[i],
            "atom2_symbol": symbols[j],
            "h1_num": int(i),
            "h2_num": int(j),
            "donor1_num": None if donor1 is None else int(donor1),
            "donor2_num": None if donor2 is None else int(donor2),
            "donor1_symbol": None if donor1 is None else symbols[donor1],
            "donor2_symbol": None if donor2 is None else symbols[donor2],
            "distance": float(distance),
            "confidence_score": dihydrogen_score(float(distance)),
        }
        attrs.update(_bcp_properties(bcp))
        G.add_edge(i, j, **attrs)

        classified_positions.append(position.copy())
        tree = cKDTree(np.asarray(classified_positions))

    return G
