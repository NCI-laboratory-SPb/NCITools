"""Aromatic π interactions: π···π stacking, X–H···π, lone pair···π."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from ncitools.constants import BOHR_TO_ANGSTROM
from ncitools.utils import get_symbol

from ..topology import _angle, _bcp_properties
from ._common import (
    _aromatic_membership,
    _covalent_neighbors,
    _nearest_aromatic_ring,
    _nearest_pair_from_bcp,
)


# ---------------------------------------------------------------------------
# Classification of a single BCP as an aromatic interaction
# ---------------------------------------------------------------------------

def _classify_aromatic_nci(
    atom1_idx: int,
    atom2_idx: int,
    bcp: Dict,
    atoms: List[Tuple[int, np.ndarray]],
    G: nx.Graph,
    angle_threshold: float = 120.0,
    stacking_dist_threshold: float = 5.5,
    offset_threshold: float = 2.0,
) -> Optional[Dict]:
    """Classify aromatic π interactions (stacking, H-pi, n-pi)."""
    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    p1 = coordinates[atom1_idx]
    p2 = coordinates[atom2_idx]

    aromatic1, rings1 = _aromatic_membership(G, atom1_idx)
    aromatic2, rings2 = _aromatic_membership(G, atom2_idx)

    if not aromatic1 and not aromatic2:
        return None

    # --- Parallel stacking (both atoms in aromatic rings) ---
    if aromatic1 and aromatic2:
        ring1 = _nearest_aromatic_ring(G, atom1_idx, p1)
        ring2 = _nearest_aromatic_ring(G, atom2_idx, p2)
        if ring1 is None or ring2 is None or ring1 == ring2:
            return None

        c1 = np.asarray(G.nodes[ring1]["position"])
        c2 = np.asarray(G.nodes[ring2]["position"])
        n1 = np.asarray(G.nodes[ring1]["normal"])
        n2 = np.asarray(G.nodes[ring2]["normal"])

        diff = c2 - c1
        centroid_distance = norm(diff) * BOHR_TO_ANGSTROM
        if centroid_distance > stacking_dist_threshold:
            return None

        ring_angle = np.degrees(
            np.arccos(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))
        )

        if ring_angle <= 30.0:
            offset = norm(
                diff - np.dot(diff, n1) * n1
            ) * BOHR_TO_ANGSTROM
            if offset > offset_threshold:
                return None
            attrs = {
                "bond_type": "stacking",
                "ring_1": G.nodes[ring1]["cycle"],
                "ring_2": G.nodes[ring2]["cycle"],
                "centroid_distance": float(centroid_distance),
                "offset": float(offset),
                "angle": float(ring_angle),
            }
            attrs.update(_bcp_properties(bcp))
            return {"u": ring1, "v": ring2, "attrs": attrs}

        return None

    # --- Only one side aromatic ---
    if aromatic1:
        aromatic_idx, other_idx = atom1_idx, atom2_idx
        rings = rings1
    else:
        aromatic_idx, other_idx = atom2_idx, atom1_idx
        rings = rings2

    ring = min(
        rings,
        key=lambda r: norm(
            coordinates[aromatic_idx] - G.nodes[r]["position"]
        ),
    )
    centre = np.asarray(G.nodes[ring]["position"])
    normal = np.asarray(G.nodes[ring]["normal"])
    radius = float(G.nodes[ring]["radius"])

    other_symbol = symbols[other_idx]

    # --- X–H···π ---
    if other_symbol == "H":
        h_idx = other_idx
        neighbours = _covalent_neighbors(G, h_idx)
        if len(neighbours) != 1:
            return None
        x_idx = neighbours[0]
        x_symbol = symbols[x_idx]
        if x_symbol not in {"C", "N", "O", "S"}:
            return None

        H = coordinates[h_idx]
        X = coordinates[x_idx]
        h_centroid_distance = norm(H - centre) * BOHR_TO_ANGSTROM
        if h_centroid_distance > 3.5:
            return None

        angle = _angle(X, H, centre)
        if angle is None or angle < angle_threshold:
            return None

        projection = H - centre
        projection = projection - np.dot(projection, normal) * normal
        projection_distance = norm(projection) * BOHR_TO_ANGSTROM
        radius_angstrom = radius * BOHR_TO_ANGSTROM
        if projection_distance > radius_angstrom + 0.25:
            return None

        attrs = {
            "bond_type": "H-pi",
            "x_num": int(x_idx),
            "h_num": int(h_idx),
            "symbol_x": x_symbol,
            "ring": G.nodes[ring]["cycle"],
            "h_centroid_distance": float(h_centroid_distance),
            "projection_distance": float(projection_distance),
            "angle": float(angle),
        }
        attrs.update(_bcp_properties(bcp))
        return {"u": h_idx, "v": ring, "attrs": attrs}

    # --- Lone pair···π ---
    if other_symbol not in {"N", "O", "S", "Se"}:
        return None

    lp_idx = other_idx
    neighbours = _covalent_neighbors(G, lp_idx)
    if not neighbours:
        return None

    donor = coordinates[lp_idx]
    distance = norm(donor - centre) * BOHR_TO_ANGSTROM
    if not (1.5 <= distance <= 3.6):
        return None

    vec = donor - centre
    vec_norm = norm(vec)
    if vec_norm <= 1.0e-12:
        return None

    angle_to_normal = np.degrees(
        np.arccos(np.clip(abs(np.dot(vec, normal)) / vec_norm, 0.0, 1.0))
    )
    if angle_to_normal > 45.0:
        return None

    best = None
    for neighbour_idx in neighbours:
        angle = _angle(centre, donor, coordinates[neighbour_idx])
        if angle is None:
            continue
        lp_angle = 180.0 - angle
        if 100.0 <= lp_angle <= 130.0:
            candidate = (abs(lp_angle - 115.0), lp_angle, neighbour_idx)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return None

    _, best_angle, best_neighbour = best

    attrs = {
        "bond_type": "n-pi",
        "lp_num": int(lp_idx),
        "symbol_lp": other_symbol,
        "ring": G.nodes[ring]["cycle"],
        "lp_centroid_distance": float(distance),
        "angle_to_normal": float(angle_to_normal),
        "angle": float(best_angle),
        "lp_neighbour_num": int(best_neighbour),
    }
    attrs.update(_bcp_properties(bcp))
    return {"u": lp_idx, "v": ring, "attrs": attrs}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_nci_with_aromatic(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    angle_threshold: float = 120.0,
    stacking_dist_threshold: float = 5.5,
    offset_threshold: float = 2.0,
) -> nx.Graph:
    """Add aromatic NCI interactions (stacking, H-pi, n-pi) to G."""
    if not bcps:
        return G

    for bcp in tqdm(bcps, desc="Classifying aromatic BCPs"):
        atom1_idx, atom2_idx = _nearest_pair_from_bcp(bcp, atoms)
        if atom1_idx == atom2_idx:
            continue

        result = _classify_aromatic_nci(
            atom1_idx, atom2_idx, bcp, atoms, G,
            angle_threshold=angle_threshold,
            stacking_dist_threshold=stacking_dist_threshold,
            offset_threshold=offset_threshold,
        )
        if result is not None:
            G.add_edge(result["u"], result["v"], **result["attrs"])

    return G