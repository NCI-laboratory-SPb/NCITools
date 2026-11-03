"""σ-hole interactions from BCPs: HB, XB, ChB, PnB."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, BONDI
from ncitools.utils import get_symbol

from ..topology import _angle, _bcp_properties
from ._common import (
    _collect_classified_bcp_positions,
    _covalent_neighbors,
    _nearest_pair_from_bcp,
)


def find_sigma_hole_interactions(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    position_tolerance: float = 0.10,
) -> nx.Graph:
    """
    Add HB / XB / ChB / PnB interactions from BCPs.

    BCPs whose position is already represented in the graph
    (e.g. by the aromatic detector) are skipped.
    """
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    classified_positions = _collect_classified_bcp_positions(G)
    tree = cKDTree(np.asarray(classified_positions)) if classified_positions else None
    tolerance_bohr = position_tolerance * ANGSTROM_TO_BOHR

    for bcp in tqdm(bcps, desc="Classifying σ-hole BCPs"):
        position = np.asarray(bcp["position"])
        if tree is not None:
            dist, _ = tree.query(position, k=1)
            if dist <= tolerance_bohr:
                continue

        atom1_idx, atom2_idx = _nearest_pair_from_bcp(bcp, atoms)
        if atom1_idx == atom2_idx:
            continue

        symbol1 = symbols[atom1_idx]
        symbol2 = symbols[atom2_idx]

        is_h1 = symbol1 == "H"
        is_h2 = symbol2 == "H"
        if is_h1 and is_h2:
            continue

        # --- Hydrogen bonds ---
        if is_h1 or is_h2:
            if is_h1:
                h_idx, acceptor_idx = atom1_idx, atom2_idx
            else:
                h_idx, acceptor_idx = atom2_idx, atom1_idx

            donor_neighbours = _covalent_neighbors(G, h_idx)
            if len(donor_neighbours) != 1:
                continue
            donor_idx = donor_neighbours[0]
            donor_symbol = symbols[donor_idx]
            acceptor_symbol = symbols[acceptor_idx]

            D = coordinates[donor_idx]
            H = coordinates[h_idx]
            A = coordinates[acceptor_idx]

            D_H = norm(H - D) * BOHR_TO_ANGSTROM
            H_A = norm(H - A) * BOHR_TO_ANGSTROM
            D_A = norm(D - A) * BOHR_TO_ANGSTROM
            hb_angle = _angle(D, H, A)

            attrs = {
                "bond_type": "HB",
                "donor_num": int(donor_idx),
                "hydrogen_num": int(h_idx),
                "acceptor_num": int(acceptor_idx),
                "donor_symbol": donor_symbol,
                "acceptor_symbol": acceptor_symbol,
                "distance": float(H_A),
                "heavy_atom_distance": float(D_A),
                "angle": float(hb_angle),
                "h_donor_distance": float(D_H),
            }
            attrs.update(_bcp_properties(bcp))
            G.add_edge(h_idx, acceptor_idx, **attrs)
            continue

        # --- Heavy-atom σ-hole bonds ---
        distance = norm(
            coordinates[atom1_idx] - coordinates[atom2_idx]
        ) * BOHR_TO_ANGSTROM
        r1 = BONDI.get(symbol1, None)
        r2 = BONDI.get(symbol2, None)
        if r1 is None or r2 is None:
            continue

        d1 = norm(
            coordinates[atom1_idx] - np.asarray(bcp["position"])
        ) * BOHR_TO_ANGSTROM
        d2 = norm(
            coordinates[atom2_idx] - np.asarray(bcp["position"])
        ) * BOHR_TO_ANGSTROM

        if d1 / r1 <= d2 / r2:
            electrophile_idx, nucleophile_idx = atom1_idx, atom2_idx
        else:
            electrophile_idx, nucleophile_idx = atom2_idx, atom1_idx

        electrophile_symbol = symbols[electrophile_idx]
        if electrophile_symbol in {"F", "Cl", "Br", "I"}:
            bond_type = "XB"
        elif electrophile_symbol in {"O", "S", "Se", "Te"}:
            bond_type = "ChB"
        elif electrophile_symbol in {"N", "P", "As", "Sb", "Bi"}:
            bond_type = "PnB"
        else:
            continue

        electrophile_neighbours = _covalent_neighbors(G, electrophile_idx)
        if not electrophile_neighbours:
            continue

        E = coordinates[electrophile_idx]
        Nu = coordinates[nucleophile_idx]
        candidate_angles = []
        for neighbour_idx in electrophile_neighbours:
            angle = _angle(coordinates[neighbour_idx], E, Nu)
            if angle is not None:
                candidate_angles.append((abs(180.0 - angle), angle, neighbour_idx))

        if not candidate_angles:
            continue

        _, contact_angle, neighbour_num = min(candidate_angles, key=lambda x: x[0])
        if contact_angle < 140.0:
            continue

        attrs = {
            "bond_type": bond_type,
            "neighbor_num": int(neighbour_num),
            "electrophile_num": int(electrophile_idx),
            "nucleophile_num": int(nucleophile_idx),
            "neighbor_symbol": symbols[neighbour_num],
            "electrophile_symbol": electrophile_symbol,
            "nucleophile_symbol": symbols[nucleophile_idx],
            "distance": float(distance),
            "angle": float(contact_angle),
        }
        attrs.update(_bcp_properties(bcp))
        G.add_edge(atom1_idx, atom2_idx, **attrs)

        classified_positions.append(position.copy())
        tree = cKDTree(np.asarray(classified_positions))

    return G