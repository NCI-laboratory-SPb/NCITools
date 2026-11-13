"""Aromatic π interactions: stacking, X–H···π, lone pair···π, X···π, ion···π."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from ncitools.constants import BOHR_TO_ANGSTROM, BONDI, METALS
from ncitools.geometry.detectors._common import acceptor_angle_range
from ncitools.utils import get_symbol

from ..soft import (
    HB_ANGLE_THRESHOLD,
    PI_ANGLE_TO_NORMAL_THRESHOLD,
    PI_CLOUD_RADII,
    SIGMA_HOLE_ANGLE_THRESHOLD,
    STACKING_ANGLE_THRESHOLD,
    STACKING_DISTANCE_THRESHOLD,
    STACKING_OFFSET_THRESHOLD,
    acceptor_angle_score,
    h_pi_score,
    ion_pi_score,
    n_pi_score,
    stacking_score,
    x_pi_score,
)
from ..topology import _angle, _bcp_properties
from ._common import (
    _aromatic_membership,
    _covalent_neighbors,
    _nearest_aromatic_ring,
    _nearest_pair_from_bcp,
)


# Halogens that are treated as potential bare anions for ion···π.  The
# HALOGENS table in ncitools.constants also contains hydrogen, which must not
# take this path (a proton is handled by the X–H···π detector).
_ION_PI_HALOGENS = frozenset({"F", "Cl", "Br", "I"})
_X_PI_HALOGENS = frozenset({"Cl", "Br", "I"})


def _is_bare_ion(G: nx.Graph, idx: int, symbol: str) -> bool:
    """
    True when ``idx`` is an ion that carries no chemical bonds.

    A metal must have no covalent bonds (coordination is allowed: a metal
    centre in a complex can still stack on a ring).  A halogen must have
    neither covalent nor coordination bonds, otherwise the BCP belongs to
    an R–X···π contact instead.
    """
    if symbol in METALS:
        return not _covalent_neighbors(G, idx, ("covalent",))
    if symbol in _ION_PI_HALOGENS:
        return not _covalent_neighbors(G, idx)
    return False


# ---------------------------------------------------------------------------
# Classification of a single BCP as an aromatic interaction
# ---------------------------------------------------------------------------

def _classify_aromatic_nci(
    atom1_idx: int,
    atom2_idx: int,
    bcp: Dict,
    atoms: List[Tuple[int, np.ndarray]],
    G: nx.Graph,
    angle_threshold: float = HB_ANGLE_THRESHOLD,
    stacking_dist_threshold: float = STACKING_DISTANCE_THRESHOLD,
    offset_threshold: float = STACKING_OFFSET_THRESHOLD,
    skipped: frozenset = frozenset(),
) -> Optional[Dict]:
    """Classify aromatic π interactions from a single BCP."""
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
        if "stacking" in skipped:
            return None

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

        if ring_angle <= STACKING_ANGLE_THRESHOLD:
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
                "confidence_score": stacking_score(
                    centroid_distance, float(ring_angle), offset,
                ),
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
    radius_angstrom = radius * BOHR_TO_ANGSTROM
    ring_cycle = set(G.nodes[ring]["cycle"])

    other_symbol = symbols[other_idx]

    # --- X–H···π ---
    if other_symbol == "H":
        if "H-pi" in skipped:
            return None

        h_idx = other_idx
        neighbours = _covalent_neighbors(G, h_idx, ("covalent",))
        if len(neighbours) != 1:
            return None
        x_idx = neighbours[0]
        x_symbol = symbols[x_idx]

        H = coordinates[h_idx]
        X = coordinates[x_idx]
        h_centroid_distance = norm(H - centre) * BOHR_TO_ANGSTROM
        if h_centroid_distance > PI_CLOUD_RADII + BONDI["H"]:
            return None

        angle = _angle(X, H, centre)
        if angle is None or angle < angle_threshold:
            return None

        projection = H - centre
        projection = projection - np.dot(projection, normal) * normal
        projection_distance = norm(projection) * BOHR_TO_ANGSTROM

        attrs = {
            "bond_type": "H-pi",
            "x_num": int(x_idx),
            "h_num": int(h_idx),
            "symbol_x": x_symbol,
            "ring": G.nodes[ring]["cycle"],
            "h_centroid_distance": float(h_centroid_distance),
            "projection_distance": float(projection_distance),
            "angle": float(angle),
            "confidence_score": h_pi_score(
                h_centroid_distance, float(angle),
                float(projection_distance), radius_angstrom,
            ),
        }
        attrs.update(_bcp_properties(bcp))
        return {"u": h_idx, "v": ring, "attrs": attrs}

    # --- Ion···π ---
    if _is_bare_ion(G, other_idx, other_symbol):
        if "ion-pi" in skipped:
            return None

        ion = coordinates[other_idx]
        vec = ion - centre
        vec_norm = norm(vec)
        if vec_norm <= 1.0e-12:
            return None

        distance = vec_norm * BOHR_TO_ANGSTROM
        angle_to_normal = float(np.degrees(np.arccos(np.clip(
            np.dot(vec, normal) / vec_norm, -1.0, 1.0,
        ))))

        attrs = {
            "bond_type": "ion-pi",
            "ion_num": int(other_idx),
            "symbol_ion": other_symbol,
            "ring": G.nodes[ring]["cycle"],
            "ion_centroid_distance": float(distance),
            "angle_to_normal": angle_to_normal,
            "confidence_score": ion_pi_score(distance, angle_to_normal),
        }
        attrs.update(_bcp_properties(bcp))
        return {"u": other_idx, "v": ring, "attrs": attrs}

    # --- R–X···π ---
    if other_symbol in _X_PI_HALOGENS:
        if "X-pi" in skipped:
            return None

        neighbours = _covalent_neighbors(G, other_idx)
        if len(neighbours) != 1:
            return None
        r_idx = neighbours[0]
        if other_idx in ring_cycle or r_idx in ring_cycle:
            return None

        X = coordinates[other_idx]
        R = coordinates[r_idx]

        x_centroid = norm(X - centre) * BOHR_TO_ANGSTROM
        vec = centre - X
        vec_norm = norm(vec)
        if vec_norm <= 1.0e-12:
            return None

        angle_to_normal = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(vec, normal)) / vec_norm, 0.0, 1.0,
        ))))
        angle = _angle(R, X, centre)
        if angle is None or angle < SIGMA_HOLE_ANGLE_THRESHOLD:
            return None

        attrs = {
            "bond_type": "X-pi",
            "r_num": int(r_idx),
            "x_num": int(other_idx),
            "symbol_r": symbols[r_idx],
            "symbol_x": other_symbol,
            "ring": G.nodes[ring]["cycle"],
            "x_centroid_distance": float(x_centroid),
            "angle": float(angle),
            "angle_to_normal": angle_to_normal,
            "confidence_score": x_pi_score(
                other_symbol, x_centroid, float(angle), angle_to_normal,
            ),
        }
        attrs.update(_bcp_properties(bcp))
        return {"u": other_idx, "v": ring, "attrs": attrs}

    # --- Lone pair···π ---
    lp_idx = other_idx
    neighbours = _covalent_neighbors(G, lp_idx, ("covalent",))
    if not neighbours:
        return None

    num_neighbours = G.nodes[lp_idx].get("num_neighbours", len(neighbours))
    if acceptor_angle_range(other_symbol, num_neighbours) is None:
        return None
    if "n-pi" in skipped:
        return None

    donor = coordinates[lp_idx]
    distance = norm(donor - centre) * BOHR_TO_ANGSTROM

    vec = donor - centre
    vec_norm = norm(vec)
    if vec_norm <= 1.0e-12:
        return None

    angle_to_normal = np.degrees(
        np.arccos(np.clip(abs(np.dot(vec, normal)) / vec_norm, 0.0, 1.0))
    )
    if angle_to_normal > PI_ANGLE_TO_NORMAL_THRESHOLD:
        return None

    projection = vec - np.dot(vec, normal) * normal
    projection_distance = float(norm(projection) * BOHR_TO_ANGSTROM)

    # Y–lp···π angle: the same convention as the geometry detector (angle at
    # the donor between the donor→centroid and donor→Y directions).
    best = None
    for neighbour_idx in neighbours:
        angle = _angle(centre, donor, coordinates[neighbour_idx])
        if angle is None:
            continue
        score_y = acceptor_angle_score(other_symbol, num_neighbours, angle)
        candidate = (score_y, angle, neighbour_idx)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None or best[0] <= 0.0:
        return None

    score_y, best_angle, best_neighbour = best

    attrs = {
        "bond_type": "n-pi",
        "lp_num": int(lp_idx),
        "symbol_lp": other_symbol,
        "ring": G.nodes[ring]["cycle"],
        "lp_centroid_distance": float(distance),
        "angle_to_normal": float(angle_to_normal),
        "projection_distance": projection_distance,
        "angle": float(best_angle),
        "lp_neighbour_num": int(best_neighbour),
        "confidence_score": n_pi_score(
            other_symbol, float(distance), float(angle_to_normal),
            projection_distance, radius_angstrom, score_y,
        ),
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
    angle_threshold: float = HB_ANGLE_THRESHOLD,
    stacking_dist_threshold: float = STACKING_DISTANCE_THRESHOLD,
    offset_threshold: float = STACKING_OFFSET_THRESHOLD,
    skipped: frozenset = frozenset(),
) -> nx.Graph:
    """Add aromatic NCI interactions to G.

    ``skipped`` disables individual families (``stacking``, ``H-pi``,
    ``n-pi``, ``X-pi``, ``ion-pi``); their BCPs then fall through to the
    ``unclassified`` detector.
    """
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
            skipped=skipped,
        )
        if result is None:
            continue

        u, v, attrs = result["u"], result["v"], result["attrs"]
        # Two BCPs may map onto the same ion/halogen···ring pair; keep the
        # stronger one instead of silently keeping whichever came last.
        if (
            G.has_edge(u, v)
            and G.edges[u, v].get("bond_type") == attrs["bond_type"]
            and attrs.get("confidence_score", 0.0)
            <= G.edges[u, v].get("confidence_score", 0.0)
        ):
            continue
        G.add_edge(u, v, **attrs)

    return G
