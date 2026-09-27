"""
Interactions involving aromatic rings.

Four families are detected here:

* π···π stacking between two aromatic centres;
* X–H···π contacts between a hydrogen and the ring face;
* n···π contacts between a lone-pair donor and the ring face;
* ion···π contacts between a bare ion and the ring face.

Each family is implemented as a private helper; the public entry point
:func:`find_nci_with_aromatic` simply chains them.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
from ase import Atoms
from tqdm import tqdm

from ncitools.constants import BONDI, RADII, CHARRY_TKATCHENKO
from ncitools.geometry.detectors._common import (
    acceptor_angle_range, vdw_radius
)
from ncitools.geometry.soft import soft_greater, soft_less, soft_range


def find_nci_with_aromatic(
    atoms: Atoms,
    G: nx.Graph,
    radii: Optional[dict] = None,
    angle_threshold: float = 110.0,
    stacking_dist_threshold: float = 5.5,
    offset_threshold: float = 2.8,
    angle_stacking_threshold: float = 30.0,
    pi_cloud_radii: float = 3.0,
    cutoff_ion_pi: float = 5.0,
    angle_ion_pi_threshold: float = 30.0,
    confidence_threshold: float = 0.8,
) -> nx.Graph:
    """
    Detect π···π, X–H···π, n···π and ion···π interactions.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    G : nx.Graph
        Molecular graph; must already contain aromatic-centre nodes.
        Modified in place.
    radii : dict, optional
        Van der Waals radii.  Defaults to
        :data:`ncitools.constants.BONDI`.
    angle_threshold : float
        Minimum X–H···π angle in degrees.
    stacking_dist_threshold : float
        Maximum centroid–centroid distance for π-stacking (Å).
    offset_threshold : float
        Maximum lateral offset between parallel rings (Å).
    angle_stacking_threshold : float
        Maximum inter-plane angle for parallel stacking (degrees).
    pi_cloud_radii : float
        Effective thickness of the aromatic π-cloud (Å).
    cutoff_ion_pi : float
        Maximum ion–centroid distance for ion···π (Å).
    angle_ion_pi_threshold : float
        Maximum angle between ion–centroid vector and ring normal.
    confidence_threshold : float
        Minimum product of soft scores required to accept an interaction.

    Returns
    -------
    nx.Graph
        The input graph with new interaction edges added.
    """
    if radii is None:
        radii = BONDI

    cycles = [
        (node, data)
        for node, data in G.nodes(data=True)
        if data.get("node_type") == "aromatic_center"
    ]
    if not cycles:
        return G

    _find_pi_stacking(
        G, cycles,
        stacking_dist_threshold,
        offset_threshold,
        angle_stacking_threshold,
        confidence_threshold,
    )
    _find_h_pi(
        atoms, G, cycles, radii,
        angle_threshold, pi_cloud_radii,
        confidence_threshold,
    )
    _find_n_pi(
        atoms, G, cycles, radii,
        pi_cloud_radii, confidence_threshold,
    )
    _find_ion_pi(
        atoms, G, cycles,
        cutoff_ion_pi, angle_ion_pi_threshold,
        confidence_threshold,
    )

    return G


# --------------------------------------------------------------------------- #
#  π···π stacking
# --------------------------------------------------------------------------- #

def _find_pi_stacking(
    G: nx.Graph,
    cycles,
    stacking_dist_threshold: float,
    offset_threshold: float,
    angle_stacking_threshold: float,
    confidence_threshold: float,
) -> None:
    """Add π···π stacking edges between parallel aromatic rings."""
    for i in tqdm(range(len(cycles)), desc="Searching for π···π stacking"):
        node1, ring1 = cycles[i]
        c1 = ring1["position"]
        n1 = ring1["normal"]

        for j in range(i + 1, len(cycles)):
            node2, ring2 = cycles[j]
            c2 = ring2["position"]
            n2 = ring2["normal"]

            centroid_distance = float(np.linalg.norm(c2 - c1))
            score_dist = soft_less(centroid_distance, stacking_dist_threshold)
            if score_dist < 0.01:
                continue

            angle = float(np.degrees(np.arccos(np.clip(
                abs(np.dot(n1, n2)), 0.0, 1.0
            ))))
            score_angle = soft_less(angle, angle_stacking_threshold)

            if angle > angle_stacking_threshold:
                continue

            diff = c2 - c1
            offset = float(np.linalg.norm(diff - np.dot(diff, n1) * n1))
            score_offset = soft_less(offset, offset_threshold)

            total = score_dist * score_angle * score_offset
            if total < confidence_threshold:
                continue

            G.add_edge(
                node1, node2,
                atom_num_cycle_1=ring1["cycle"],
                atom_num_cycle_2=ring2["cycle"],
                bond_type="stacking",
                centroid_distance=centroid_distance,
                offset=offset,
                angle=angle,
                confidence_score=float(total),
            )


# --------------------------------------------------------------------------- #
#  X–H···π
# --------------------------------------------------------------------------- #

def _find_h_pi(
    atoms: Atoms,
    G: nx.Graph,
    cycles,
    radii: dict,
    angle_threshold: float,
    pi_cloud_radii: float,
    confidence_threshold: float,
) -> None:
    """Add X–H···π edges between hydrogens and aromatic centres."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    hydrogen_indices = [i for i, s in enumerate(symbols) if s == "H"]

    for h in tqdm(hydrogen_indices, desc="Searching for X-H···π interactions"):
        neighbours = [
            n for n in G.adj[h]
            if not isinstance(n, str)
            and G.edges[h, n]["bond_type"] == "covalent"
        ]
        if len(neighbours) != 1:
            continue

        x = neighbours[0]
        x_symbol = symbols[x]

        H = positions[h]
        X = positions[x]
        cutoff = pi_cloud_radii + radii["H"]

        for ring_node, ring in cycles:
            centre = ring["position"]
            normal = ring["normal"]

            h_centroid = float(np.linalg.norm(H - centre))
            score_dist = soft_less(h_centroid, cutoff)
            if score_dist < 0.01:
                continue

            # X–H···π angle
            v1 = X - H
            v2 = centre - H
            angle = float(np.degrees(np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                -1.0, 1.0
            ))))
            score_angle = soft_greater(angle, angle_threshold)

            # Projection of H onto the aromatic plane
            vec = H - centre
            projection = vec - np.dot(vec, normal) * normal
            projection_distance = float(np.linalg.norm(projection))
            score_proj = soft_less(projection_distance, ring["radius"])

            total = score_dist * score_angle * score_proj
            if total < confidence_threshold:
                continue

            G.add_edge(
                h, ring_node,
                bond_type="H-pi",
                x_num=x,
                h_num=h,
                symbol_x=x_symbol,
                h_centroid_distance=h_centroid,
                cycle=ring["cycle"],
                angle=angle,
                projection_distance=projection_distance,
                confidence_score=float(total),
            )


# --------------------------------------------------------------------------- #
#  n···π
# --------------------------------------------------------------------------- #

def _find_n_pi(
    atoms: Atoms,
    G: nx.Graph,
    cycles,
    radii: dict,
    pi_cloud_radii: float,
    confidence_threshold: float,
) -> None:
    """Add n···π edges between lone-pair donors and aromatic centres."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    for lp in tqdm(range(len(symbols)),
                   desc="Searching for n···π interactions"):
        lp_symbol = symbols[lp]

        neighbours = [
            n for n in G.adj[lp]
            if not isinstance(n, str)
            and G.edges[lp, n].get("bond_type") == "covalent"
        ]

        num_neighbours = G.nodes[lp].get("num_neighbours", len(neighbours))
        angle_range = acceptor_angle_range(lp_symbol, num_neighbours)
        if angle_range is None:
            continue

        donor = positions[lp]
        vdw_lp = vdw_radius(lp_symbol, radii)

        for ring_node, ring in cycles:
            centre = ring["position"]
            normal = ring["normal"]

            cutoff = vdw_lp + pi_cloud_radii
            distance = float(np.linalg.norm(donor - centre))
            score_dist = soft_less(distance, cutoff)
            if score_dist < 0.01:
                continue

            vec = centre - donor
            angle_to_normal = float(np.degrees(np.arccos(np.clip(
                abs(np.dot(vec, normal)) / np.linalg.norm(vec),
                0.0, 1.0
            ))))
            score_angle_normal = soft_less(angle_to_normal, 45.0)

            vec_proj = donor - centre
            projection = vec_proj - np.dot(vec_proj, normal) * normal
            projection_distance = float(np.linalg.norm(projection))
            score_proj = soft_less(projection_distance, ring["radius"])

            # Y–lp···π angular criterion (Y = covalent neighbour of lp)
            if angle_range == "skip":
                score_angle_y, angle_y = 1.0, None
            else:
                angle_min, angle_max = angle_range
                best_score_y = 0.0
                best_angle_y = None

                for n in neighbours:
                    vec_to_n = positions[n] - donor
                    angle = float(np.degrees(np.arccos(np.clip(
                        np.dot(vec, vec_to_n)
                        / (np.linalg.norm(vec) * np.linalg.norm(vec_to_n)),
                        -1.0, 1.0
                    ))))
                    score = soft_range(angle, angle_min, angle_max)
                    if score > best_score_y:
                        best_score_y = score
                        best_angle_y = angle

                if best_score_y == 0.0:
                    continue

                score_angle_y = best_score_y
                angle_y = best_angle_y

            total = (score_dist * score_angle_normal
                     * score_proj * score_angle_y)
            if total < confidence_threshold:
                continue

            G.add_edge(
                lp, ring_node,
                bond_type="n-pi",
                lp_num=lp,
                symbol_lp=lp_symbol,
                cycle=ring["cycle"],
                lp_centroid_distance=distance,
                angle=None if angle_y is None else float(angle_y),
                angle_to_normal=angle_to_normal,
                projection_distance=projection_distance,
                confidence_score=float(total),
            )


# --------------------------------------------------------------------------- #
#  ion···π
# --------------------------------------------------------------------------- #

def _find_ion_pi(
    atoms: Atoms,
    G: nx.Graph,
    cycles,
    cutoff_ion_pi: float,
    angle_ion_pi_threshold: float,
    confidence_threshold: float,
) -> None:
    """Add ion···π edges between bare ions and aromatic centres."""
    from ncitools.constants import HALOGENS, METALS

    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    indices = [
        i for i, s in enumerate(symbols)
        if (s in METALS or s in HALOGENS) and G.degree(i) == 0
    ]

    for i in indices:
        for ring_node, ring in cycles:
            centre = ring["position"]
            normal = ring["normal"]
            vec = positions[i] - centre

            dist = float(np.linalg.norm(vec))
            score_dist = soft_less(dist, cutoff_ion_pi)

            if dist < 1e-8:
                continue

            angle = float(np.degrees(np.arccos(np.clip(
                np.dot(vec, normal) / dist, -1.0, 1.0
            ))))
            score_angle = soft_less(angle, angle_ion_pi_threshold)

            total = score_dist * score_angle
            if total < confidence_threshold:
                continue

            G.add_edge(
                i, ring_node,
                bond_type="ion-pi",
                ion_num=i,
                symbol_ion=symbols[i],
                cycle=ring["cycle"],
                ion_centroid_distance=dist,
                angle_to_normal=angle,
                confidence_score=float(total),
            )


