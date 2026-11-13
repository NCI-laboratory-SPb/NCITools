"""Shared helpers for topology detectors."""

from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import norm

from ncitools.constants import ANGSTROM_TO_BOHR


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _covalent_neighbors(
    G: nx.Graph,
    atom_idx: int,
    bond_types: Tuple[str, ...] = ("covalent", "coordination"),
) -> List[int]:
    """
    Neighbours of ``atom_idx`` connected by one of ``bond_types``.

    By default both covalent and coordination edges count, mirroring the
    ``("covalent", "coordination")`` partner rule used throughout the
    geometry detectors.  Pass ``bond_types=("covalent",)`` where the
    geometry module restricts itself to covalent bonds (e.g. the parent of a
    hydrogen in X–H···π).
    """
    types = set(bond_types)
    result = []
    for neighbour in G.adj[atom_idx]:
        data = G.get_edge_data(atom_idx, neighbour)
        if data and data.get("bond_type") in types:
            result.append(neighbour)
    return result


def _edge_has_type(G: nx.Graph, u, v, types) -> bool:
    """True if edge u–v exists with bond_type in types."""
    if not G.has_edge(u, v):
        return False
    data = G.get_edge_data(u, v)
    return data is not None and data.get("bond_type") in set(types)


# ---------------------------------------------------------------------------
# Aromatic helpers
# ---------------------------------------------------------------------------

def _is_aromatic_node(node, data: dict) -> bool:
    """Robustly identify aromatic-centre nodes regardless of exact convention."""
    if data.get("node_type") == "aromatic_center":
        return True
    if data.get("aromatic"):
        return True
    return isinstance(node, str) and node.startswith("ring_")


def _aromatic_rings_for_atom(G: nx.Graph, atom_idx: int) -> List[str]:
    """Aromatic ring nodes attached to atom_idx."""
    result = []
    for neighbour in G.adj[atom_idx]:
        if isinstance(neighbour, str):
            ndata = G.nodes.get(neighbour, {})
            if _is_aromatic_node(neighbour, ndata):
                result.append(neighbour)
    return result


def _aromatic_membership(
    G: nx.Graph,
    atom_idx: int,
) -> Tuple[bool, List[str]]:
    """True if atom belongs to (or is H attached to) an aromatic system."""
    rings = _aromatic_rings_for_atom(G, atom_idx)
    if rings:
        return True, rings

    if G.nodes[atom_idx].get("element") == 1:
        for neighbour in _covalent_neighbors(G, atom_idx, ("covalent",)):
            rings = _aromatic_rings_for_atom(G, neighbour)
            if rings:
                return True, rings
    return False, []


def _nearest_aromatic_ring(
    G: nx.Graph,
    atom_idx: int,
    position: np.ndarray,
) -> Optional[str]:
    """Nearest aromatic ring node to a given atom position."""
    aromatic, rings = _aromatic_membership(G, atom_idx)
    if not aromatic:
        return None
    return min(
        rings,
        key=lambda ring: norm(
            np.asarray(position) - np.asarray(G.nodes[ring]["position"])
        ),
    )


# ---------------------------------------------------------------------------
# BCP / plane helpers
# ---------------------------------------------------------------------------

def _nearest_pair_from_bcp(bcp: dict, atoms) -> Tuple[int, int]:
    """Use the atom pair recorded during BCP search."""
    return int(bcp["atom1"]), int(bcp["atom2"])


def _fit_ring_plane(
    coordinates: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Least-squares plane: (centre, normal, rms_dev, max_dev)."""
    coords = np.asarray(coordinates, dtype=float)
    centre = coords.mean(axis=0)
    centered = coords - centre

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vh[-1]
    n = norm(normal)
    if n <= 1.0e-12:
        return None
    normal /= n

    distances = centered @ normal
    rms = float(np.sqrt(np.mean(distances ** 2)))
    max_dev = float(np.max(np.abs(distances)))
    return centre, normal, rms, max_dev


def _collect_classified_bcp_positions(G: nx.Graph) -> List[np.ndarray]:
    """Positions of all non-covalent, non-aromatic edges (already classified NCI)."""
    positions = []
    for _, _, data in G.edges(data=True):
        bond_type = data.get("bond_type")
        if bond_type in {None, "covalent", "aromatic"}:
            continue
        position = data.get("position")
        if position is not None:
            positions.append(np.asarray(position))
    return positions


# ---------------------------------------------------------------------------
# Bridge with geometry.graph_builder
# ---------------------------------------------------------------------------

def _convert_graph_to_bohr(G: nx.Graph) -> None:
    """
    Convert node position/radius attributes from Angstrom (units used by
    the ``geometry`` module) to Bohr (internal units of ``topology``).
    """
    for _, data in G.nodes(data=True):
        if "position" in data and data["position"] is not None:
            data["position"] = np.asarray(data["position"]) * ANGSTROM_TO_BOHR
        if "radius" in data and data["radius"] is not None:
            data["radius"] = float(data["radius"]) * ANGSTROM_TO_BOHR
        if "plane_d" in data and data["plane_d"] is not None:
            data["plane_d"] = float(data["plane_d"]) * ANGSTROM_TO_BOHR


def _ensure_atom_elements(G: nx.Graph, atoms_bohr) -> None:
    """Add ``element`` attribute to atom nodes if geometry didn't set it."""
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "atom" or "element" in data:
            continue
        if isinstance(node, int) and 0 <= node < len(atoms_bohr):
            data["element"] = int(atoms_bohr[node][0])


def _prepare_aromatic_rings(G: nx.Graph, atoms_bohr) -> None:
    """
    Enrich aromatic-centre nodes with the metadata needed by the aromatic
    detector (``cycle``, ``normal``, ``plane_d``, ``radius``).

    Assumes node positions are already in Bohr (call ``_convert_graph_to_bohr``
    first).
    """
    coords = np.asarray([a[1] for a in atoms_bohr], dtype=float)

    for node, data in G.nodes(data=True):
        if not _is_aromatic_node(node, data):
            continue

        cycle = data.get("cycle")
        if cycle is None:
            cycle = tuple(
                sorted(
                    int(n) for n in G.neighbors(node)
                    if not isinstance(n, str)
                )
            )
            data["cycle"] = cycle

        ring_coords = coords[list(cycle)]
        centre = data.get("position")
        if centre is None:
            centre = ring_coords.mean(axis=0)
        else:
            centre = np.asarray(centre)
        data["position"] = centre

        if "normal" not in data or "plane_d" not in data:
            centered = ring_coords - centre
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            normal = vh[-1]
            nrm = norm(normal)
            if nrm > 1.0e-12:
                normal = normal / nrm
            data["normal"] = normal
            data["plane_d"] = float(-np.dot(normal, centre))

        if "radius" not in data:
            data["radius"] = float(
                np.mean(norm(ring_coords - centre, axis=1))
            )

        data.setdefault("aromatic", True)
        data.setdefault("size", len(cycle))


def _prepare_graph_for_nci(G: nx.Graph, atoms_bohr) -> None:
    """Full preparation pipeline for a geometry-built covalent graph."""
    _convert_graph_to_bohr(G)
    _ensure_atom_elements(G, atoms_bohr)
    _prepare_aromatic_rings(G, atoms_bohr)