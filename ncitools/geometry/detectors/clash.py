from __future__ import annotations

from scipy.spatial import cKDTree
import numpy as np
import networkx as nx
from ase import Atoms

def _get_node_coords(
        node: int | str,
        positions: np.ndarray,
        G: nx.Graph
) -> np.ndarray | None:
    """
    Extract 3D coordinates for a given graph node.
    Works for both atomic indices (int) and aromatic ring centroids (str).
    """
    if isinstance(node, int):
        return positions[node]
    if isinstance(node, str) and "position" in G.nodes[node]:
        return G.nodes[node]["position"]
    return None


def _segment_distance(
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray,
        num_samples: int = 5
) -> float:
    """
    Approximate minimum distance between two 3D line segments (p1-p2 and q1-q2)
    using uniform point sampling.
    """
    t = np.linspace(0, 1, num_samples)

    # Generate points along both segments
    points_p = np.outer(1 - t, p1) + np.outer(t, p2)
    points_q = np.outer(1 - t, q1) + np.outer(t, q2)

    # Calculate pairwise distances between all sampled points
    diff = points_p[:, None, :] - points_q[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return float(np.min(dists))


def check_spatial_clashes(
        atoms: Atoms,
        G: nx.Graph,
        clash_threshold: float = 1.0
) -> list[str]:
    """
    Identify independent edges (covalent, coordination, or non-covalent)
    that cross or pass too close to each other in 3D space.
    Incident edges are excluded.
    """
    warnings = []
    edges = []
    midpoints = []
    half_lengths = []

    positions = atoms.positions

    # Aromatic edges filter
    for u, v, data in G.edges(data=True):
        if data.get("bond_type") == "aromatic":
            continue

        p1 = _get_node_coords(u, positions, G)
        p2 = _get_node_coords(v, positions, G)
        if p1 is None or p2 is None:
            continue

        # Calculate the centers of all valid edges
        edges.append((u, v, data, p1, p2))
        midpoints.append((p1 + p2) * 0.5)
        half_lengths.append(np.linalg.norm(p1 - p2) * 0.5)

    if not edges:
        return []

    midpoints = np.array(midpoints)
    half_lengths = np.array(half_lengths)

    # Build a K-D tree based on the midpoints
    tree = cKDTree(midpoints)
    max_search_radius = 2.0 * np.max(half_lengths) + clash_threshold

    # Find candidate pairs within maximum interaction distance
    potential_pairs = tree.query_pairs(r=max_search_radius)

    # Topological neighborhood filter
    cov_neighborhood = {}
    for node in G.nodes():
        neighbors = {node}
        for nbr, edge_data in G[node].items():
            if edge_data.get("bond_type") in ("covalent", "coordination", "aromatic"):
                neighbors.add(nbr)
        cov_neighborhood[node] = set(neighbors)

    for i, j in potential_pairs:
        u1, v1, data1, p1, p2 = edges[i]
        u2, v2, data2, q1, q2 = edges[j]

        # Skip all incident edges
        neighborhood1 = cov_neighborhood.get(u1, {u1}) | cov_neighborhood.get(v1, {v1})
        if not neighborhood1.isdisjoint({u2, v2}):
            continue

        dist = _segment_distance(p1, p2, q1, q2)
        if dist < clash_threshold:
            type1 = data1.get("bond_type", "unknown")
            type2 = data2.get("bond_type", "unknown")

            # Format nodes if they are atoms
            label_u1 = u1 + 1 if isinstance(u1, int) else u1
            label_u2 = u2 + 1 if isinstance(u2, int) else u2
            label_v1 = v1 + 1 if isinstance(v1, int) else v1
            label_v2 = v2 + 1 if isinstance(v2, int) else v2

            message = (
                f"ATTENTION: interactions {type1} ({label_u1}-{label_v1}) and"
                f" {type2} ({label_u2}-{label_v2}) are located close together"
                f" in space dist: {dist:.2f} Å"
            )
            warnings.append(message)
    return warnings

