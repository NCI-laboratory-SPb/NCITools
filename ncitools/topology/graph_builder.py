"""
Graph construction and NCI classification from BCPs.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Union

import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.constants import bohr_to_angstrom, BONDI
from ncitools.utils import get_symbol
from .topology import _angle, _bcp_properties, ANGSTROM_TO_BOHR


# ----------------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------------

def _covalent_neighbors(
    G: nx.Graph,
    atom_idx: int,
) -> List[int]:
    """
    Return covalently connected atom neighbours.

    Works with nx.Graph (single edge per pair).
    """
    result = []
    for neighbour in G.adj[atom_idx]:
        data = G.get_edge_data(atom_idx, neighbour)
        if data and data.get("bond_type") == "covalent":
            result.append(neighbour)
    return result


def _edge_has_type(
    G: nx.Graph,
    u: Union[int, str],
    v: Union[int, str],
    types: set,
) -> bool:
    """
    True if an edge u-v exists with bond_type in types.
    """
    if not G.has_edge(u, v):
        return False
    data = G.get_edge_data(u, v)
    return data and data.get("bond_type") in set(types)


# ----------------------------------------------------------------------------
# Aromatic/ring detection
# ----------------------------------------------------------------------------

def _fit_ring_plane(
    coordinates: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Fit a least-squares plane.

    Returns
    -------
    centre, normal, rms_deviation, max_deviation
    """
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
    rms = float(np.sqrt(np.mean(distances**2)))
    max_dev = float(np.max(np.abs(distances)))
    return centre, normal, rms, max_dev


def _heuristic_aromatic_cycle(
    cycle: List[int],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    planarity_tol: float = 0.10,
) -> bool:
    """
    Conservative heuristic for common aromatic rings.

    This is deliberately not claimed to be a universal aromaticity
    detector.

    The detector requires:
      - 5 or 6 membered ring
      - high planarity
      - ring atoms limited to C/N/O/S
      - every ring atom has a plausible sp2-like local environment
      - approximate 4n+2 pi-electron count

    Complex fused/polycyclic aromatic systems should preferably be
    supplied explicitly.
    """
    if len(cycle) not in {5, 6}:
        return False

    symbols = [get_symbol(atoms[i][0]) for i in cycle]
    allowed = {"C", "N", "O", "S"}
    if not all(s in allowed for s in symbols):
        return False

    coords = np.asarray([atoms[i][1] for i in cycle])
    plane = _fit_ring_plane(coords)
    if plane is None:
        return False

    _, _, rms, _ = plane
    if rms > planarity_tol * ANGSTROM_TO_BOHR:
        return False

    # Approximate pi-electron counting.
    pi_electrons = 0
    for atom_idx, symbol in zip(cycle, symbols):
        if symbol == "C":
            pi_electrons += 1
        elif symbol == "N":
            neighbours = _covalent_neighbors(G, atom_idx)
            has_h = any(get_symbol(atoms[n][0]) == "H" for n in neighbours)
            pi_electrons += 2 if has_h else 1
        elif symbol in {"O", "S"}:
            pi_electrons += 2

    return pi_electrons >= 2 and (pi_electrons - 2) % 4 == 0


def build_covalent_graph(
    bcps_covalent: List[Dict],
    atoms: List[Tuple[int, np.ndarray]],
    max_cycle_size: int = 8,
    planarity_tol: float = 0.10,
    aromatic_mode: str = "heuristic",
) -> nx.Graph:
    """
    Build molecular graph from covalent BCPs.

    Only aromatic rings (heuristic) are added as ring nodes.
    All other planar cycles are ignored for NCI classification.

    aromatic_mode:
        'heuristic'
        'none'
    """
    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    G = nx.Graph()

    # Atom nodes
    for i, (Z, position) in enumerate(atoms):
        G.add_node(
            i,
            node_type="atom",
            element=int(Z),
            symbol=get_symbol(Z),
            position=np.asarray(position).copy(),
        )

    # Covalent edges
    for bcp in bcps_covalent:
        i = int(bcp["atom1"])
        j = int(bcp["atom2"])
        if i == j:
            continue
        attrs = {
            "bond_type": "covalent",
            "position": np.asarray(bcp["position"]).copy(),
            "rho": float(bcp["rho"]),
            "laplacian": float(bcp["laplacian"]),
            "eigenvalues": np.asarray(bcp["eigenvalues"]).copy(),
            "ellipticity": float(bcp["ellipticity"]),
            "grad_norm": float(bcp["grad_norm"]),
        }
        G.add_edge(i, j, **attrs)

    # Cycles - only aromatic rings are kept as ring nodes
    simple_graph = nx.Graph()
    for u, v, data in G.edges(data=True):
        if data.get("bond_type") == "covalent":
            simple_graph.add_edge(u, v)

    cycles = nx.cycle_basis(simple_graph)
    ring_index = 0

    for cycle in cycles:
        size = len(cycle)
        if size < 5 or size > max_cycle_size:
            continue

        coords = coordinates[np.asarray(cycle, dtype=int)]
        plane = _fit_ring_plane(coords)
        if plane is None:
            continue

        centre, normal, rms_planarity, max_planarity = plane
        rms_angstrom = rms_planarity * bohr_to_angstrom
        if rms_angstrom > planarity_tol:
            continue

        radius = float(np.mean(np.linalg.norm(coords - centre, axis=1)))

        aromatic = False
        if aromatic_mode == "heuristic":
            aromatic = _heuristic_aromatic_cycle(
                cycle, G, atoms, planarity_tol=planarity_tol
            )
        elif aromatic_mode == "none":
            aromatic = False
        else:
            raise ValueError("aromatic_mode must be 'heuristic' or 'none'.")

        if aromatic:
            ring_node = f"ring_{ring_index}"
            ring_index += 1
            cycle_tuple = tuple(sorted(int(i) for i in cycle))
            plane_d = float(-np.dot(normal, centre))

            G.add_node(
                ring_node,
                node_type="aromatic_center",
                position=centre.copy(),
                normal=normal.copy(),
                plane_d=plane_d,
                radius=radius,
                cycle=cycle_tuple,
                size=size,
                aromatic=True,
                rms_planarity=rms_planarity,
                max_planarity_deviation=max_planarity,
            )

            for atom_idx in cycle:
                G.add_edge(ring_node, int(atom_idx), bond_type="aromatic", ring=cycle_tuple)

    return G


# ----------------------------------------------------------------------------
# Aromatic helpers
# ----------------------------------------------------------------------------

def _aromatic_rings_for_atom(
    G: nx.Graph,
    atom_idx: int,
) -> List[str]:
    """
    Return aromatic ring nodes associated with atom_idx.

    Simplified: any neighbour starting with "ring_" is considered
    an aromatic ring (only such nodes are created in build_covalent_graph).
    """
    result = []
    for neighbour in G.adj[atom_idx]:
        if isinstance(neighbour, str) and neighbour.startswith("ring_"):
            result.append(neighbour)
    return result


def _aromatic_membership(
    G: nx.Graph,
    atom_idx: int,
) -> Tuple[bool, List[str]]:
    """
    Determine whether atom belongs to an aromatic system.
    """
    rings = _aromatic_rings_for_atom(G, atom_idx)
    if rings:
        return True, rings

    # H attached to aromatic atom.
    if G.nodes[atom_idx].get("element") == 1:
        for neighbour in _covalent_neighbors(G, atom_idx):
            rings = _aromatic_rings_for_atom(G, neighbour)
            if rings:
                return True, rings
    return False, []


def _nearest_aromatic_ring(
    G: nx.Graph,
    atom_idx: int,
    position: np.ndarray,
) -> Optional[str]:
    """
    Find the nearest aromatic ring node for a given atom.
    """
    aromatic, rings = _aromatic_membership(G, atom_idx)
    if not aromatic:
        return None

    return min(
        rings,
        key=lambda ring: norm(np.asarray(position) - np.asarray(G.nodes[ring]["position"]))
    )


# ----------------------------------------------------------------------------
# BCP property helper (for reuse)
# ----------------------------------------------------------------------------

def _bcp_properties(bcp: Dict) -> Dict:
    """Extract common BCP properties into a dict."""
    return {
        "position": np.asarray(bcp["position"]).copy(),
        "rho": float(bcp["rho"]),
        "laplacian": float(bcp["laplacian"]),
        "ellipticity": float(bcp["ellipticity"]),
        "eigenvalues": np.asarray(bcp["eigenvalues"]).copy(),
        "grad_norm": float(bcp["grad_norm"]),
    }


# ----------------------------------------------------------------------------
# Aromatic interaction classifier
# ----------------------------------------------------------------------------

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
    """
    Classify aromatic pi interactions.

    Returns None if no robust aromatic interpretation is found.
    """
    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    p1 = coordinates[atom1_idx]
    p2 = coordinates[atom2_idx]

    aromatic1, rings1 = _aromatic_membership(G, atom1_idx)
    aromatic2, rings2 = _aromatic_membership(G, atom2_idx)

    if not aromatic1 and not aromatic2:
        return None

    # Both atoms associated with aromatic systems
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
        centroid_distance = norm(diff) * bohr_to_angstrom
        if centroid_distance > stacking_dist_threshold:
            return None

        ring_angle = np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0)))

        # Parallel stacking
        if ring_angle <= 30.0:
            offset = norm(diff - np.dot(diff, n1) * n1) * bohr_to_angstrom
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

    # Only one atom is aromatic
    if aromatic1:
        aromatic_idx, other_idx = atom1_idx, atom2_idx
        rings = rings1
    else:
        aromatic_idx, other_idx = atom2_idx, atom1_idx
        rings = rings2

    ring = min(
        rings,
        key=lambda r: norm(coordinates[aromatic_idx] - G.nodes[r]["position"])
    )
    centre = np.asarray(G.nodes[ring]["position"])
    normal = np.asarray(G.nodes[ring]["normal"])
    radius = float(G.nodes[ring]["radius"])

    other_symbol = symbols[other_idx]

    # X-H ... pi
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
        H_centroid_distance = norm(H - centre) * bohr_to_angstrom
        if H_centroid_distance > 3.5:
            return None

        angle = _angle(X, H, centre)
        if angle is None or angle < angle_threshold:
            return None

        projection = H - centre
        projection = projection - np.dot(projection, normal) * normal
        projection_distance = norm(projection) * bohr_to_angstrom
        radius_angstrom = radius * bohr_to_angstrom
        if projection_distance > radius_angstrom + 0.25:
            return None

        attrs = {
            "bond_type": "H-pi",
            "x_num": int(x_idx),
            "h_num": int(h_idx),
            "symbol_x": x_symbol,
            "ring": G.nodes[ring]["cycle"],
            "h_centroid_distance": float(H_centroid_distance),
            "projection_distance": float(projection_distance),
            "angle": float(angle),
        }
        attrs.update(_bcp_properties(bcp))
        return {"u": h_idx, "v": ring, "attrs": attrs}

    # Lone pair ... pi
    if other_symbol not in {"N", "O", "S", "Se"}:
        return None

    lp_idx = other_idx
    neighbours = _covalent_neighbors(G, lp_idx)
    if not neighbours:
        return None

    donor = coordinates[lp_idx]
    distance = norm(donor - centre) * bohr_to_angstrom
    if not (1.5 <= distance <= 3.6):
        return None

    vec = donor - centre
    vec_norm = norm(vec)
    if vec_norm <= 1.0e-12:
        return None

    angle_to_normal = np.degrees(np.arccos(np.clip(abs(np.dot(vec, normal)) / vec_norm, 0.0, 1.0)))
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
    print('centre', centre)
    print('donor', donor)
    print('neigh', coordinates[neighbour_idx])

    return {"u": lp_idx, "v": ring, "attrs": attrs}


# ----------------------------------------------------------------------------
# Ordinary NCI classifier
# ----------------------------------------------------------------------------

def _nearest_pair_from_bcp(bcp: Dict, atoms: List) -> Tuple[int, int]:
    """Use the atom pair already determined during BCP search."""
    return int(bcp["atom1"]), int(bcp["atom2"])


def build_graph_with_nci(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    angle_threshold: float = 120.0,
    stacking_dist_threshold: float = 5.5,
    offset_threshold: float = 2.0,
) -> nx.Graph:
    """
    Add classified NCI interactions.

    The atom pair comes directly from the BCP search.
    """
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    for bcp in tqdm(bcps, desc="Classifying NCI BCPs"):
        atom1_idx, atom2_idx = _nearest_pair_from_bcp(bcp, atoms)
        if atom1_idx == atom2_idx:
            continue

        symbol1 = symbols[atom1_idx]
        symbol2 = symbols[atom2_idx]

        # Aromatic interactions first
        aromatic_result = _classify_aromatic_nci(
            atom1_idx, atom2_idx, bcp, atoms, G,
            angle_threshold=angle_threshold,
            stacking_dist_threshold=stacking_dist_threshold,
            offset_threshold=offset_threshold,
        )
        if aromatic_result is not None:
            G.add_edge(aromatic_result["u"], aromatic_result["v"], **aromatic_result["attrs"])
            continue

        # Hydrogen bonds
        is_h1 = symbol1 == "H"
        is_h2 = symbol2 == "H"
        if is_h1 and is_h2:
            continue

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

            D_H = norm(H - D) * bohr_to_angstrom
            H_A = norm(H - A) * bohr_to_angstrom
            D_A = norm(D - A) * bohr_to_angstrom
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

        # Heavy atom sigma-hole interactions
        distance = norm(coordinates[atom1_idx] - coordinates[atom2_idx]) * bohr_to_angstrom
        r1 = BONDI.get(symbol1, None)
        r2 = BONDI.get(symbol2, None)
        if r1 is None or r2 is None:
            continue

        d1 = norm(coordinates[atom1_idx] - np.asarray(bcp["position"])) * bohr_to_angstrom
        d2 = norm(coordinates[atom2_idx] - np.asarray(bcp["position"])) * bohr_to_angstrom
        norm1 = d1 / r1
        norm2 = d2 / r2

        if norm1 <= norm2:
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
        N = coordinates[nucleophile_idx]
        candidate_angles = []
        for neighbour_idx in electrophile_neighbours:
            angle = _angle(coordinates[neighbour_idx], E, N)
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

    return G


# ----------------------------------------------------------------------------
# Carbonyl n -> pi*
# ----------------------------------------------------------------------------

def _identify_carbonyl_carbons(
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    carbonyl_max_distance: float = 1.35,
) -> Dict[int, int]:
    """
    Identify carbonyl carbons from the covalent graph.

    A C atom is considered carbonyl-like if:
      - it has at least one covalent O neighbour;
      - the shortest C-O covalent distance <= threshold.

    This is more robust than requiring exactly three neighbours.
    """
    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    carbonyls = {}
    for idx, symbol in enumerate(symbols):
        if symbol != "C":
            continue
        neighbours = _covalent_neighbors(G, idx)
        oxygen_neighbours = []
        for neighbour in neighbours:
            if symbols[neighbour] != "O":
                continue
            dist = norm(coordinates[idx] - coordinates[neighbour]) * bohr_to_angstrom
            if dist <= carbonyl_max_distance:
                oxygen_neighbours.append(neighbour)

        if oxygen_neighbours:
            carbonyl_oxygen = min(oxygen_neighbours, key=lambda j: norm(coordinates[idx] - coordinates[j]))
            carbonyls[idx] = carbonyl_oxygen

    return carbonyls


def build_graph_with_carbonyl(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    distance_cutoff: float = 3.22,
    angle_min: float = 90.0,
    angle_max: float = 130.0,
) -> nx.Graph:
    """
    Detect n -> pi* interactions.

    Geometry:
        donor Y ... C=O

        Y...C <= 3.22 Å
        angle Y...C-O = 95-125 degrees

    These correspond to the commonly used Bürgi-Dunitz-like
    geometry of carbonyl n -> pi* interactions.
    """
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]
    carbonyls = _identify_carbonyl_carbons(G, atoms)

    for bcp in tqdm(bcps, desc="Searching n->pi* interactions"):
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

        distance = norm(coordinates[donor_oxygen] - coordinates[acceptor_carbon]) * bohr_to_angstrom
        if distance > distance_cutoff:
            continue

        angle = _angle(coordinates[donor_oxygen], coordinates[acceptor_carbon], coordinates[carbonyl_oxygen])
        if angle is None or not (angle_min <= angle <= angle_max):
            continue

        # Carbonyl plane and pyramidalization-like descriptor
        carbon_neighbours = _covalent_neighbors(G, acceptor_carbon)
        substituents = [n for n in carbon_neighbours if n != carbonyl_oxygen]
        plane_angle = None
        if len(substituents) >= 2:
            coords = np.asarray([coordinates[n] for n in substituents])
            plane = _fit_ring_plane(coords)
            if plane is not None:
                _, normal, _, _ = plane
                vector = coordinates[donor_oxygen] - coordinates[acceptor_carbon]
                vnorm = norm(vector)
                if vnorm > 1.0e-12:
                    plane_angle = np.degrees(np.arcsin(np.clip(abs(np.dot(vector, normal)) / vnorm, 0.0, 1.0)))

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

    return G


# ----------------------------------------------------------------------------
# Unclassified NCI contacts
# ----------------------------------------------------------------------------

def _collect_classified_bcp_positions(G: nx.Graph) -> List[np.ndarray]:
    """
    Return positions of all BCPs already represented in graph edges.
    """
    positions = []
    for _, _, data in G.edges(data=True):
        bond_type = data.get("bond_type")
        if bond_type in {None, "covalent", "aromatic"}:
            continue
        position = data.get("position")
        if position is not None:
            positions.append(np.asarray(position))
    return positions


def build_graph_with_unclassified(
    bcps: List[Dict],
    G: nx.Graph,
    atoms: List[Tuple[int, np.ndarray]],
    position_tolerance: float = 0.10,
) -> nx.Graph:
    """
    Add remaining NCI BCPs as unclassified contacts.

    position_tolerance is in Angstrom.
    """
    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    symbols = [get_symbol(atom[0]) for atom in atoms]

    classified_positions = _collect_classified_bcp_positions(G)
    classified_tree = cKDTree(np.asarray(classified_positions)) if classified_positions else None
    tolerance_bohr = position_tolerance * ANGSTROM_TO_BOHR

    for bcp in tqdm(bcps, desc="Collecting unclassified NCI"):
        position = np.asarray(bcp["position"])
        if classified_tree is not None:
            dist, _ = classified_tree.query(position, k=1)
            if dist <= tolerance_bohr:
                continue

        i = int(bcp["atom1"])
        j = int(bcp["atom2"])
        if i == j:
            continue

        # Do not duplicate a pair already carrying an NCI edge.
        if _edge_has_type(
            G, i, j,
            {"HB", "XB", "ChB", "PnB", "stacking", "T-stacking", "H-pi", "n-pi", "carbonyl"}
        ):
            continue

        # Avoid same-ring internal contacts.
        aromatic1, rings1 = _aromatic_membership(G, i)
        aromatic2, rings2 = _aromatic_membership(G, j)
        if aromatic1 and aromatic2 and set(rings1) & set(rings2):
            continue

        # Skip if the two atoms belong to two different aromatic rings
        # that are already connected by a stacking or T-stacking interaction.
        if aromatic1 and aromatic2:
            skip = False
            for ri in rings1:
                for rj in rings2:
                    if _edge_has_type(G, ri, rj, {"stacking", "T-stacking"}):
                        skip = True
                        break
                if skip:
                    break
            if skip:
                continue

        distance = norm(coordinates[i] - coordinates[j]) * bohr_to_angstrom
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
        classified_tree = cKDTree(np.asarray(classified_positions))

    return G