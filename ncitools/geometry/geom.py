""" ДОБАВЬ ТЕТРЕЛЬНЫЕ СВЯЗИ (C-bonds в литературе) ???"""

"""
Geometry‑based detection of non‑covalent interactions  using covalent radii and van der Waals radii.
"""

import os
import ase
import random
import textwrap
from ase import Atoms
import networkx as nx
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime
from ase.neighborlist import neighbor_list
from ncitools.constants import RADII, BONDI
from ncitools.correlations import Rozenberg_2000
from ncitools.utils import read_input
from ncitools.quotes import quotes

# --------------------------------------------------------------------------- #
#  Covalent bond identification & graph construction
# --------------------------------------------------------------------------- #
def build_graph_with_covalent_pairs(
        atoms: Atoms,
        radii: dict = RADII,
        tolerance: float = 0.1,
        max_cycle_size: int = 8,
        planarity_tol: float = 0.10
) -> nx.Graph:
    """
    Build a molecular graph from atomic coordinates.

    Covalent bonds are determined from covalent radii.
    Planar 5-8 membered cycles are represented by additional
    aromatic centre nodes.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    radii : dict
        Covalent radii.
    tolerance : float
        Bond tolerance (Å).
    max_cycle_size : int
        Maximum ring size considered aromatic.
    planarity_tol : float
        RMS deviation from best-fit plane (Å).

    Returns
    -------
    nx.Graph
        Molecular graph.
    """

    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    cutoffs = [radii[s] + tolerance for s in symbols]

    i_list, j_list = neighbor_list("ij", atoms, cutoffs)

    G = nx.Graph()

    # Atom nodes
    for i, atom in enumerate(atoms):
        G.add_node(
            i,
            node_type="atom",
            element=atom.symbol,
            position=atom.position.copy()
        )

    # Covalent bonds
    for i, j in zip(i_list, j_list):
        if i >= j:
            continue

        cutoff = (radii[atoms[i].symbol] + radii[atoms[j].symbol] + tolerance)
        distance = atoms.get_distance(i, j, mic=True)

        if distance <= cutoff:
            G.add_edge(i, j, bond_type="covalent")

    # Aromatic cycle detection
    aromatic_index = 0

    for cycle in nx.cycle_basis(G):
        cycle_tuple = tuple(sorted(cycle))
        ring_size = len(cycle)

        if ring_size < 5 or ring_size > max_cycle_size:
            continue

        coords = atoms.positions[cycle]
        centre = coords.mean(axis=0)

        centered = coords - centre
        _, _, vh = np.linalg.svd(centered)

        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        distances = centered @ normal

        rms = np.sqrt(np.mean(distances ** 2))
        max_dev = np.max(np.abs(distances))
        planar = rms <= planarity_tol

        if not planar:
            continue

        radius = np.mean(np.linalg.norm(coords - centre, axis=1))
        plane_d = -np.dot(normal, centre)

        ring_node = f"ring_{aromatic_index}"
        aromatic_index += 1

        G.add_node(
            ring_node,
            node_type="aromatic_center",
            position=centre,
            normal=normal,
            plane_d=float(plane_d),
            radius=float(radius),
            cycle=cycle_tuple,
            size=ring_size,
            aromatic=True,
            rms_planarity=float(rms),
            max_planarity_deviation=float(max_dev)
        )

        # Connect aromatic centre to every atom in the ring
        for atom_idx in cycle:
            G.add_edge(
                ring_node,
                atom_idx,
                bond_type="aromatic",
                ring=cycle_tuple
            )

    return G



# --------------------------------------------------------------------------- #
#  Generic HB, XB, ChB and PnB interactions search
# --------------------------------------------------------------------------- #
def find_nci_bonds(atoms: ase.Atoms,
                   G: nx.Graph,
                   central_elements: list,
                   angle_threshold: float,
                   bond_type: str,
                   desc: str = "",
                   radii: dict = BONDI) -> nx.Graph:
    """
    Add non‑covalent interaction edges D–X···A to the molecular graph.

    For every atom X belonging to `central_elements` the function examines
    its covalently bonded neighbours D.  For each D, all heavy atoms A
    (neither H nor C) are tested.  A contact is added as a new edge in `G`
    if the X···A distance is ≤ sum of van der Waals radii and the
    D–X···A angle is ≥ `angle_threshold`.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    G : nx.Graph
        Graph already containing covalent edges.  It is modified in place.
    central_elements : list[str]
        Chemical symbols acting as the bridge atom X (e.g. ['H'] for HB).
    angle_threshold : float
        Minimum D–X···A angle in degrees.
    bond_type : str
        Label stored in the edge attribute 'bond_type' ('HB', 'XB', …).
    desc : str
        Description shown in the tqdm progress bar.
    radii : dict
        Van der Waals radii (default: ncitools.constants.BONDI).

    Returns
    -------
    nx.Graph
        The input graph `G` with new NCI edges added.
    """
    symbols = atoms.get_chemical_symbols()
    # Indices of atoms that can be X
    x_indices = [i for i, sym in enumerate(symbols)
                 if sym in central_elements]
    # Possible acceptors: all heavy atoms except carbon
    acceptor_indices = [i for i, sym in enumerate(symbols)
                        if sym != 'H' and sym != 'C']

    if len(acceptor_indices) == 0:
        return G

    for x in tqdm(x_indices, desc=desc):
        x_el = symbols[x]
        for d in list(G.adj[x]):                     # d is covalently bonded to X
            d_el = symbols[d]

            # Distances from X to every potential acceptor
            dists = atoms.get_distances(x, acceptor_indices, mic=True)
            zipped = sorted(zip(dists, acceptor_indices), key=lambda p: p[0])
            if not zipped:
                continue
            sorted_dists, sorted_inds = zip(*zipped)

            # The nearest heavy atom can be the donor D itself – skip it
            for dist, a in zip(sorted_dists[1:], sorted_inds[1:]):
                a_el = symbols[a]
                if dist <= radii[x_el] + radii[a_el]:
                    angle = atoms.get_angle(d, x, a, mic=True)
                    if angle >= angle_threshold:
                        da_dist = atoms.get_distance(a, d, mic=True)
                        cov_dist = atoms.get_distance(x, d, mic=True)
                        attr = {
                            'bond_type': bond_type,
                            'd_num': d, 'x_num': x, 'a_num': a,
                            'symbol_d': d_el, 'symbol_x': x_el,
                            'symbol_a': a_el,
                            'length_cov': cov_dist,
                            'length_xb': dist,
                            'length_da': da_dist,
                            'angle': angle
                        }
                        if bond_type == 'HB':
                            attr['length_hb'] = dist
                            attr['h_num'] = x
                        G.add_edge(x, a, **attr)
                else:
                    # distances only grow further down the sorted list
                    break

    return G



# --------------------------------------------------------------------------- #
#  Generic π···π, X-H···π and n···π interactions search
# --------------------------------------------------------------------------- #
def find_nci_with_aromatic(
        atoms: ase.Atoms,
        G: nx.Graph,
        angle_threshold: float = 120,
        stacking_dist_threshold: float = 5.5,
        offset_threshold: float = 2.0,
) -> nx.Graph:
    """
    Add π···π, X-H···π and n···π interactions.

    Parameters
    ----------
    atoms : ase.Atoms

    G : nx.Graph
        Molecular graph.
    angle_threshold : float
        Minimum X-H-π angle.
    stacking_dist_threshold: float
        Maximum distance between centroids for parallel stacking
    offset_threshold: float
        Maximum value for offset in parallel stacking

    Returns
    -------
    nx.Graph
        Updated graph.
    """

    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    cycles = [
        (node, data)
        for node, data in G.nodes(data=True)
        if data["node_type"] == "aromatic_center"
    ]

    # π···π stacking
    for i in tqdm(range(len(cycles)), desc="Searching for π···π stacking"):

        node1, ring1 = cycles[i]

        c1 = ring1["position"]
        n1 = ring1["normal"]

        for j in range(i + 1, len(cycles)):

            node2, ring2 = cycles[j]

            c2 = ring2["position"]
            n2 = ring2["normal"]

            centroid_distance = np.linalg.norm(c2 - c1)

            if centroid_distance > stacking_dist_threshold:
                continue

            angle = np.degrees(
                np.arccos(
                    np.clip(
                        abs(np.dot(n1, n2)),
                        0.0,
                        1.0
                    )
                )
            )

            # parallel stacking
            if angle <= 30.0:
                diff = c2 - c1
                offset = np.linalg.norm(diff - np.dot(diff, n1) * n1)

                if offset <= offset_threshold:
                    G.add_edge(
                        node1,
                        node2,
                        atom_num_cycle_1=np.min(ring1['cycle']),
                        atom_num_cycle_2=np.min(ring2['cycle']),
                        bond_type="stacking",
                        centroid_distance=float(centroid_distance),
                        offset=float(offset),
                        angle=float(angle)
                    )

            # T-stacking
            elif angle >= 80.0:
                G.add_edge(
                    node1,
                    node2,
                    atom_num_cycle_1 = np.min(ring1['cycle']),
                    atom_num_cycle_2 = np.min(ring2['cycle']),
                    bond_type="T-stacking",
                    centroid_distance=float(centroid_distance),
                    angle=float(angle)
                )

    # X-H···π interactions
    hydrogen_indices = [i for i, s in enumerate(symbols) if s == "H"]

    for h in tqdm(hydrogen_indices, desc="Searching for X-H···π interactions"):

        neighbours = [
            n for n in G.adj[h]
            if G.edges[h, n]["bond_type"] == "covalent"
        ]

        if len(neighbours) != 1:
            continue

        x = neighbours[0]
        x_symbol = symbols[x]

        if x_symbol not in ("C", "N", "O", "S"):
            continue

        H = positions[h]
        X = positions[x]

        cutoff = 3.5 if x_symbol in ("C", "S") else 3.0

        for ring_node, ring in cycles:

            centre = ring["position"]
            normal = ring["normal"]

            h_centroid = np.linalg.norm(H - centre)

            if h_centroid > cutoff:
                continue

            # X-H-π angle
            v1 = X - H
            v2 = centre - H

            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(v1, v2) /
                        (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1.0,
                        1.0
                    )
                )
            )

            if angle < angle_threshold:
                continue

            # projection of H onto aromatic plane
            vec = H - centre
            projection = vec - np.dot(vec, normal) * normal
            projection_distance = np.linalg.norm(projection)

            if projection_distance > ring["radius"] + 0.1:
                continue

            G.add_edge(
                h,
                ring_node,
                bond_type="H-pi",
                x_num=x,
                h_num=h,
                symbol_x=x_symbol,
                h_centroid_distance=float(h_centroid),
                angle=float(angle),
                projection_distance=float(projection_distance)
            )

    # n···π interactions
    lp_atoms = {"O", "N", "S", "Se"}

    for lp in tqdm(range(len(symbols)), desc="Searching for n···π interactions"):
        lp_symbol = symbols[lp]

        if lp_symbol not in lp_atoms:
            continue

        # Simple lone-pair filter:
        # at least one covalent neighbour and no hypervalent atoms
        neighbours = [
            n for n in G.adj[lp]
        ]

        if len(neighbours) == 0:
            continue

        if lp_symbol == "O" and len(neighbours) > 2:
            continue

        if lp_symbol == "N" and len(neighbours) > 3:
            continue

        if (lp_symbol == "S" or lp_symbol == "Se")  and len(neighbours) > 4:
            continue

        donor = positions[lp]

        for ring_node, ring in cycles:
            centre = ring["position"]
            normal = ring["normal"]

            # Distance criterion
            distance = np.linalg.norm(donor - centre)

            if lp_symbol == "S" or lp_symbol == "Se":
                if distance < 2.5 or distance > 4.0:
                    continue
            else:
                if distance < 2.5 or distance > 3.6:
                    continue

            # Angle to ring plane
            vec = donor - centre

            angle_to_normal = np.degrees(
                np.arccos(
                    np.clip(
                        abs(np.dot(vec, normal)) / np.linalg.norm(vec),
                        0.0,
                        1.0
                    )
                )
            )

            for n in neighbours:
                vec_to_n = positions[n] - donor
                angle = 180 - np.degrees(
                   np.arccos(
                        np.clip(
                            np.dot(vec, -vec_to_n) / (np.linalg.norm(vec) * np.linalg.norm(vec_to_n)),
                            0.0,
                            1.0
                        )
                    )
                )

                if (100 <= angle <= 130) & (0.0 <= angle_to_normal <= 45.0):
                    G.add_edge(
                        lp,
                        ring_node,
                        bond_type="n-pi",
                        lp_num=lp,
                        symbol_lp=lp_symbol,
                        lp_centroid_distance=float(distance),
                        angle=float(angle)
                    )

    return G




# --------------------------------------------------------------------------- #
#  Generic n···π with carbonyls
# --------------------------------------------------------------------------- #
def find_carbonyl_interactions(
        atoms: ase.Atoms,
        G: nx.Graph,
        distance_cutoff: float = 3.2,
        angle_min: float = 95.0,
        angle_max: float = 125.0,
        plane_angle_min: float = 70.0,
        plane_angle_max: float = 110.0,
) -> nx.Graph:
    """
    Detect n→π* interactions between lone-pair donors (X=O) and
    carbonyl acceptors using purely geometric criteria.

    Donors
    ------
    Any oxygen atom having exactly one covalent neighbour.

    Acceptors
    ---------
    Carbonyl groups O=C identified as oxygen atoms with exactly one
    covalent neighbour which is carbon.

    Criteria
    --------
    * O···C distance <= distance_cutoff
    * Bürgi-Dunitz angle within [angle_min, angle_max]
    * Angle between donor and acceptor planes within
      [plane_angle_min, plane_angle_max]

    Returns
    -------
    nx.Graph
    """

    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    # ============================================================
    # Helper: plane normal from SVD
    # ============================================================

    def plane_normal(center, oxygen):

        neighbours = [
            n for n in G.adj[center]
            if G.edges[center, n]["bond_type"] == "covalent"
        ]

        coords = [positions[center], positions[oxygen]]

        for n in neighbours:
            if n != oxygen:
                coords.append(positions[n])

        coords = np.asarray(coords)

        if len(coords) < 3:
            return None

        centroid = coords.mean(axis=0)
        centered = coords - centroid

        _, _, vh = np.linalg.svd(centered)

        normal = vh[-1]

        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            return None

        return normal / norm

    # ============================================================
    # Collect donor groups
    # ============================================================

    donors = []

    for o in range(len(symbols)):

        if symbols[o] != "O":
            continue

        neighbours = [
            n for n in G.adj[o]
            if G.edges[o, n]["bond_type"] == "covalent"
        ]

        if len(neighbours) != 1:
            continue

        center = neighbours[0]

        donors.append({
            "O": o,
            "center": center,
            "normal": plane_normal(center, o)
        })

    # ============================================================
    # Collect carbonyl acceptors
    # ============================================================

    acceptors = []

    for o in range(len(symbols)):

        if symbols[o] != "O":
            continue

        neighbours = [
            n for n in G.adj[o]
            if G.edges[o, n]["bond_type"] == "covalent"
        ]

        if len(neighbours) != 1:
            continue

        carbon = neighbours[0]

        if symbols[carbon] != "C":
            continue

        acceptors.append({
            "O": o,
            "C": carbon,
            "normal": plane_normal(carbon, o)
        })

    # ============================================================
    # Search interactions
    # ============================================================

    for donor in tqdm(donors, desc="Searching for n→π* interactions with carbonyls"):

        O_d = donor["O"]
        donor_center = donor["center"]
        n_d = donor["normal"]

        for acceptor in acceptors:

            O_a = acceptor["O"]
            C_a = acceptor["C"]
            n_a = acceptor["normal"]

            # Same group
            if O_d == O_a:
                continue

            # Distance
            distance = atoms.get_distance(
                O_d,
                C_a,
                mic=True
            )

            if distance > distance_cutoff:
                continue

            # Bürgi-Dunitz angle
            bd_angle = atoms.get_angle(
                O_d,
                C_a,
                O_a,
                mic=True
            )

            if not (angle_min <= bd_angle <= angle_max):
                continue

            # Plane orientation
            plane_angle = None

            if n_d is not None and n_a is not None:

                plane_angle = np.degrees(
                    np.arccos(
                        np.clip(
                            abs(np.dot(n_d, n_a)),
                            0.0,
                            1.0
                        )
                    )
                )

                if not (
                    plane_angle_min
                    <= plane_angle
                    <= plane_angle_max
                ):
                    continue

            G.add_edge(
                O_d,
                C_a,
                bond_type="carbonyl",
                donor_oxygen=O_d,
                donor_center=donor_center,
                acceptor_carbon=C_a,
                acceptor_oxygen=O_a,
                distance=float(distance),
                burgi_dunitz_angle=float(bd_angle),
                plane_angle=None if plane_angle is None else float(plane_angle)
            )

    return G



# --------------------------------------------------------------------------- #
#  Output writer
# --------------------------------------------------------------------------- #
def output(
        G: nx.Graph,
        filename: str = "default",
        ext: str = ".default",
        correlation=Rozenberg_2000,
) -> None:
    """
    Write a formatted .nci file summarising all detected interactions.
    """

    now = datetime.now()

    # ------------------------------------------------------------------
    # Group edges by interaction type
    # ------------------------------------------------------------------

    edges_by_type = {}

    for _, _, data in G.edges(data=True):
        btype = data.get("bond_type")

        if btype is None:
            continue

        edges_by_type.setdefault(btype, []).append(data)

    # ------------------------------------------------------------------
    # Section configuration
    # ------------------------------------------------------------------

    sections = [
        {
            "key": "HB",
            "title": "HYDROGEN BONDS",
            "header": [
                "№",
                "Type",
                "D",
                "H",
                "A",
                "D-H (Å)",
                "H···A (Å)",
                "D···A (Å)",
                "Angle (°)",
                "Energy (kcal/mol)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_d']}-H···{d['symbol_a']}",
                int(d["d_num"]) + 1,
                int(d["h_num"]) + 1,
                int(d["a_num"]) + 1,
                round(d["length_cov"], 3),
                round(d["length_hb"], 3),
                round(d["length_da"], 3),
                round(d["angle"], 1),
                round(
                    float(
                        correlation(
                            d["length_hb"],
                            f"{d['symbol_d']}-H...{d['symbol_a']}",
                        )
                    ),
                    1,
                ),
            ],
        },

        {
            "key": "XB",
            "title": "HALOGEN BONDS",
            "header": [
                "№",
                "Type",
                "D",
                "X",
                "A",
                "D-X (Å)",
                "X···A (Å)",
                "D···A (Å)",
                "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_d']}-{d['symbol_x']}···{d['symbol_a']}",
                int(d["d_num"]) + 1,
                int(d["x_num"]) + 1,
                int(d["a_num"]) + 1,
                round(d["length_cov"], 3),
                round(d["length_xb"], 3),
                round(d["length_da"], 3),
                round(d["angle"], 1),
            ],
        },

        {
            "key": "ChB",
            "title": "CHALCOGEN BONDS",
            "header": [
                "№",
                "Type",
                "D",
                "X",
                "A",
                "D-X (Å)",
                "X···A (Å)",
                "D···A (Å)",
                "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_d']}-{d['symbol_x']}···{d['symbol_a']}",
                int(d["d_num"]) + 1,
                int(d["x_num"]) + 1,
                int(d["a_num"]) + 1,
                round(d["length_cov"], 3),
                round(d["length_xb"], 3),
                round(d["length_da"], 3),
                round(d["angle"], 1),
            ],
        },

        {
            "key": "PnB",
            "title": "PNICTOGEN BONDS",
            "header": [
                "№",
                "Type",
                "D",
                "X",
                "A",
                "D-X (Å)",
                "X···A (Å)",
                "D···A (Å)",
                "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_d']}-{d['symbol_x']}···{d['symbol_a']}",
                int(d["d_num"]) + 1,
                int(d["x_num"]) + 1,
                int(d["a_num"]) + 1,
                round(d["length_cov"], 3),
                round(d["length_xb"], 3),
                round(d["length_da"], 3),
                round(d["angle"], 1),
            ],
        },

        {
            "key": "stacking",
            "title": "π-π STACKING",
            "header": [
                "№",
                "Type",
                "Atom of Ring 1",
                "Atom of Ring 2",
                "Centroid (Å)",
                "Offset (Å)",
                "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                "parallel",
                int(d["atom_num_cycle_1"]) + 1,
                int(d["atom_num_cycle_2"]) + 1,
                round(d["centroid_distance"], 3),
                round(d["offset"], 3),
                round(d["angle"], 1),
            ],
        },

        {
            "key": "T-stacking",
            "title": "T-SHAPED STACKING",
            "header": [
                "№",
                "Type",
                "Atom of Ring 1",
                "Atom of Ring 2",
                "Centroid (Å)",
                "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                "T-shaped",
                int(d["atom_num_cycle_1"]) + 1,
                int(d["atom_num_cycle_2"]) + 1,
                round(d["centroid_distance"], 3),
                round(d["angle"], 1),
            ],
        },

        {
            "key": "H-pi",
            "title": "X-H···π INTERACTIONS",
            "header": [
                "№",
                "Type",
                "X",
                "H",
                "H···π (Å)",
                "Angle (°)",
                "Projection (Å)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_x']}-H···π",
                int(d["x_num"]) + 1,
                int(d["h_num"]) + 1,
                round(d["h_centroid_distance"], 3),
                round(d["angle"], 1),
                round(d["projection_distance"], 3),
            ],
        },

        {
            "key": "n-pi",
            "title": "LONE PAIR···π",
            "header": [
                "№",
                "Type",
                "Atom",
                "Distance (Å)",
                "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_lp']}···π",
                int(d["lp_num"]) + 1,
                round(d["lp_centroid_distance"], 3),
                round(d["angle"], 1),
            ],
        },

        {
            "key": "carbonyl",
            "title": "CARBONYL n→π*",
            "header": [
                "№",
                "Type",
                "Donor O",
                "Acceptor C",
                "Distance (Å)",
                "BD angle (°)",
                "Plane angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                "n→π*",
                int(d["donor_oxygen"]) + 1,
                int(d["acceptor_carbon"]) + 1,
                round(d["distance"], 3),
                round(d["burgi_dunitz_angle"], 1),
                "—" if d["plane_angle"] is None else round(d["plane_angle"], 1),
            ],
        },
    ]

    # ------------------------------------------------------------------
    # Write file
    # ------------------------------------------------------------------

    with open(f"{filename}.nci", "w", encoding="utf-8") as f:

        f.write(
            f"NCITools: Geometry based analysis\n"
            f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Input file: {filename}{ext}\n"
        )

        for sec in sections:

            rows = [
                sec["row_func"](edge, i)
                for i, edge in enumerate(edges_by_type.get(sec["key"], []))
            ]

            if not rows:
                continue

            f.write("\n")
            f.write("=" * 90 + "\n")
            f.write(sec["title"] + "\n")
            f.write("=" * 90 + "\n\n")

            table = tabulate(
                rows,
                headers=sec["header"],
                tablefmt="github",
                floatfmt=".3f",
                numalign="right",
                stralign="center",
            )

            f.write(table)
            f.write("\n")

        WIDTH = 40

        philosopher, quote = random.choice(quotes)

        f.write("\n\n")

        f.write(
            textwrap.fill(
                quote.upper(),
                width=WIDTH,
                initial_indent=" ",
                subsequent_indent=" ",
            )
        )

        f.write(f"\n   -- {philosopher.upper()}\n")



# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    # Example usage (adjust path as needed)
    file = r"C:\Users\User\Navuka\Proteins_NCI_analysis\SCF for manual\H_optimized\GFN2-xTB\SCF\HF\1ubq_HF-pcseg-1_opt_xtb_xyz.xyz"
    basename = os.path.basename(file)
    name, ext = os.path.splitext(basename)
    _, _, _, atoms = read_input(file)

    # 1. Covalent graph (now returned directly)
    G = build_graph_with_covalent_pairs(atoms)

    # 2. Non‑covalent interactions
    G = find_nci_bonds(atoms, G, ['H'],                      120, 'HB',
                       desc='Searching for hydrogen bonds')
    G = find_nci_bonds(atoms, G, ['Cl', 'Br', 'I'],         150, 'XB',
                       desc='Searching for halogen bonds')
    G = find_nci_bonds(atoms, G, ['S', 'Se', 'Te'],         150, 'ChB',
                       desc='Searching for chalcogen bonds')
    G = find_nci_bonds(atoms, G, ['P', 'As', 'Sb', 'Bi'],   150, 'PnB',
                       desc='Searching for pnictogen bonds')
    G = find_nci_with_aromatic(atoms, G)
    G = find_carbonyl_interactions(atoms, G)

    # 3. Write results
    output(G, filename=name, ext=ext)


if __name__ == "__main__":
    main()