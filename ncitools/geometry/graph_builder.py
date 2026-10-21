"""
Construction of the molecular graph from atomic coordinates.

The graph is built in four stages:

1. Each atom becomes a node carrying its element and position.
2. Covalent bonds between non-metal atoms are inferred from covalent radii.
3. Metal–non-metal contacts are classified as coordination bonds using a
   more generous cutoff (ionic + covalent radii + tolerance).
4. Planar 5–9 membered cycles with a HOMA index above a threshold are added
   as explicit ``aromatic_center`` nodes connected to every atom of the ring.

The resulting graph is the input to every detector in
:mod:`ncitools.geometry.detectors`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from ncitools.constants import HALOGENS, HOMA_PARAMS, METALS, RADII
from ncitools.geometry.geom import (
    best_fit_plane,
    calculate_homa,
    planarity_metrics,
)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def build_graph_with_covalent_pairs(
    atoms: Atoms,
    radii: Optional[dict] = None,
    tolerance: float = 0.2,
    min_cycle_size: int = 5,
    max_cycle_size: int = 7,
    planarity_tol: float = 0.10,
    homa_threshold: float = 0.65,
    homa_params: Optional[dict] = None,
) -> nx.Graph:
    """
    Build a molecular graph from atomic coordinates.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    radii : dict, optional
        Covalent radii in ångströms.  Defaults to
        :data:`ncitools.constants.RADII`.
    tolerance : float
        Extra slack added to covalent cutoffs (in ångströms).
    min_cycle_size, max_cycle_size : int
        Ring-size window for aromatic-cycle candidates.
    planarity_tol : float
        Maximum RMS deviation (Å) from the best-fit plane for a cycle
        to be considered planar.
    homa_threshold : float
        Minimum HOMA index required for a planar ring to be marked
        aromatic.
    homa_params : dict, optional
        HOMA parameters, passed through to
        :func:`ncitools.geometry.geom.calculate_homa`.

    Returns
    -------
    networkx.Graph
        Molecular graph with nodes of type ``"atom"`` and
        ``"aromatic_center"``, and edges of type ``"covalent"``,
        ``"coordination"`` and ``"aromatic"``.
    """
    if radii is None:
        radii = RADII
    if homa_params is None:
        homa_params = HOMA_PARAMS

    symbols = atoms.get_chemical_symbols()

    # ------------------------------------------------------------------ #
    # Atom nodes
    # ------------------------------------------------------------------ #
    G = nx.Graph()
    for i, atom in enumerate(atoms):
        G.add_node(
            i,
            node_type="atom",
            element=atom.symbol,
            position=atom.position.copy(),
        )

    nonmetal_indices = [i for i, s in enumerate(symbols) if s not in METALS]
    metal_indices = [i for i, s in enumerate(symbols) if s in METALS]

    _add_covalent_bonds(atoms, G, nonmetal_indices, radii, tolerance)
    _add_coordination_bonds(
        atoms, G, metal_indices, nonmetal_indices, radii, tolerance
    )
    _add_aromatic_centres(
        atoms, G, nonmetal_indices,
        min_cycle_size, max_cycle_size,
        planarity_tol, homa_threshold, homa_params,
    )
    _assign_coordination_numbers(atoms, G, symbols)
    _check_hydrogen_connectivity(atoms, G)

    return G


# --------------------------------------------------------------------------- #
#  Covalent bonds
# --------------------------------------------------------------------------- #

def _add_covalent_bonds(
    atoms: Atoms,
    G: nx.Graph,
    nonmetal_indices: Sequence[int],
    radii: dict,
    tolerance: float,
) -> None:
    """Add covalent bonds between pairs of non-metal atoms."""
    if not nonmetal_indices:
        return

    symbols = atoms.get_chemical_symbols()
    nonmetal_atoms = atoms[nonmetal_indices]

    cutoffs = [
        radii[nonmetal_atoms[i].symbol] + tolerance
        for i in range(len(nonmetal_atoms))
    ]

    i_list, j_list = neighbor_list("ij", nonmetal_atoms, cutoffs)

    for i_local, j_local in zip(i_list, j_list):
        if i_local >= j_local:
            continue

        i = nonmetal_indices[i_local]
        j = nonmetal_indices[j_local]

        cutoff = radii[symbols[i]] + radii[symbols[j]] + tolerance
        if atoms.get_distance(i, j, mic=True) <= cutoff:
            G.add_edge(i, j, bond_type="covalent")


# --------------------------------------------------------------------------- #
#  Metal coordination bonds
# --------------------------------------------------------------------------- #

def _add_coordination_bonds(
    atoms: Atoms,
    G: nx.Graph,
    metal_indices: Sequence[int],
    nonmetal_indices: Sequence[int],
    radii: dict,
    tolerance: float,
) -> None:
    """Add coordination bonds between metals and nearby non-metals."""
    symbols = atoms.get_chemical_symbols()

    for metal_idx in metal_indices:
        metal_radius = radii[symbols[metal_idx]]

        for nonmetal_idx in nonmetal_indices:
            nonmetal_symbol = symbols[nonmetal_idx]

            # An isolated halogen (e.g. a coordinated halide) has a larger
            # effective radius than its covalent radius suggests.
            if nonmetal_symbol in HALOGENS and G.degree(nonmetal_idx) == 0:
                nonmetal_radius = HALOGENS[nonmetal_symbol]
            else:
                nonmetal_radius = radii[nonmetal_symbol]

            cutoff = metal_radius + nonmetal_radius + 3.0 * tolerance
            if atoms.get_distance(metal_idx, nonmetal_idx, mic=True) < cutoff:
                G.add_edge(
                    metal_idx, nonmetal_idx,
                    bond_type="coordination",
                )


# --------------------------------------------------------------------------- #
#  Aromatic centres
# --------------------------------------------------------------------------- #

def _add_aromatic_centres(
    atoms: Atoms,
    G: nx.Graph,
    nonmetal_indices: Sequence[int],
    min_cycle_size: int,
    max_cycle_size: int,
    planarity_tol: float,
    homa_threshold: float,
    homa_params: dict,
) -> None:
    """
    Identify planar, HOMA-aromatic rings and add one virtual node per ring.

    The virtual node carries the ring geometry (centre, normal, radius)
    and is connected to every atom of the ring with an ``aromatic`` edge.
    Only covalent non-metal bonds participate in cycles, so coordination
    bonds cannot create artificial rings around metal centres.
    """
    covalent = nx.Graph()
    covalent.add_nodes_from(nonmetal_indices)
    for i, j, data in G.edges(data=True):
        if data.get("bond_type") == "covalent":
            covalent.add_edge(i, j)

    aromatic_index = 0
    cycles = nx.simple_cycles(covalent, length_bound=max_cycle_size)

    for cycle in cycles:
        ring_size = len(cycle)
        if not (min_cycle_size <= ring_size <= max_cycle_size):
            continue

        coords = atoms.positions[cycle]

        rms, max_dev = planarity_metrics(coords)
        if rms > planarity_tol:
            continue

        homa = calculate_homa(atoms, cycle, homa_params=homa_params)
        if homa is None or homa < homa_threshold:
            continue

        centre, normal = best_fit_plane(coords)
        radius = float(np.mean(np.linalg.norm(coords - centre, axis=1)))
        plane_d = float(-np.dot(normal, centre))
        cycle_tuple = tuple(sorted(cycle))

        ring_node = f"ring_{aromatic_index}"
        aromatic_index += 1

        G.add_node(
            ring_node,
            node_type="aromatic_center",
            position=centre,
            normal=normal,
            plane_d=plane_d,
            radius=radius,
            cycle=cycle_tuple,
            size=ring_size,
            aromatic=True,
            homa=float(homa),
            rms_planarity=float(rms),
            max_planarity_deviation=float(max_dev),
        )

        for atom_idx in cycle:
            G.add_edge(
                ring_node, atom_idx,
                bond_type="aromatic",
                ring=cycle_tuple,
            )


# --------------------------------------------------------------------------- #
#  Coordination numbers and sanity checks
# --------------------------------------------------------------------------- #

def _count_covalent_neighbours(G: nx.Graph, atom_idx: int) -> int:
    """Number of covalent + coordination neighbours of ``atom_idx``."""
    return sum(
        1
        for nbr in G.neighbors(atom_idx)
        if not isinstance(nbr, str)
        and G.edges[atom_idx, nbr].get("bond_type")
        in ("covalent", "coordination")
    )


def _assign_coordination_numbers(
    atoms: Atoms,
    G: nx.Graph,
    symbols: Sequence[str],
) -> None:
    """
    Store the coordination number of each atom under ``num_neighbours``.

    Planar trivalent pnictogens (e.g. amide nitrogen) get an extra count
    because their lone pair effectively occupies a fourth coordination
    position.
    """
    pnictogens = {"N", "P", "As", "Sb", "Bi"}

    for atom_idx in range(len(atoms)):
        num_neighbours = _count_covalent_neighbours(G, atom_idx)

        if symbols[atom_idx] in pnictogens and num_neighbours == 3:
            if _is_planar_centre(atoms, G, atom_idx, tolerance=0.15):
                num_neighbours += 1

        G.nodes[atom_idx]["num_neighbours"] = num_neighbours


def _is_planar_centre(
    atoms: Atoms,
    G: nx.Graph,
    atom_idx: int,
    tolerance: float = 0.1,
) -> bool:
    """
    Return True if ``atom_idx`` lies within ``tolerance`` ångströms of
    the plane defined by its covalent/coordination neighbours.
    """
    n_pos = atoms.positions[atom_idx]
    neighbors = list(neighbour
            for neighbour in G.neighbors(atom_idx)
            if G.edges[atom_idx, neighbour].get("bond_type") in ("covalent", "coordination"))

    p1, p2, p3 = atoms.positions[neighbors]

    center = (p1 + p2 + p3) / 3

    distance = np.linalg.norm(center - n_pos)

    return distance <= tolerance


def _check_hydrogen_connectivity(atoms: Atoms, G: nx.Graph) -> None:
    """Warn if any hydrogen is bonded to more or fewer than one heavy atom."""
    for atom_idx, atom in enumerate(atoms):
        if atom.symbol != "H":
            continue

        count = sum(
            1
            for nbr in G.neighbors(atom_idx)
            if not isinstance(nbr, str)
            and G.edges[atom_idx, nbr].get("bond_type")
            in ("covalent", "coordination")
        )
        if count != 1:
            print(
                f"WARNING: hydrogen atom {atom_idx + 1} has a number of "
                "covalent and/or coordination neighbours different from 1."
            )
    return