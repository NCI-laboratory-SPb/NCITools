"""
Metallophilic-like contacts between closed-shell metal centres.

Two metal atoms are flagged as a potential metallophilic contact when
their distance falls between the sum of their covalent radii (which
would indicate a real bond) and the sum of their van der Waals radii
(which would indicate no interaction at all).

Only the metals known to form metallophilic contacts are considered,
namely Au, Ag, Cu, Hg, Ir, Pt, Pd and Ni.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
from ase import Atoms
from tqdm import tqdm

from ncitools.constants import CHARRY_TKATCHENKO, RADII
from ncitools.geometry.soft import soft_range


METALLOPHILIC_METALS = ("Au", "Ag", "Cu", "Hg", "Ir", "Pt", "Pd", "Ni")


def find_intermetalls_contact(
    atoms: Atoms,
    G: nx.Graph,
    radii_vdw: Optional[dict] = None,
    radii_cov: Optional[dict] = None,
    confidence_threshold: float = 0.8,
) -> nx.Graph:
    """
    Detect intermetallic close contacts from purely geometric criteria.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    G : nx.Graph
        Molecular graph.  Modified in place.
    radii_vdw : dict, optional
        Van der Waals radii.  Defaults to
        :data:`ncitools.constants.CHARRY_TKATCHENKO`.
    radii_cov : dict, optional
        Covalent radii.  Defaults to
        :data:`ncitools.constants.RADII`.
    confidence_threshold : float
        Minimum soft score required to accept a contact.

    Returns
    -------
    nx.Graph
        The input graph with new ``metallophilic`` edges added.
    """
    if radii_vdw is None:
        radii_vdw = CHARRY_TKATCHENKO
    if radii_cov is None:
        radii_cov = RADII

    symbols = atoms.get_chemical_symbols()
    indices = [
        i for i, s in enumerate(symbols)
        if s in METALLOPHILIC_METALS
    ]

    if len(indices) <= 1:
        return G

    for pos, i in tqdm(
        enumerate(indices),
        desc="Searching for metallophilic interactions",
    ):
        sym_i = symbols[i]

        for j in indices[pos:]:
            sym_j = symbols[j]
            distance = atoms.get_distance(i, j, mic=True)

            cov_cutoff = radii_cov[sym_i] + radii_cov[sym_j]
            vdw_cutoff = radii_vdw[sym_i] + radii_vdw[sym_j]

            total = soft_range(distance, cov_cutoff, vdw_cutoff)
            if total < confidence_threshold:
                continue

            G.add_edge(
                i, j,
                bond_type="metallophilic",
                metal_1=i,
                metal_2=j,
                metal_sym_1=sym_i,
                metal_sym_2=sym_j,
                distance=float(distance),
                confidence_score=float(total),
            )

    return G