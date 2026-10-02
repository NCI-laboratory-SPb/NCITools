"""
σ-hole interactions: hydrogen, halogen, chalcogen, pnictogen and tetrel bonds.

A σ-hole contact D–X···A is characterised by

* X being an electron-poor region (H, halogen, chalcogen, pnictogen, tetrel);
* D being a covalent partner of X;
* A being a Lewis base.

Three geometric criteria are combined multiplicatively:

1. the X···A distance is shorter than the sum of the van der Waals radii;
2. the D–X···A angle is larger than ``angle_threshold`` (interactions are
   directional and prefer a linear D–X···A arrangement);
3. the X–A–Y angle at the acceptor respects the acceptor's hybridisation
   (see :func:`ncitools.geometry.detectors._common.acceptor_angle_range`).

Each criterion contributes a soft score in [0, 1]; their product must
exceed ``confidence_threshold`` for the contact to be recorded.
"""

from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx
from ase import Atoms
from tqdm import tqdm

from ncitools.constants import BONDI, RADII, CHARRY_TKATCHENKO
from ncitools.geometry.detectors._common import soft_acceptor_angle_score, vdw_radius
from ncitools.geometry.soft import soft_greater, soft_less


def find_nci_bonds(
    atoms: Atoms,
    G: nx.Graph,
    central_elements: Iterable[str],
    angle_threshold: float,
    bond_type: str,
    desc: str = "",
    radii: Optional[dict] = None,
    confidence_threshold: float = 0.8,
) -> nx.Graph:
    """
    Add σ-hole interaction edges D–X···A to ``G``.

    For every atom X belonging to ``central_elements``, each of its
    covalent or coordination partners D is examined, and every heavy
    atom A (not H, not necessarily not C) is tested as a potential
    acceptor.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    G : nx.Graph
        Molecular graph containing at least the covalent edges.
        Modified in place.
    central_elements : iterable of str
        Element symbols acting as the bridge atom X, e.g. ``['H']`` for
        hydrogen bonds, ``['F', 'Cl', 'Br', 'I']`` for halogen bonds.
    angle_threshold : float
        Minimum D–X···A angle in degrees.
    bond_type : str
        Edge label stored as ``bond_type`` (``'HB'``, ``'XB'``, …).
    desc : str
        Description shown in the tqdm progress bar.
    radii : dict, optional
        Van der Waals radii.  Defaults to
        :data:`ncitools.constants.BONDI`.
    confidence_threshold : float
        Minimum product of soft scores required to accept an interaction.

    Returns
    -------
    nx.Graph
        The input graph with new interaction edges added.
    """
    if radii is None:
        radii = BONDI

    symbols = atoms.get_chemical_symbols()

    x_indices = [i for i, s in enumerate(symbols) if s in central_elements]
    acceptor_indices = [i for i, s in enumerate(symbols) if s != "H"]

    if not acceptor_indices:
        return G

    for x in tqdm(x_indices, desc=desc):
        x_el = symbols[x]

        for d in list(G.adj[x]):
            d_el = symbols[d]

            if isinstance(d, str) or d_el == 'H':
                continue
            if G.edges[x, d]["bond_type"] not in ("covalent", "coordination"):
                continue


            # Sort acceptors by distance from X so we can stop early.
            distances = atoms.get_distances(x, acceptor_indices, mic=True)
            pairs = sorted(zip(distances, acceptor_indices),
                           key=lambda p: p[0])
            if not pairs:
                continue

            sorted_dists, sorted_inds = zip(*pairs)

            # The closest heavy atom is normally the donor D itself.
            for dist, a in zip(sorted_dists[1:], sorted_inds[1:]):
                a_el = symbols[a]

                cutoff = vdw_radius(x_el, radii) + vdw_radius(a_el, radii)
                score_dist = soft_less(dist, cutoff)
                if score_dist < 0.01:
                    # Distances only grow further down the sorted list.
                    break

                angle_dxa = atoms.get_angle(d, x, a, mic=True)
                score_angle_dxa = soft_greater(angle_dxa, angle_threshold)

                score_angle_xay, angle_xay = soft_acceptor_angle_score(
                    atoms, G, x, a, symbols
                )

                total = score_dist * score_angle_dxa * score_angle_xay

                if total < confidence_threshold:
                    continue

                G.add_edge(
                    x, a,
                    bond_type=bond_type,
                    d_num=d,
                    x_num=x,
                    a_num=a,
                    symbol_d=d_el,
                    symbol_x=x_el,
                    symbol_a=a_el,
                    length_cov=atoms.get_distance(x, d, mic=True),
                    length_xb=dist,
                    length_da=atoms.get_distance(a, d, mic=True),
                    angle=angle_dxa,
                    angle_xay=angle_xay,
                    confidence_score=total,
                )

    return G


