"""
n→π* interactions between a lone-pair donor and a carbonyl acceptor.

Following a soft geometric approach, an n→π* contact between a
non-hydrogen donor atom (Nu) and a carbonyl group (C=O) is accepted when

* the Nu···C distance is shorter than the sum of the van der Waals radii;
* the O=C···Nu angle (the "Bürgi–Dunitz angle") lies inside a
  characteristic window;
* the R–Nu···C angle matches the characteristic lone-pair direction of
  the donor atom.  The allowed range is taken from
  ``acceptor_angle_range`` for the donor element and its coordination
  number.  The best R–Nu···C angle among the covalent neighbours R of Nu
  is scored with ``soft_acceptor_angle_score``.

All three criteria are combined as a product of soft scores.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
from ase import Atoms
from tqdm import tqdm

from ncitools.constants import BONDI
from ncitools.geometry.soft import soft_less, soft_range

from ._common import (
    acceptor_angle_range,
    soft_acceptor_angle_score,
    vdw_radius,
)


def find_carbonyl_interactions(
    atoms: Atoms,
    G: nx.Graph,
    distance_cutoff: Optional[float] = None,
    angle_min: float = 90.0,
    angle_max: float = 130.0,
    confidence_threshold: float = 0.8,
) -> nx.Graph:
    """
    Add carbonyl n→π* interaction edges to ``G``.

    Donors
    ------
    Any non-hydrogen atom for which ``acceptor_angle_range`` returns a
    non-``None`` value, i.e. any atom with at least one available lone
    pair.  The best R–Nu···C angle for each donor is evaluated later with
    ``soft_acceptor_angle_score``.

    Acceptors
    ---------
    Carbonyl groups C=O, i.e. oxygen atoms with exactly one covalent
    neighbour which is a carbon.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    G : nx.Graph
        Molecular graph.  Modified in place.
    distance_cutoff : float, optional
        Maximum Nu···C distance in ångströms.  If ``None``, the cutoff is
        computed for each donor as
        ``vdw_radius(donor) + vdw_radius('C')``.
    angle_min, angle_max : float
        Allowed Bürgi–Dunitz angle range (degrees), i.e. the O=C···Nu
        angle at the carbonyl carbon.
    confidence_threshold : float
        Minimum product of soft scores required to accept an interaction.

    Returns
    -------
    nx.Graph
        The input graph with new ``carbonyl`` edges added.
    """
    symbols = atoms.get_chemical_symbols()

    donors = _collect_donors(atoms, G, symbols)
    acceptors = _collect_acceptors(atoms, G, symbols)

    for donor in tqdm(
        donors,
        desc="Searching for n→π* interactions with carbonyls",
    ):
        _match_donor_against_acceptors(
            atoms, G, donor, acceptors, symbols,
            distance_cutoff,
            angle_min, angle_max,
            confidence_threshold,
        )

    return G


# --------------------------------------------------------------------------- #
#  Donor / acceptor collection
# --------------------------------------------------------------------------- #

def _covalent_neighbours(G: nx.Graph, idx: int) -> list:
    """Heavy (non-virtual) covalent neighbours of ``idx``."""
    return [
        n for n in G.adj[idx]
        if not isinstance(n, str)
        and G.edges[idx, n].get("bond_type") in ("covalent", "coordination")
    ]


def _collect_donors(atoms: Atoms, G: nx.Graph, symbols) -> list:
    """
    Lone-pair donors: any non-hydrogen atom for which
    ``acceptor_angle_range`` returns a non-``None`` value.

    The coordination number is stored on the graph node as
    ``num_neighbours`` because ``soft_acceptor_angle_score`` expects it.
    """
    donors = []

    for idx, symbol in enumerate(symbols):
        if symbol == "H":
            continue

        num_neighbours = G.nodes[idx]["num_neighbours"]

        angle_range = acceptor_angle_range(symbol, num_neighbours)
        if angle_range is None:
            continue

        donors.append({
            "atom": idx,
            "element": symbol,
        })

    return donors


def _collect_acceptors(atoms: Atoms, G: nx.Graph, symbols) -> list:
    """Carbonyl acceptors: O with exactly one covalent neighbour, a carbon."""
    acceptors = []

    for o, symbol in enumerate(symbols):
        if symbol != "O":
            continue

        neighbours = _covalent_neighbours(G, o)
        if len(neighbours) != 1:
            continue

        carbon = neighbours[0]
        if symbols[carbon] != "C" or G.nodes[carbon]["num_neighbours"] == 4:
            continue

        acceptors.append({
            "oxygen": o,
            "carbon": carbon,
        })

    return acceptors


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #

def _match_donor_against_acceptors(
    atoms: Atoms,
    G: nx.Graph,
    donor: dict,
    acceptors: list,
    symbols,
    distance_cutoff: Optional[float],
    angle_min: float,
    angle_max: float,
    confidence_threshold: float,
) -> None:
    """Test every carbonyl acceptor against a single donor."""
    donor_atom = donor["atom"]
    donor_element = donor["element"]

    for acceptor in acceptors:
        o_a = acceptor["oxygen"]
        c_a = acceptor["carbon"]

        # Donor and acceptor oxygen may not be the same atom.
        if donor_atom in G.adj[c_a]:
            continue

        distance = atoms.get_distance(donor_atom, c_a, mic=True)

        if distance_cutoff is None:
            cutoff = (
                vdw_radius(donor_element, BONDI)
                + vdw_radius("C", BONDI)
            )
        else:
            cutoff = distance_cutoff

        score_dist = soft_less(distance, cutoff)
        if score_dist < 0.01:
            continue

        # Bürgi–Dunitz angle: O=C···Nu, i.e. the angle at the carbonyl
        # carbon between the carbonyl oxygen and the donor atom.
        bd_angle = atoms.get_angle(o_a, c_a, donor_atom, mic=True)
        score_bd = soft_range(bd_angle, angle_min, angle_max)
        if score_bd < 0.01:
            continue

        # Best R–Nu···C angle for this donor.
        score_angle, angle = soft_acceptor_angle_score(
            atoms, G, c_a, donor_atom, symbols
        )
        if score_angle < 0.01:
            continue

        total = score_dist * score_bd * score_angle
        if total < confidence_threshold:
            continue

        G.add_edge(
            donor_atom, c_a,
            bond_type="carbonyl",
            donor_atom=donor_atom,
            donor_element=donor_element,
            acceptor_carbon=c_a,
            acceptor_oxygen=o_a,
            distance=float(distance),
            burgi_dunitz_angle=float(bd_angle),
            angle=float(angle) if angle is not None else None,
            confidence_score=float(total),
        )