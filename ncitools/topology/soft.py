"""Soft (fuzzy) confidence scores for topology-detected contacts.

This module mirrors :mod:`ncitools.geometry.soft`: every contact found by a
topology detector is scored as a product of smooth criteria in ``[0, 1]``.
The formulae are copied from the geometry detectors so that the *same*
contact receives the *same* confidence score in both modules.

Only the primitives are imported from :mod:`ncitools.geometry.soft`; the
per-family products live here because the geometry implementations are
embedded in their detector loops.

All thresholds below are the geometry defaults (BONDI radii), so a congruence
check between the two modules is meaningful.  No threshold is applied here:
the detectors only store ``confidence_score`` and the report prints it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from ase import Atoms

from ncitools.constants import (
    BONDI,
    BOHR_TO_ANGSTROM,
    CHARRY_TKATCHENKO,
    RADII,
)
from ncitools.geometry.detectors._common import (
    acceptor_angle_range,
    soft_acceptor_angle_score,
    vdw_radius,
)
from ncitools.geometry.soft import soft_greater, soft_less, soft_range
from ncitools.utils import get_symbol


# --------------------------------------------------------------------------- #
#  Geometry defaults shared with the geometry module
# --------------------------------------------------------------------------- #

HB_ANGLE_THRESHOLD = 110.0
SIGMA_HOLE_ANGLE_THRESHOLD = 150.0

PI_CLOUD_RADII = 3.0
STACKING_DISTANCE_THRESHOLD = 5.5
STACKING_ANGLE_THRESHOLD = 30.0
STACKING_OFFSET_THRESHOLD = 2.8
PI_ANGLE_TO_NORMAL_THRESHOLD = 45.0

ION_PI_DISTANCE_THRESHOLD = 5.0
ION_PI_ANGLE_THRESHOLD = 30.0

CARBONYL_ANGLE_MIN = 90.0
CARBONYL_ANGLE_MAX = 130.0


# --------------------------------------------------------------------------- #
#  Atoms adapter
# --------------------------------------------------------------------------- #

def ase_atoms(atoms: Sequence[Tuple[int, np.ndarray]]) -> Atoms:
    """Build a non-periodic :class:`ase.Atoms` from topology coordinates.

    ``atoms`` follows the topology convention ``(atomic_number, position)``
    with positions in Bohr; the returned object uses Angstrom, matching
    :func:`ncitools.geometry.detectors._common.soft_acceptor_angle_score`.
    """
    symbols = [get_symbol(int(atomic_number)) for atomic_number, _ in atoms]
    positions = np.asarray([position for _, position in atoms], dtype=float)
    return Atoms(symbols=symbols, positions=positions * BOHR_TO_ANGSTROM)


# --------------------------------------------------------------------------- #
#  σ-hole family
# --------------------------------------------------------------------------- #

def sigma_hole_score(
    atoms: Atoms,
    G: nx.Graph,
    x: int,
    a: int,
    angle: float,
    angle_threshold: float,
) -> Tuple[float, Optional[float]]:
    """Confidence for a D–X···A contact.

    ``angle`` is the already measured D–X···A angle.  Returns the product and
    the X–A–Y angle chosen by the acceptor criterion (``None`` when skipped).
    """
    symbols = atoms.get_chemical_symbols()

    distance = atoms.get_distance(x, a)
    cutoff = vdw_radius(symbols[x], BONDI) + vdw_radius(symbols[a], BONDI)
    score_dist = soft_less(distance, cutoff)
    score_angle = soft_greater(angle, angle_threshold)

    score_acceptor, angle_xay = soft_acceptor_angle_score(atoms, G, x, a, symbols)

    return float(score_dist * score_angle * score_acceptor), angle_xay


def acceptor_angle_score(
    element: str,
    num_neighbours: int,
    angle: Optional[float],
) -> float:
    """Soft score of the Y–A···X angle for an acceptor ``element``."""
    angle_range = acceptor_angle_range(element, num_neighbours)
    if angle_range is None:
        return 0.0
    if angle_range == "skip":
        return 1.0
    if angle is None:
        return 0.0
    return float(soft_range(angle, *angle_range))


# --------------------------------------------------------------------------- #
#  Aromatic π interactions
# --------------------------------------------------------------------------- #

def stacking_score(
    centroid_distance: float,
    angle: float,
    offset: float,
) -> float:
    """Confidence for parallel π···π stacking."""
    return float(
        soft_less(centroid_distance, STACKING_DISTANCE_THRESHOLD)
        * soft_less(angle, STACKING_ANGLE_THRESHOLD)
        * soft_less(offset, STACKING_OFFSET_THRESHOLD)
    )


def h_pi_score(
    h_centroid_distance: float,
    angle: float,
    projection_distance: float,
    ring_radius: float,
) -> float:
    """Confidence for an X–H···π contact (distances in Angstrom)."""
    return float(
        soft_less(h_centroid_distance, PI_CLOUD_RADII + BONDI["H"])
        * soft_greater(angle, HB_ANGLE_THRESHOLD)
        * soft_less(projection_distance, ring_radius)
    )


def n_pi_score(
    lp_symbol: str,
    distance: float,
    angle_to_normal: float,
    projection_distance: float,
    ring_radius: float,
    score_angle_y: float,
) -> float:
    """Confidence for a lone-pair···π contact (distances in Angstrom)."""
    return float(
        soft_less(distance, vdw_radius(lp_symbol, BONDI) + PI_CLOUD_RADII)
        * soft_less(angle_to_normal, PI_ANGLE_TO_NORMAL_THRESHOLD)
        * soft_less(projection_distance, ring_radius)
        * score_angle_y
    )


def ion_pi_score(distance: float, angle_to_normal: float) -> float:
    """Confidence for an ion···π contact.

    The angle uses the signed projection onto the ring normal, exactly as the
    geometry detector does.
    """
    return float(
        soft_less(distance, ION_PI_DISTANCE_THRESHOLD)
        * soft_less(angle_to_normal, ION_PI_ANGLE_THRESHOLD)
    )


def x_pi_score(
    halogen_symbol: str,
    distance: float,
    angle: float,
    angle_to_normal: float,
) -> float:
    """Confidence for an R–X···π contact (distances in Angstrom)."""
    return float(
        soft_less(distance, PI_CLOUD_RADII + vdw_radius(halogen_symbol, BONDI))
        * soft_greater(angle, SIGMA_HOLE_ANGLE_THRESHOLD)
        * soft_less(angle_to_normal, PI_ANGLE_TO_NORMAL_THRESHOLD)
    )


# --------------------------------------------------------------------------- #
#  Other families
# --------------------------------------------------------------------------- #

def carbonyl_score(
    atoms: Atoms,
    G: nx.Graph,
    donor: int,
    carbon: int,
    burgi_dunitz_angle: float,
) -> Tuple[float, Optional[float]]:
    """Confidence for a Y···C=O contact.

    Returns the product and the Y–C–O acceptor angle chosen by the geometry
    criterion (``None`` when no substituent is available).
    """
    symbols = atoms.get_chemical_symbols()

    distance = atoms.get_distance(donor, carbon)
    cutoff = vdw_radius(symbols[donor], BONDI) + vdw_radius("C", BONDI)
    score_dist = soft_less(distance, cutoff)
    score_bd = soft_range(burgi_dunitz_angle, CARBONYL_ANGLE_MIN, CARBONYL_ANGLE_MAX)

    score_angle, angle = soft_acceptor_angle_score(atoms, G, carbon, donor, symbols)

    return float(score_dist * score_bd * score_angle), angle


def metallophilic_score(
    distance: float,
    symbol_1: str,
    symbol_2: str,
) -> float:
    """Confidence for a metal···metal contact (geometry uses RADII/CHARRY)."""
    cov_cutoff = RADII[symbol_1] + RADII[symbol_2]
    vdw_cutoff = (
        CHARRY_TKATCHENKO[symbol_1] + CHARRY_TKATCHENKO[symbol_2]
    )
    return float(soft_range(distance, cov_cutoff, vdw_cutoff))


def dihydrogen_score(distance: float) -> float:
    """Distance-only confidence for an H···H contact."""
    return float(soft_less(distance, 2.0 * BONDI["H"]))
