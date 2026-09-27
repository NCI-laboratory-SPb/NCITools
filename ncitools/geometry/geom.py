"""
Pure-geometry helpers used by the graph builder and by the detectors.

This module contains functions that operate on atomic coordinates alone:
HOMA aromaticity, best-fit planes, and planarity metrics.  Nothing here
depends on the molecular graph, which makes the functions easy to test
and reuse.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from ase import Atoms

from ncitools.constants import HOMA_PARAMS


# --------------------------------------------------------------------------- #
#  HOMA (Harmonic Oscillator Model of Aromaticity)
# --------------------------------------------------------------------------- #

def canonical_bond_key(symbol_i: str, symbol_j: str) -> str:
    """Return the canonical (alphabetically sorted) key for a bond type."""
    return "".join(sorted((symbol_i, symbol_j)))


def calculate_homa(
    atoms: Atoms,
    cycle: Sequence[int],
    homa_params: Optional[dict] = None,
) -> Optional[float]:
    """
    Compute the HOMA index of a molecular cycle.

    The HOMA index is defined as

        HOMA = 1 - (1 / n) * Σ α_i (R_i - R_opt,i)²

    where the sum runs over the ``n`` bonds of the cycle.  A value close
    to 1 indicates a fully delocalised aromatic ring.

    Parameters
    ----------
    atoms : ase.Atoms
        Molecular structure.
    cycle : sequence of int
        Atom indices forming the ring, in connectivity order.
    homa_params : dict, optional
        Mapping ``"XY" -> (R_opt, α)``.  Defaults to
        :data:`ncitools.constants.HOMA_PARAMS`.

    Returns
    -------
    float or None
        The HOMA value, or ``None`` if any bond type in the cycle lacks
        HOMA parameters.
    """
    if homa_params is None:
        homa_params = HOMA_PARAMS

    n = len(cycle)
    penalty = 0.0

    for k in range(n):
        i = cycle[k]
        j = cycle[(k + 1) % n]

        key = canonical_bond_key(atoms[i].symbol, atoms[j].symbol)

        if key not in homa_params:
            return None

        r_opt, alpha = homa_params[key]
        distance = atoms.get_distance(i, j, mic=True)
        penalty += alpha * (distance - r_opt) ** 2

    return 1.0 - penalty / n


# --------------------------------------------------------------------------- #
#  Best-fit planes and planarity
# --------------------------------------------------------------------------- #

def best_fit_plane(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the centroid and the unit normal of the best-fit plane.

    The plane is found by singular value decomposition of the centred
    coordinates; its normal is the right-singular vector corresponding
    to the smallest singular value.
    """
    coords = np.asarray(coords, dtype=float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return centroid, normal


def planarity_metrics(coords: np.ndarray) -> tuple[float, float]:
    """
    Return (RMS deviation, maximum deviation) of the points from their
    best-fit plane, in ångströms.
    """
    coords = np.asarray(coords, dtype=float)
    centroid, normal = best_fit_plane(coords)
    distances = (coords - centroid) @ normal
    rms = float(np.sqrt(np.mean(distances ** 2)))
    max_dev = float(np.max(np.abs(distances)))
    return rms, max_dev


def plane_deviation(
    point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> float:
    """Signed distance from ``point`` to the plane through ``plane_point``."""
    return float(np.dot(np.asarray(point) - plane_point, plane_normal))