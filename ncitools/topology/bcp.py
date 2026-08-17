"""
Topological analysis of electron density: BCP search, NCI filtering, energy estimation.
"""

import numpy as np
from scipy.ndimage import spline_filter
from scipy.spatial import cKDTree
from numpy.linalg import solve, eigvalsh, norm
from concurrent.futures import ProcessPoolExecutor
import itertools
import networkx as nx
from tqdm import tqdm
from datetime import datetime
import os
from tabulate import tabulate
import random
import textwrap

from ncitools.quotes import quotes
from ncitools.correlations import correlation_1   # default energy estimator
from ncitools.utils import get_symbol
from ncitools.constants import bohr_to_angstrom, BONDI

# =========================================================
# 1. Быстрое чтение cube
# =========================================================

def read_cube(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    natoms = int(float(lines[2].split()[0]))
    origin = np.array(list(map(float, lines[2].split()[1:4])))

    nx, dx = int(float(lines[3].split()[0])), float(lines[3].split()[1])
    ny, dy = int(float(lines[4].split()[0])), float(lines[4].split()[2])
    nz, dz = int(float(lines[5].split()[0])), float(lines[5].split()[3])

    spacing = np.array([dx, dy, dz])

    atoms = []
    for i in range(natoms):
        parts = lines[6 + i].split()
        Z = int(float(parts[0]))
        coord = np.array(list(map(float, parts[2:5])))
        atoms.append((Z, coord))

    data = np.fromstring(" ".join(lines[6 + natoms:]), sep=" ")
    density = data.reshape((nx, ny, nz))

    return origin, spacing, density, atoms

# ============================================================================
#  B‑spline basis functions (cubic, order 3) and their derivatives
# ============================================================================

def basis(t):
    """Cubic B‑spline basis function β(t)."""
    t = np.asarray(t)
    result = np.zeros_like(t)
    # intervals
    m1 = (t >= -2) & (t <= -1)
    m2 = (t > -1) & (t <= 0)
    m3 = (t > 0) & (t <= 1)
    m4 = (t > 1) & (t <= 2)
    t1 = t[m1]
    result[m1] = (1.0/6.0) * (t1 + 2.0)**3
    t2 = t[m2]
    result[m2] = (1.0/6.0) * (-3.0*t2**3 - 6.0*t2**2 + 4.0)
    t3 = t[m3]
    result[m3] = (1.0/6.0) * ( 3.0*t3**3 - 6.0*t3**2 + 4.0)
    t4 = t[m4]
    result[m4] = (1.0/6.0) * (2.0 - t4)**3
    return result

def basis_deriv1(t):
    """First derivative β'(t)."""
    t = np.asarray(t)
    result = np.zeros_like(t)
    m1 = (t >= -2) & (t <= -1)
    m2 = (t > -1) & (t <= 0)
    m3 = (t > 0) & (t <= 1)
    m4 = (t > 1) & (t <= 2)
    t1 = t[m1]
    result[m1] = 0.5 * (t1 + 2.0)**2
    t2 = t[m2]
    result[m2] = 0.5 * (-3.0*t2**2 - 4.0*t2)
    t3 = t[m3]
    result[m3] = 0.5 * ( 3.0*t3**2 - 4.0*t3)
    t4 = t[m4]
    result[m4] = -0.5 * (2.0 - t4)**2
    return result

def basis_deriv2(t):
    """Second derivative β''(t)."""
    t = np.asarray(t)
    result = np.zeros_like(t)
    m1 = (t >= -2) & (t <= -1)
    m2 = (t > -1) & (t <= 0)
    m3 = (t > 0) & (t <= 1)
    m4 = (t > 1) & (t <= 2)
    t1 = t[m1]
    result[m1] = t1 + 2.0
    t2 = t[m2]
    result[m2] = -3.0*t2 - 2.0
    t3 = t[m3]
    result[m3] =  3.0*t3 - 2.0
    t4 = t[m4]
    result[m4] = 2.0 - t4
    return result

# ============================================================================
#  B‑spline interpolator with analytical derivatives
# ============================================================================

class BSplineAIM:
    """
    Interpolator for a 3D scalar field (electron density) using cubic B‑splines.
    Provides value, gradient and Hessian at any point in world coordinates.
    """

    def __init__(self, coeffs, origin, spacing):
        self.coeffs = coeffs                 # spline coefficients (same shape as density)
        self.origin = np.asarray(origin)     # (x0, y0, z0)
        self.spacing = np.asarray(spacing)   # (dx, dy, dz)
        self.inv_spacing = 1.0 / self.spacing

    @classmethod
    def from_density(cls, density, origin, spacing):
        """Construct from a raw density grid (calls spline_filter internally)."""
        coeffs = spline_filter(density, order=3)
        return cls(coeffs, origin, spacing)

    def world_to_grid(self, r):
        """Convert world coordinates to grid (pixel) coordinates."""
        return (r - self.origin) * self.inv_spacing

    def _local_coeffs_and_weights(self, g):
        """
        For a grid point g = (x,y,z) (float) return the indices of the
        surrounding knots and the basis function values (and derivatives)
        needed to evaluate the spline.
        """
        # Indices of the knot left of g
        i0 = int(np.floor(g[0]))
        j0 = int(np.floor(g[1]))
        k0 = int(np.floor(g[2]))

        # Candidate indices (4 per direction because cubic spline support = [-2,2])
        i_cand = np.arange(i0 - 1, i0 + 3)
        j_cand = np.arange(j0 - 1, j0 + 3)
        k_cand = np.arange(k0 - 1, k0 + 3)

        # Clip to the valid range (grid boundary handling, similar to 'nearest')
        shape = self.coeffs.shape
        i_inds = np.unique(np.clip(i_cand, 0, shape[0] - 1))
        j_inds = np.unique(np.clip(j_cand, 0, shape[1] - 1))
        k_inds = np.unique(np.clip(k_cand, 0, shape[2] - 1))

        # Distances to the actual knots
        ti = g[0] - i_inds
        tj = g[1] - j_inds
        tk = g[2] - k_inds

        # Basis values and derivatives
        Bi = basis(ti)
        Bj = basis(tj)
        Bk = basis(tk)

        dBi = basis_deriv1(ti)
        dBj = basis_deriv1(tj)
        dBk = basis_deriv1(tk)

        d2Bi = basis_deriv2(ti)
        d2Bj = basis_deriv2(tj)
        d2Bk = basis_deriv2(tk)

        return i_inds, j_inds, k_inds, Bi, Bj, Bk, dBi, dBj, dBk, d2Bi, d2Bj, d2Bk

    def compute_all(self, r):
        """
        Return (value, gradient, Hessian) at world point r in one call.
        """
        g = self.world_to_grid(r)
        (i_inds, j_inds, k_inds,
         Bi, Bj, Bk,
         dBi, dBj, dBk,
         d2Bi, d2Bj, d2Bk) = self._local_coeffs_and_weights(g)

        val = 0.0
        grad = np.zeros(3)
        hess = np.zeros((3, 3))

        # Loop over the (at most) 4x4x4 = 64 contributing knots
        for ii, bi, dbi, d2bi in zip(i_inds, Bi, dBi, d2Bi):
            for jj, bj, dbj, d2bj in zip(j_inds, Bj, dBj, d2Bj):
                for kk, bk, dbk, d2bk in zip(k_inds, Bk, dBk, d2Bk):
                    c = self.coeffs[ii, jj, kk]
                    val += c * bi * bj * bk
                    grad[0] += c * dbi * bj * bk
                    grad[1] += c * bi * dbj * bk
                    grad[2] += c * bi * bj * dbk
                    hess[0, 0] += c * d2bi * bj * bk
                    hess[1, 1] += c * bi * d2bj * bk
                    hess[2, 2] += c * bi * bj * d2bk
                    hess[0, 1] += c * dbi * dbj * bk
                    hess[0, 2] += c * dbi * bj * dbk
                    hess[1, 2] += c * bi * dbj * dbk

        # Symmetrise mixed derivatives
        hess[1, 0] = hess[0, 1]
        hess[2, 0] = hess[0, 2]
        hess[2, 1] = hess[1, 2]

        # Convert derivatives from grid to world coordinates
        grad = grad * self.inv_spacing
        hess = hess * np.outer(self.inv_spacing, self.inv_spacing)

        return val, grad, hess

    def value(self, r):
        val, _, _ = self.compute_all(r)
        return val

    def gradient(self, r):
        _, grad, _ = self.compute_all(r)
        return grad

    def hessian(self, r):
        _, _, hess = self.compute_all(r)
        return hess


# ============================================================================
#  Newton search for a critical point (gradient = 0)
# ============================================================================

def newton_bcp(interp, r0, tol=1e-6, max_iter=200):
    """
    Find a critical point (where gradient vanishes) using Newton's method.
    Returns the position or None if it fails.
    """
    r = r0.copy()
    for _ in range(max_iter):
        _, grad, hess = interp.compute_all(r)
        g_norm = norm(grad)
        if g_norm < tol:
            return r
        try:
            step = solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        r = r - 0.5 * step   # full Newton step (damping removed for speed)
    return None


# ============================================================================
#  AIM analysis at a critical point
# ============================================================================

def aim_analysis(interp, r):
    """Compute all AIM quantities at point r."""
    rho, grad, hess = interp.compute_all(r)
    eigs = eigvalsh(hess)
    lap = np.sum(eigs)
    lambda1, lambda2, lambda3 = eigs
    ellipticity = (lambda1 / lambda2) - 1.0 if lambda2 != 0 else 0.0
    return {
        "position": r.copy(),
        "rho": rho,
        "laplacian": lap,
        "eigenvalues": eigs,
        "ellipticity": ellipticity,
        "grad_norm": norm(grad)
    }


# ============================================================================
#  Parallel search for bond critical points (BCP) between atom pairs
# ============================================================================

def search_pair(args):
    """
    Worker function for parallel execution.
    Args: (coeffs, origin, spacing, atom1, atom2)
    """
    coeffs, origin, spacing, atom1, atom2 = args
    # Create interpolator locally (avoids pickling issues)
    interp = BSplineAIM(coeffs, origin, spacing)

    # Initial guess = midpoint of the two atoms
    r0 = 0.5 * (atom1[1] + atom2[1])

    r_cp = newton_bcp(interp, r0)
    if r_cp is None:
        return None

    # Verify it is a (3,-1) critical point (two negative eigenvalues)
    _, _, hess = interp.compute_all(r_cp)
    eigs = eigvalsh(hess)
    if np.sum(eigs < 0) == 2:
        return aim_analysis(interp, r_cp)
    return None


def find_bcp_parallel(interp, atoms, cutoff=3.0, nproc=4):
    """
    Find all bond critical points by scanning atom pairs up to a distance cutoff.
    Parallelised with multiprocessing.
    """
    cutoff /= bohr_to_angstrom
    # Prepare arguments for each pair
    pairs = []
    for a1, a2 in itertools.combinations(atoms, 2):
        if norm(a1[1] - a2[1]) < cutoff:
            pairs.append((interp.coeffs, interp.origin, interp.spacing, a1, a2))

    # Run in parallel
    with ProcessPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(search_pair, pairs))

    # Filter out None
    cps = [r for r in results if r is not None]

    # Remove duplicates (cluster within 0.1 Å)
    if cps:
        coords = np.array([cp["position"] for cp in cps])
        tree = cKDTree(coords)
        groups = tree.query_ball_tree(tree, r=0.1)
        unique = []
        used = set()
        for i, group in enumerate(groups):
            if i in used:
                continue
            used.update(group)
            unique.append(cps[i])
        return unique
    return []


def find_nci(
    bcps,
    atoms=None,
    rho_min=0.001,
    rho_max=0.1
):
    """
    Split BCPs into non-covalent and covalent sets
    according to electron density.

    The 'atoms' argument is retained for API compatibility,
    but is no longer required.
    """

    bcps_nci = [
        bcp
        for bcp in bcps
        if rho_min <= bcp["rho"] <= rho_max
    ]

    bcps_covalent = [
        bcp
        for bcp in bcps
        if bcp["rho"] > rho_max
    ]

    return bcps_nci, bcps_covalent



# ============================================================
# Helpers for NCI graph construction
# ============================================================

def _covalent_neighbors(G, atom_idx):
    """
    Return covalently bonded neighbours of an atom.
    """
    return [
        n for n in G.adj[atom_idx]
        if G.edges[atom_idx, n].get("bond_type") == "covalent"
    ]


def _angle(a, b, c):
    """
    Calculate angle ABC in degrees.

    Parameters
    ----------
    a, b, c : array-like
        Cartesian coordinates.

    Returns
    -------
    float or None
        Angle in degrees.
    """
    v1 = np.asarray(a) - np.asarray(b)
    v2 = np.asarray(c) - np.asarray(b)

    n1 = norm(v1)
    n2 = norm(v2)

    if n1 == 0.0 or n2 == 0.0:
        return None

    cos_angle = np.dot(v1, v2) / (n1 * n2)

    return float(
        np.degrees(
            np.arccos(
                np.clip(cos_angle, -1.0, 1.0)
            )
        )
    )


def _aromatic_rings_for_atom(G, atom_idx):
    """
    Return aromatic-center nodes directly connected to atom_idx
    through an aromatic edge.
    """
    return [
        neighbour
        for neighbour in G.adj[atom_idx]
        if (
            G.edges[atom_idx, neighbour].get("bond_type")
            == "aromatic"
            and
            G.nodes[neighbour].get("node_type")
            == "aromatic_center"
        )
    ]


def _aromatic_membership(G, atom_idx):
    """
    Determine whether an atom belongs to an aromatic interaction.

    An atom is considered aromatic if:
      1. it is directly connected to an aromatic center, or
      2. it is H covalently bonded to an atom belonging to an
         aromatic ring.

    Returns
    -------
    (bool, list)
        Whether the atom belongs to an aromatic system and the
        corresponding aromatic-center nodes.
    """

    rings = _aromatic_rings_for_atom(G, atom_idx)

    if rings:
        return True, rings

    # Hydrogen attached to an aromatic atom
    if G.nodes[atom_idx].get("element") == 1:

        for neighbour in _covalent_neighbors(G, atom_idx):

            neighbour_rings = _aromatic_rings_for_atom(G, neighbour)

            if neighbour_rings:
                return True, neighbour_rings

    return False, []


def _nearest_aromatic_ring(G, atom_idx, position):
    """
    Return the nearest aromatic ring associated with atom_idx.
    """
    is_aromatic, rings = _aromatic_membership(
        G,
        atom_idx
    )

    if not is_aromatic:
        return None

    return min(
        rings,
        key=lambda ring_node: norm(
            np.asarray(position)
            - np.asarray(
                G.nodes[ring_node]["position"]
            )
        )
    )


def _bcp_properties(bcp):
    """
    Return common topological properties stored on an NCI edge.
    """
    return {
        "position": np.asarray(
            bcp["position"]
        ).copy(),

        "rho": float(
            bcp["rho"]
        ),

        "laplacian": float(
            bcp["laplacian"]
        ),

        "ellipticity": float(
            bcp["ellipticity"]
        ),

        "eigenvalues": np.asarray(
            bcp["eigenvalues"]
        ).copy(),
    }


# ============================================================
# Aromatic interaction classification
# ============================================================

def _classify_aromatic_nci(
    atom1_idx,
    atom2_idx,
    bcp,
    atoms,
    G,
    angle_threshold=120.0,
    stacking_dist_threshold=10.5,
    offset_threshold=3.8,
):
    """
    Determine whether an NCI BCP corresponds to an interaction
    involving an aromatic pi-system.

    Returns
    -------
    dict or None

        If an aromatic interaction is found:

        {
            "u": graph_node_1,
            "v": graph_node_2,
            "attrs": edge_attributes
        }

        Otherwise None.
    """

    positions = np.asarray([atom[1] for atom in atoms])
    symbols = [get_symbol(atom[0]) for atom in atoms]

    p1 = positions[atom1_idx]
    p2 = positions[atom2_idx]

    aromatic1, rings1 = _aromatic_membership(G, atom1_idx)
    aromatic2, rings2 = _aromatic_membership(G, atom2_idx)

    # --------------------------------------------------------
    # No aromatic system involved
    # --------------------------------------------------------

    if not aromatic1 and not aromatic2:
        return None

    # ========================================================
    # BOTH atoms belong to aromatic systems
    # ========================================================

    if aromatic1 and aromatic2:
        ring1 = _nearest_aromatic_ring(G, atom1_idx, p1)
        ring2 = _nearest_aromatic_ring(G, atom2_idx, p2)

        if ring1 is None or ring2 is None:
            return None

        # Both atoms belong to the same aromatic ring.
        # This is not an intermolecular pi interaction.
        if ring1 == ring2:
            return None

        c1 = np.asarray(G.nodes[ring1]["position"])
        c2 = np.asarray(G.nodes[ring2]["position"])
        diff = c2 - c1

        n1 = np.asarray(G.nodes[ring1]["normal"])
        n2 = np.asarray(G.nodes[ring2]["normal"])

        centroid_distance = np.linalg.norm(diff)

        if centroid_distance > stacking_dist_threshold:
            return None

        # Angle between the normals of the two aromatic planes
        ring_angle = np.degrees(
            np.arccos(
                np.clip(
                    abs(np.dot(n1, n2)),
                    0.0,
                    1.0
                )
            )
        )

        # ----------------------------------------------------
        # Parallel pi-pi stacking
        # ----------------------------------------------------
        if ring_angle <= 30.0:

            offset = np.linalg.norm(diff - np.dot(diff, n1) * n1)

            if offset > offset_threshold:
                return None

            attrs = {
                "bond_type": "stacking",
                "ring_1": ring1,
                "ring_2": ring2,
                "atom_num_cycle_1": int(np.min(G.nodes[ring1]["cycle"])),
                "atom_num_cycle_2": int(np.min(G.nodes[ring2]["cycle"])),
                "centroid_distance": float(centroid_distance),
                "offset": float(offset),
                "angle": float(ring_angle),
            }

            attrs.update(_bcp_properties(bcp))

            return {
                "u": ring1,
                "v": ring2,
                "attrs": attrs
            }

        # ----------------------------------------------------
        # T-stacking
        # ----------------------------------------------------

        elif ring_angle >= 80.0:

            attrs = {
                "bond_type": "T-stacking",
                "ring_1": ring1,
                "ring_2": ring2,
                "atom_num_cycle_1": int(np.min(G.nodes[ring1]["cycle"])),
                "atom_num_cycle_2": int(np.min(G.nodes[ring2]["cycle"])),
                "centroid_distance": float(centroid_distance),
                "angle": float(ring_angle),
            }

            attrs.update(_bcp_properties(bcp))

            return {
                "u": ring1,
                "v": ring2,
                "attrs": attrs
            }

        return None

    # ========================================================
    # ONLY ONE atom belongs to an aromatic system
    # ========================================================

    if aromatic1:
        aromatic_idx = atom1_idx
        other_idx = atom2_idx
        rings = rings1

    else:
        aromatic_idx = atom2_idx
        other_idx = atom1_idx
        rings = rings2

    aromatic_position = positions[aromatic_idx]

    # If an atom belongs to several rings, select the closest one.
    ring = min(
        rings,
        key=lambda ring_node: norm(
            aromatic_position
            - G.nodes[ring_node]["position"]
        )
    )

    centre = np.asarray(G.nodes[ring]["position"])
    normal = np.asarray(G.nodes[ring]["normal"])
    radius = float(G.nodes[ring]["radius"])

    other_symbol = symbols[other_idx]

    # ========================================================
    # X-H ... pi
    # ========================================================

    if other_symbol == "H":

        h_idx = other_idx

        h_neighbours = _covalent_neighbors(
            G,
            h_idx
        )

        if len(h_neighbours) != 1:
            return None

        x_idx = h_neighbours[0]
        x_symbol = symbols[x_idx]

        if x_symbol not in (
            "C",
            "N",
            "O",
            "S"
        ):
            return None

        H = positions[h_idx]
        X = positions[x_idx]

        cutoff = (6.62 if x_symbol in ("C", "S") else 5.7)

        h_centroid_distance = norm(H - centre)

        if h_centroid_distance > cutoff:
            return None

        # X-H-pi angle
        angle = _angle(X, H, centre)

        if angle is None:
            return None

        if angle < angle_threshold:
            return None

        # Projection of H onto the aromatic plane
        vec = H - centre

        projection = (vec - np.dot(vec, normal) * normal)
        projection_distance = norm(projection)

        if projection_distance > radius + 0.2:
            return None

        attrs = {
            "bond_type": "H-pi",
            "x_num": int(x_idx),
            "h_num": int(h_idx),
            "symbol_x": x_symbol,
            "ring": ring,
            "atom_num_cycle": int(np.min(G.nodes[ring]["cycle"])),
            "h_centroid_distance": float(h_centroid_distance),
            "projection_distance": float(projection_distance),
            "angle": float(angle),
        }

        attrs.update(_bcp_properties(bcp))

        return {
            "u": h_idx,
            "v": ring,
            "attrs": attrs
        }

    # ========================================================
    # Lone pair ... pi
    # ========================================================

    lp_atoms = {
        "O",
        "N",
        "S",
        "Se"
    }

    if other_symbol not in lp_atoms:
        return None

    lp_idx = other_idx
    donor = positions[lp_idx]

    neighbours = _covalent_neighbors(
        G,
        lp_idx
    )

    if len(neighbours) == 0:
        return None

    # Simple lone-pair filter
    if other_symbol == "O" and len(neighbours) > 2:
        return None

    if other_symbol == "N" and len(neighbours) > 3:
        return None

    if other_symbol in ("S", "Se") and len(neighbours) > 4:
        return None

    # Distance from lone-pair atom to ring centre
    lp_centroid_distance = norm(donor - centre)

    if other_symbol in ("S", "Se"):

        if not (4.7 <= lp_centroid_distance <= 7.6):
            return None

    else:

        if not (4.7 <= lp_centroid_distance <= 6.8):
            return None

    # --------------------------------------------------------
    # Angle between donor-ring-centre vector and ring normal
    # --------------------------------------------------------

    vec = donor - centre
    vec_norm = np.linalg.norm(vec)

    if vec_norm == 0.0:
        return None

    angle_to_normal = np.degrees(
        np.arccos(
            np.clip(
                abs(np.dot(vec, normal)) / vec_norm,
                0.0,
                1.0
            )
        )
    )

    # --------------------------------------------------------
    # centre-ring / lone-pair atom / covalent neighbour angle
    #
    # IMPORTANT:
    #
    # angle = angle(CENTRE - LP - NEIGHBOUR)
    #
    # No abs() and no vector inversion are used here.
    # Therefore the angle is genuinely in [0, 180] degrees.
    # --------------------------------------------------------

    best_angle = None
    best_neighbour = None

    for neighbour_idx in neighbours:

        neighbour = positions[neighbour_idx]
        angle = 180 - _angle(centre, donor, neighbour)

        if angle is None:
            continue

        if (100.0 <= angle <= 130.0) & (0.0 <= angle_to_normal <= 45.0):

            best_angle = angle
            best_neighbour = neighbour_idx
            continue

    if best_angle is None:
        return None

    attrs = {
        "bond_type": "n-pi",
        "lp_num": int(lp_idx),
        "symbol_lp": other_symbol,
        "ring": ring,
        "atom_num_cycle": int(
            np.min(
                G.nodes[ring]["cycle"]
            )
        ),

        "lp_centroid_distance": float(
            lp_centroid_distance
        ),

        "angle_to_normal": float(
            angle_to_normal
        ),

        "angle": float(best_angle),
        "lp_neighbour_num": int(
            best_neighbour
        ),
    }

    attrs.update(
        _bcp_properties(bcp)
    )

    return {
        "u": lp_idx,
        "v": ring,
        "attrs": attrs
    }


# ============================================================
# Main NCI graph construction
# ============================================================
def build_covalent_graph(
    bcps_covalent,
    atoms,
    max_cycle_size: int = 8,
    planarity_tol: float = 0.10
) -> nx.Graph:
    """
    Build a molecular NetworkX graph containing atoms, covalent
    bonds and aromatic ring centres.

    The function is self-contained: if a covalent BCP does not
    already contain 'atom1' and 'atom2', the two nearest atoms
    to the BCP are determined automatically.

    Parameters
    ----------
    bcps_covalent : list of dict
        Covalent bond critical points. Each BCP must contain at least
        'position', 'rho', 'laplacian', 'eigenvalues' and 'ellipticity'.

        If 'atom1' and 'atom2' are already present, they are used.
        Otherwise they are determined from the two nearest atoms.

    atoms : list of tuples
        Molecular atoms in the format:

            [(Z, position), ...]

        where Z is the atomic number and position is a Cartesian
        coordinate array.

    max_cycle_size : int, default=8
        Maximum ring size considered when searching for cycles.

    planarity_tol : float, default=0.10
        Maximum RMS deviation from the fitted ring plane.

    Returns
    -------
    nx.Graph
        Graph containing:

        - atom nodes with node_type='atom'
        - covalent edges with bond_type='covalent'
        - aromatic-center nodes with node_type='aromatic_center'
        - aromatic edges with bond_type='aromatic'
    """

    # =========================================================
    # Atomic coordinates
    # =========================================================

    coordinates = np.asarray([
        atom[1]
        for atom in atoms
    ])

    # KD-tree is used only if BCPs do not already contain
    # atom1 / atom2.
    atom_tree = cKDTree(coordinates)

    # =========================================================
    # Create graph
    # =========================================================

    G = nx.Graph()

    # Add all atom nodes explicitly.
    for i, (Z, position) in enumerate(atoms):

        G.add_node(
            i,
            node_type="atom",
            element=int(Z),
            position=np.asarray(position).copy()
        )

    # =========================================================
    # Add covalent bonds
    # =========================================================

    for bcp in bcps_covalent:

        # -----------------------------------------------------
        # Determine the two atoms belonging to the BCP
        # -----------------------------------------------------

        if (
            "atom1" in bcp
            and
            "atom2" in bcp
        ):

            atom1 = int(bcp["atom1"])
            atom2 = int(bcp["atom2"])

        else:

            position = np.asarray(
                bcp["position"]
            )

            distances, indices = atom_tree.query(
                position,
                k=2
            )

            atom1 = int(indices[0])
            atom2 = int(indices[1])

        # -----------------------------------------------------
        # Safety checks
        # -----------------------------------------------------

        if atom1 == atom2:
            continue

        if not (
            0 <= atom1 < len(atoms)
            and
            0 <= atom2 < len(atoms)
        ):
            continue

        # -----------------------------------------------------
        # Add covalent edge
        # -----------------------------------------------------

        G.add_edge(
            atom1,
            atom2,

            bond_type="covalent",

            position=np.asarray(
                bcp["position"]
            ).copy(),

            rho=float(
                bcp["rho"]
            ),

            laplacian=float(
                bcp["laplacian"]
            ),

            eigenvalues=np.asarray(
                bcp["eigenvalues"]
            ).copy(),

            ellipticity=float(
                bcp["ellipticity"]
            )
        )

    # =========================================================
    # Find cycles
    # =========================================================

    cycles = nx.cycle_basis(G)

    aromatic_index = 0

    for cycle in cycles:

        ring_size = len(cycle)

        # -----------------------------------------------------
        # Ring-size filter
        # -----------------------------------------------------

        if (
            ring_size < 5
            or
            ring_size > max_cycle_size
        ):
            continue

        # -----------------------------------------------------
        # Ring coordinates
        # -----------------------------------------------------

        coords = coordinates[
            np.asarray(
                cycle,
                dtype=int
            )
        ]

        centre = coords.mean(
            axis=0
        )

        # -----------------------------------------------------
        # Least-squares plane through ring atoms
        # -----------------------------------------------------

        centered = coords - centre

        try:

            _, _, vh = np.linalg.svd(
                centered,
                full_matrices=False
            )

        except np.linalg.LinAlgError:

            continue

        # Normal to the best-fit plane
        normal = vh[-1]

        normal_norm = norm(
            normal
        )

        if normal_norm == 0.0:
            continue

        normal = normal / normal_norm

        # -----------------------------------------------------
        # Deviation of atoms from the plane
        # -----------------------------------------------------

        distances = centered @ normal

        rms_planarity = np.sqrt(
            np.mean(
                distances ** 2
            )
        )

        max_planarity_deviation = np.max(
            np.abs(distances)
        )

        # -----------------------------------------------------
        # Planarity filter
        # -----------------------------------------------------

        if rms_planarity > planarity_tol:
            continue

        # -----------------------------------------------------
        # Ring radius
        # -----------------------------------------------------

        radius = np.mean(
            np.linalg.norm(
                coords - centre,
                axis=1
            )
        )

        # -----------------------------------------------------
        # Plane equation:
        #
        # n · r + d = 0
        # -----------------------------------------------------

        plane_d = -np.dot(
            normal,
            centre
        )

        # -----------------------------------------------------
        # Store cycle in deterministic form
        # -----------------------------------------------------

        cycle_tuple = tuple(
            sorted(
                int(i)
                for i in cycle
            )
        )

        ring_node = (
            f"ring_{aromatic_index}"
        )

        aromatic_index += 1

        # =====================================================
        # Add aromatic-center node
        # =====================================================

        G.add_node(
            ring_node,

            node_type="aromatic_center",

            position=np.asarray(
                centre
            ).copy(),

            normal=np.asarray(
                normal
            ).copy(),

            plane_d=float(
                plane_d
            ),

            radius=float(
                radius
            ),

            cycle=cycle_tuple,

            size=int(
                ring_size
            ),

            aromatic=True,

            rms_planarity=float(
                rms_planarity
            ),

            max_planarity_deviation=float(
                max_planarity_deviation
            )
        )

        # =====================================================
        # Connect ring centre to every ring atom
        # =====================================================

        for atom_idx in cycle:

            G.add_edge(
                ring_node,
                int(atom_idx),

                bond_type="aromatic",

                ring=cycle_tuple
            )

    return G


def build_graph_with_nci(
    bcps,
    G,
    atoms,
    angle_threshold=120.0,
    stacking_dist_threshold=10.4,
    offset_threshold=3.8,
    bcp_angle_threshold=150.0,
    search_radius = 3.8,
):
    """
    Add non-covalent interactions to an existing molecular graph.

    Each BCP is assigned to the two closest atoms. The BCP is then
    classified as:

        HB       hydrogen bond
        XB       halogen bond
        ChB      chalcogen bond
        PnB      pnictogen bond
        H-pi     X-H ... pi interaction
        n-pi     lone-pair ... pi interaction
        stacking parallel pi-pi stacking
        T-stacking perpendicular / T-shaped pi interaction

    Aromatic interactions are checked before ordinary HB/XB/ChB/PnB
    classification.

    Parameters
    ----------
    bcps : list of dict
        NCI bond critical points.

    G : nx.Graph
        Existing molecular graph containing atom nodes, covalent
        bonds and aromatic-center nodes.

    atoms : list of (Z, position)
        Atomic numbers and Cartesian coordinates.

    angle_threshold : float
        Minimum X-H-pi angle.

    stacking_dist_threshold : float
        Maximum distance between aromatic ring centroids.

    offset_threshold : float
        Maximum lateral offset for parallel pi-pi stacking.

    Returns
    -------
    nx.Graph
        Updated molecular graph.
    """

    if not bcps:
        return G

    # --------------------------------------------------------
    # Atomic coordinates
    # --------------------------------------------------------

    coordinates = np.asarray([atom[1] for atom in atoms])
    symbols = [get_symbol(atom[0]) for atom in atoms]

    # Build once; do not construct a KD-tree for every BCP.
    atom_tree = cKDTree(coordinates)

    # ========================================================
    # Process BCPs
    # ========================================================

    for bcp in tqdm(bcps, desc="Building graph from NCI BCPs"):

        position = np.asarray(bcp["position"])

        # ----------------------------------------------------
        # Find all atoms within search_radius from the BCP
        # ----------------------------------------------------

        indices_in_radius = atom_tree.query_ball_point(position, search_radius)

        if len(indices_in_radius) < 2:
            continue

        # ----------------------------------------------------
        # Find the best pair: angle >= threshold and minimal sum of distances
        # ----------------------------------------------------

        best_pair = None
        best_sum_dist = float('inf')

        for i in range(len(indices_in_radius)):
            idx1 = indices_in_radius[i]
            for idx2 in indices_in_radius[i+1:]:
                if idx1 == idx2:
                    continue

                angle = _angle(coordinates[idx1], position, coordinates[idx2])
                if angle is None or angle < bcp_angle_threshold:
                    continue

                d1 = norm(coordinates[idx1] - position)
                d2 = norm(coordinates[idx2] - position)
                sum_dist = d1 + d2

                if sum_dist < best_sum_dist:
                    best_sum_dist = sum_dist
                    best_pair = (idx1, idx2)

        if best_pair is None:
            continue

        atom1_idx, atom2_idx = best_pair

        if G.has_edge(atom1_idx, atom2_idx):
            continue

        symbol1 = symbols[atom1_idx]
        symbol2 = symbols[atom2_idx]

        # ====================================================
        # 1. Aromatic interactions
       # ====================================================

        aromatic_result = _classify_aromatic_nci(
            atom1_idx=atom1_idx,
            atom2_idx=atom2_idx,
            bcp=bcp,
            atoms=atoms,
            G=G,
            angle_threshold=angle_threshold,
            stacking_dist_threshold=stacking_dist_threshold,
            offset_threshold=offset_threshold,
        )

        if aromatic_result is not None:

            G.add_edge(
                aromatic_result["u"],
                aromatic_result["v"],
                **aromatic_result["attrs"]
            )

            continue

        # ====================================================
        # 2. Hydrogen bonds
        # ====================================================

        is_h1 = symbol1 == "H"
        is_h2 = symbol2 == "H"

        # H ... H is ignored
        if is_h1 and is_h2:
            continue

        if is_h1 or is_h2:

            if is_h1:
                h_idx = atom1_idx
                acceptor_idx = atom2_idx
            else:
                h_idx = atom2_idx
                acceptor_idx = atom1_idx

            # ------------------------------------------------
            # Find covalent donor of H
            # ------------------------------------------------

            donor_neighbours = _covalent_neighbors(G, h_idx)

            if len(donor_neighbours) != 1:
                continue

            donor_idx = donor_neighbours[0]
            donor_symbol = symbols[donor_idx]
            acceptor_symbol = symbols[acceptor_idx]

            D = coordinates[donor_idx]
            H = coordinates[h_idx]
            A = coordinates[acceptor_idx]

            # H ... D distance
            h_donor_distance = norm(H - D)

            # H ... A distance
            h_acceptor_distance = norm(H - A)

            # D ... A distance
            heavy_atom_distance = norm(D - A)

            # D-H ... A angle
            hb_angle = _angle(D, H, A)

            if hb_angle is None:
                continue

            attrs = {
                "bond_type": "HB",
                "donor_num": int(donor_idx),
                "hydrogen_num": int(h_idx),
                "acceptor_num": int(acceptor_idx),
                "donor_symbol": donor_symbol,
                "acceptor_symbol":acceptor_symbol,
                "distance": float(h_acceptor_distance),
                "heavy_atom_distance": float(heavy_atom_distance),
                "angle": float(hb_angle),
                "h_donor_distance": float(h_donor_distance)
            }

            attrs.update(_bcp_properties(bcp))

            G.add_edge(h_idx, acceptor_idx, **attrs)

            continue

        # ====================================================
        # 3. Heavy-atom interactions:
        # XB / ChB / PnB
        # ====================================================

        distance1 = norm(coordinates[atom1_idx] - position) / BONDI[symbols[atom1_idx]]
        distance2 = norm(coordinates[atom2_idx] - position) / BONDI[symbols[atom2_idx]]

        # Atom closest to BCP = electrophilic atom
        if distance1 <= distance2:

            electrophile_idx = atom1_idx
            nucleophile_idx = atom2_idx

        else:

            electrophile_idx = atom2_idx
            nucleophile_idx = atom1_idx

        electrophile_symbol = symbols[electrophile_idx]

        # ----------------------------------------------------
        # Determine interaction type
        # ----------------------------------------------------

        if electrophile_symbol in {"F", "Cl", "Br", "I"}:
            bond_type = "XB"

        elif electrophile_symbol in {"O", "S", "Se", "Te"}:
            bond_type = "ChB"

        elif electrophile_symbol in {"N", "P", "As", "Sb", "Bi"}:
            bond_type = "PnB"

        else:
            # Not a recognised sigma-hole interaction.
            continue

        # ----------------------------------------------------
        # Distance between interacting atoms
        # ----------------------------------------------------

        interaction_distance = norm(coordinates[atom1_idx] - coordinates[atom2_idx])

        # ----------------------------------------------------
        # Electrophile ... nucleophile angle
        #
        # For sigma-hole interactions, the angle is defined
        # at the electrophilic atom:
        #
        # covalent neighbour - electrophile - nucleophile
        #
        # If the electrophile has multiple covalent neighbours,
        # select the angle closest to linearity.
        # ----------------------------------------------------

        electrophile_neighbours = _covalent_neighbors(G, electrophile_idx)
        contact_angle = None

        if electrophile_neighbours:

            E = coordinates[electrophile_idx]
            N = coordinates[nucleophile_idx]
            candidate_angles = []
            neighbor_nums = []

            for neighbour_idx in electrophile_neighbours:

                neighbour = coordinates[neighbour_idx]
                angle = _angle(neighbour, E, N)
                neighbor_nums.append(neighbour_idx)

                if angle is not None:
                    candidate_angles.append(angle)

            if candidate_angles:
                best_idx = min(range(len(candidate_angles)), key=lambda i: abs(180.0 - candidate_angles[i]))
                contact_angle = candidate_angles[best_idx]
                neighbor_num = neighbor_nums[best_idx]

        attrs = {
            "bond_type": bond_type,
            "neighbor_num": int(neighbor_num),
            "electrophile_num": int(electrophile_idx),
            "nucleophile_num": int(nucleophile_idx),
            "neighbor_symbol": symbols[neighbor_num],
            "electrophile_symbol": electrophile_symbol,
            "nucleophile_symbol": symbols[nucleophile_idx],
            "distance": float(interaction_distance),
            "angle": (None if contact_angle is None else float(contact_angle)),
        }

        attrs.update(_bcp_properties(bcp))

        G.add_edge(
            atom1_idx,
            atom2_idx,
            **attrs
        )

    return G

def build_graph_with_carbonyl(
    bcps,
    G,
    atoms,
    distance_cutoff=6.1,
):
    """
    Find carbonyl-type n -> pi* interactions among NCI BCPs.

    A BCP is classified as a carbonyl interaction if:

        1. One of the two atoms nearest to the BCP is oxygen.
        2. The other atom is carbon.
        3. This carbon has exactly three covalently bonded
           neighbours in graph G.
        4. Exactly one of these covalent neighbours is oxygen.
           Such a carbon is treated as a carbonyl carbon.
        5. The O-C pair is not itself a covalent bond.

    The oxygen is treated as the electron-density donor. It does
    not have to be a carbonyl oxygen: it may belong to a carbonyl,
    hydroxyl, ether, etc.

    Parameters
    ----------
    bcps : list of dict
        Non-covalent BCPs.

    G : nx.Graph
        Molecular graph containing atoms and covalent bonds.

    atoms : list of (Z, position)
        Atomic numbers and Cartesian coordinates.

    distance_cutoff : float or None
        Optional maximum O...C distance in bohr. If None, no
        additional distance criterion is applied.

    Returns
    -------
    nx.Graph
        Graph with carbonyl interaction edges added.
    """

    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms])
    symbols = [get_symbol(atom[0]) for atom in atoms]
    atom_tree = cKDTree(coordinates)

    # =========================================================
    # Determine carbonyl carbon atoms from the covalent graph
    # =========================================================

    carbonyl_carbons = set()

    for atom_idx, symbol in enumerate(symbols):

        if symbol != "C":
            continue

        covalent_neighbours = _covalent_neighbors(G, atom_idx)

        # Carbonyl carbon must have exactly three covalent
        # neighbours.
        if len(covalent_neighbours) != 3:
            continue

        # Exactly one of them must be oxygen.
        oxygen_neighbours = [neighbour for neighbour in covalent_neighbours if symbols[neighbour] == "O"]

        if len(oxygen_neighbours) < 1:
            continue

        carbonyl_carbons.add(atom_idx)

    # =========================================================
    # Process NCI BCPs
    # =========================================================

    for bcp in tqdm(bcps, desc="Searching carbonyl interactions"):

        position = np.asarray(bcp["position"])

        # -----------------------------------------------------
        # Two atoms nearest to the BCP
        # -----------------------------------------------------

        distances, indices = atom_tree.query(position, k=2)

        atom1_idx = int(indices[0])
        atom2_idx = int(indices[1])

        if atom1_idx == atom2_idx:
            continue

        symbol1 = symbols[atom1_idx]
        symbol2 = symbols[atom2_idx]

        # -----------------------------------------------------
        # One atom must be O, the other carbonyl C
        # -----------------------------------------------------

        if symbol1 == "O" and atom2_idx in carbonyl_carbons:

            donor_oxygen = atom1_idx
            acceptor_carbon = atom2_idx

        elif symbol2 == "O" and atom1_idx in carbonyl_carbons:

            donor_oxygen = atom2_idx
            acceptor_carbon = atom1_idx

        else:
            continue

        # -----------------------------------------------------
        # Do not classify an ordinary covalent C-O bond
        # as a non-covalent carbonyl interaction.
        # -----------------------------------------------------

        if G.has_edge(donor_oxygen, acceptor_carbon) and G.edges[donor_oxygen, acceptor_carbon].get("bond_type") == "covalent":
            continue

        # -----------------------------------------------------
        # O...C distance
        # -----------------------------------------------------

        interaction_distance = norm(coordinates[donor_oxygen] - coordinates[acceptor_carbon])

        if (distance_cutoff is not None and interaction_distance > distance_cutoff):
            continue

        # =====================================================
        # Carbonyl geometry
        # =====================================================

        C = coordinates[acceptor_carbon]
        O = coordinates[donor_oxygen]

        # -----------------------------------------------------
        # Burgi-Dunitz-type angle
        #
        # For a carbonyl carbon, find the covalent neighbour
        # direction closest to the incoming O...C vector.
        # -----------------------------------------------------

        carbonyl_neighbours = _covalent_neighbors(G, acceptor_carbon)
        candidate_angles = []

        for neighbour_idx in carbonyl_neighbours:

            X = coordinates[neighbour_idx]

            angle = _angle(X, C, O)

            if angle is not None:
                candidate_angles.append((abs(180.0 - angle), angle, neighbour_idx))

        if candidate_angles:
            _, burgi_dunitz_angle, neighbour_idx = min(candidate_angles, key=lambda x: x[0])

        else:

            burgi_dunitz_angle = None
            neighbour_idx = None

        plane_angle = None

        if len(carbonyl_neighbours) == 3:

            neighbour_coords = np.asarray([coordinates[idx] for idx in carbonyl_neighbours])

            plane_centre = neighbour_coords.mean(axis=0)
            centered = (neighbour_coords - plane_centre)

            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normal = vh[-1]
                normal_norm = norm(normal)

                if normal_norm > 0.0:

                    normal = (normal / normal_norm)

                    CO_vector = O - C
                    CO_norm = norm(CO_vector)

                    if CO_norm > 0.0:

                        angle_to_normal = np.degrees(
                            np.arccos(
                                np.clip(
                                    abs(
                                        np.dot(
                                            CO_vector,
                                            normal
                                        )
                                    )
                                    / CO_norm,
                                    0.0,
                                    1.0
                                )
                            )
                        )

                        # Convert angle to the angle between
                        # the O...C vector and the carbonyl plane.
                        plane_angle = (
                            90.0
                            - angle_to_normal
                        )

            except np.linalg.LinAlgError:
                plane_angle = None

        # =====================================================
        # Edge attributes
        # =====================================================

        attrs = {
            "bond_type": "carbonyl",
            "donor_oxygen": int(donor_oxygen),
            "acceptor_carbon": int(acceptor_carbon),
            "donor_symbol": symbols[donor_oxygen],
            "acceptor_symbol": symbols[acceptor_carbon],
            "distance": float(interaction_distance),
            "burgi_dunitz_angle": (None if burgi_dunitz_angle is None else float(burgi_dunitz_angle)),
            "plane_angle": (None if plane_angle is None else float(plane_angle)),
            "carbonyl_oxygen": int(_covalent_neighbors(G, acceptor_carbon)[0]
                if (
                    len(_covalent_neighbors(G, acceptor_carbon)) == 1
            )
                else next(
                    neighbour for neighbour in _covalent_neighbors(G, acceptor_carbon) if symbols[neighbour] == "O"
                )
            ),
        }

        attrs.update(
            _bcp_properties(bcp)
        )

        # -----------------------------------------------------
        # Add the interaction edge
        # -----------------------------------------------------

        G.add_edge(
            donor_oxygen,
            acceptor_carbon,
            **attrs
        )

    return G

def build_graph_with_unclassified(
    bcps,
    G,
    atoms,
    position_tolerance=0.1,
    search_radius=3.8,
    bcp_angle_threshold=150,
):
    """
    Add all remaining NCI BCPs that were not assigned to a
    classified interaction as 'unclassified' graph edges.

    A BCP is considered already classified if an edge in G
    contains an interaction BCP at essentially the same spatial
    position. The comparison is performed using position_tolerance
    in bohr.

    This function deliberately does not impose additional
    chemical criteria: the purpose is to retain information about
    potentially relevant non-covalent contacts that could not be
    assigned to HB, XB, ChB, PnB, pi interactions, carbonyl
    interactions, etc.

    Additionally, two types of BCPs are skipped:
      1. If one atom belongs to an aromatic ring and the other atom
         is also associated with the same aromatic ring (e.g., as a
         substituent or another ring atom). This is detected by
         checking if the sets of aromatic centers (from
         _aromatic_membership) intersect.
      2. If both atoms belong to aromatic rings and there already
         exists a "stacking" or "T-stacking" edge between the
         corresponding aromatic centers.

    Parameters
    ----------
    bcps : list of dict
        Original NCI BCPs satisfying the rho criterion.

    G : nx.Graph
        Molecular graph after all classified NCI interactions
        have been added.

    atoms : list of (Z, position)
        Atomic numbers and Cartesian coordinates.

    position_tolerance : float
        Maximum distance between two BCP positions for them to
        be considered the same BCP. Units: bohr.

    search_radius : float
        Radius around the BCP to search for candidate atoms.

    bcp_angle_threshold : float
        Minimum angle (degrees) between the two atoms as seen
        from the BCP. Pairs with angle below this value are rejected.

    Returns
    -------
    nx.Graph
        Graph with additional 'unclassified' edges.
    """

    if not bcps:
        return G

    coordinates = np.asarray([atom[1] for atom in atoms])
    symbols = [get_symbol(atom[0]) for atom in atoms]
    atom_tree = cKDTree(coordinates)

    # =========================================================
    # Collect positions of BCPs already represented in the graph
    # =========================================================

    classified_positions = []

    classified_types = {"covalent", "aromatic",}

    for _, _, data in G.edges(data=True):
        bond_type = data.get("bond_type")

        if bond_type in classified_types:
            continue

        position = data.get("position")

        if position is None:
            continue

        classified_positions.append(np.asarray(position))

    # KD-tree makes repeated BCP-position comparisons inexpensive.
    if classified_positions:
        classified_tree = cKDTree(np.asarray(classified_positions))

    else:
        classified_tree = None

    # =========================================================
    # Process all NCI BCPs
    # =========================================================

    for bcp in tqdm(bcps, desc="Searching unclassified NCI contacts"):

        position = np.asarray(bcp["position"])

        # ----------------------------------------------------
        # Skip if this BCP is already classified
        # ----------------------------------------------------

        if classified_tree is not None:
            distance, _ = classified_tree.query(position, k=1)
            if distance <= position_tolerance:
                continue

        # ----------------------------------------------------
        # Find all atoms within search_radius from the BCP
        # ----------------------------------------------------

        indices_in_radius = atom_tree.query_ball_point(position, search_radius)

        if len(indices_in_radius) < 2:
            continue

        # ----------------------------------------------------
        # Find the best pair: angle >= threshold and minimal sum of distances
        # ----------------------------------------------------

        best_pair = None
        best_sum_dist = float('inf')

        for i in range(len(indices_in_radius)):
            idx1 = indices_in_radius[i]
            for idx2 in indices_in_radius[i+1:]:
                if idx1 == idx2:
                    continue

                angle = _angle(coordinates[idx1], position, coordinates[idx2])
                if angle is None or angle < bcp_angle_threshold:
                    continue

                d1 = norm(coordinates[idx1] - position)
                d2 = norm(coordinates[idx2] - position)
                sum_dist = d1 + d2

                if sum_dist < best_sum_dist:
                    best_sum_dist = sum_dist
                    best_pair = (idx1, idx2)

        if best_pair is None:
            continue

        atom1_idx, atom2_idx = np.min(best_pair), np.max(best_pair)

        # ----------------------------------------------------
        # Skip if this edge already exists in the graph
        # ----------------------------------------------------

        if G.has_edge(atom1_idx, atom2_idx):
            continue

        # ----------------------------------------------------
        # Condition 1: Both atoms are associated with the same aromatic ring
        # ----------------------------------------------------
        skip = False

        aromatic1, rings1 = _aromatic_membership(G, atom1_idx)
        aromatic2, rings2 = _aromatic_membership(G, atom2_idx)

        # If they share any common aromatic center, skip
        if aromatic1:
            for ring in rings1:
                # Check if atom2 has an edge to this ring of type H-pi or n-pi
                if G.has_edge(atom2_idx, ring):
                    bond_type = G.edges[atom2_idx, ring].get('bond_type')
                    if bond_type in ('H-pi', 'n-pi'):
                        skip = True
                        break

        if not skip and aromatic2:
            for ring in rings2:
                if G.has_edge(atom1_idx, ring):
                    bond_type = G.edges[atom1_idx, ring].get('bond_type')
                    if bond_type in ('H-pi', 'n-pi'):
                        skip = True
                        break

        # ----------------------------------------------------
        # Condition 2: If both are aromatic and there is already
        # a stacking/T-stacking edge between their rings
        # ----------------------------------------------------

        if aromatic1 and aromatic2:
            # Check all pairs of rings
            skip_due_to_stacking = False
            for ring1 in rings1:
                for ring2 in rings2:
                    # Check if there is an edge between these two ring nodes
                    if G.has_edge(ring1, ring2):
                        edge_data = G.edges[ring1, ring2]
                        bond_type = edge_data.get("bond_type")
                        if bond_type in {"stacking", "T-stacking"}:
                            skip_due_to_stacking = True
                            break
                if skip_due_to_stacking:
                    break
            if skip_due_to_stacking:
                continue

        # -----------------------------------------------------
        # Atom-atom distance
        # -----------------------------------------------------

        interaction_distance = norm(coordinates[atom1_idx] - coordinates[atom2_idx])

        attrs = {
            "bond_type": "unclassified",
            "atom1_num": int(atom1_idx),
            "atom2_num": int(atom2_idx),
            "atom1_symbol": symbols[atom1_idx],
            "atom2_symbol": symbols[atom2_idx],
            "distance": float(interaction_distance),
        }

        attrs.update(_bcp_properties(bcp))

        G.add_edge(
            atom1_idx,
            atom2_idx,
            **attrs
        )

        # -----------------------------------------------------
        # Immediately register this BCP as classified.
        # -----------------------------------------------------

        classified_positions.append(position.copy())

        # Rebuild KD-tree after adding each new position
        classified_tree = cKDTree(np.asarray(classified_positions))

    return G

# --------------------------------------------------------------------------- #
# Output writer
# --------------------------------------------------------------------------- #

def output(
    G: nx.Graph,
    filename: str = "default",
    ext: str = ".default",
    correlation=correlation_1,
) -> None:
    """
    Write a formatted .nci file summarising all detected interactions.

    All geometrical quantities stored in the graph are assumed to be
    expressed in bohr. They are converted to Å only when writing the
    output file.

    Non-geometrical quantities such as rho, Laplacian and ellipticity
    are written without conversion.

    Parameters
    ----------
    G : nx.Graph
        Molecular graph containing covalent and non-covalent interactions.

    filename : str
        Output filename without extension.

    ext : str
        Extension of the original input file.

    correlation : callable
        Energy-estimation function for hydrogen bonds.
    """

    now = datetime.now()

    BOHR_TO_ANGSTROM = bohr_to_angstrom

    # ===================================================================== #
    # Group graph edges by interaction type
    # ===================================================================== #

    edges_by_type = {}

    for _, _, data in G.edges(data=True):

        bond_type = data.get(
            "bond_type"
        )

        if bond_type is None:
            continue

        edges_by_type.setdefault(
            bond_type,
            []
        ).append(data)

    # ===================================================================== #
    # Helper functions
    # ===================================================================== #

    def atom_number(data, key):
        """
        Return atom number in human-readable 1-based notation.
        """

        value = data.get(
            key
        )

        if value is None:
            return "—"

        return int(value) + 1

    def fmt_float(data, key, digits=3):
        """
        Format a floating-point edge attribute without unit conversion.
        """

        value = data.get(
            key
        )

        if value is None:
            return "—"

        return round(
            float(value),
            digits
        )

    def fmt_distance(data, key, digits=3):
        """
        Retrieve a distance stored in bohr and convert it to Å.
        """

        value = data.get(
            key
        )

        if value is None:
            return "—"

        return round(
            float(value)
            * BOHR_TO_ANGSTROM,
            digits
        )

    def fmt_rho(data):
        """
        Electron density at the BCP.
        """

        return fmt_float(
            data,
            "rho",
            5
        )

    def fmt_laplacian(data):
        """
        Laplacian of electron density at the BCP.
        """

        return fmt_float(
            data,
            "laplacian",
            5
        )

    def fmt_ellipticity(data):
        """
        Ellipticity of the electron density.
        """

        return fmt_float(
            data,
            "ellipticity",
            3
        )

    # ===================================================================== #
    # Section configuration
    # ===================================================================== #

    sections = [

        # ----------------------------------------------------------------- #
        # Hydrogen bonds
        # ----------------------------------------------------------------- #

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
                "ρ, a.u.",
                "∇²ρ, a.u.",
                "ε",
                "Energy (kcal/mol)",
            ],

            "row_func": lambda d, i: [
                i + 1,

                f"{d['donor_symbol']}-H···"
                f"{d['acceptor_symbol']}",

                atom_number(
                    d,
                    "donor_num"
                ),

                atom_number(
                    d,
                    "hydrogen_num"
                ),

                atom_number(
                    d,
                    "acceptor_num"
                ),

                fmt_distance(
                    d,
                    "h_donor_distance"
                ),

                fmt_distance(
                    d,
                    "distance"
                ),

                fmt_distance(
                    d,
                    "heavy_atom_distance"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),

                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),

                round(
                    float(
                        correlation(
                            float(d["rho"]),
                            f"{d['donor_symbol']}-H..."
                            f"{d['acceptor_symbol']}",
                        )
                    ),
                    1,
                ),
            ],
        },

        # ----------------------------------------------------------------- #
        # Halogen bonds
        # ----------------------------------------------------------------- #

        {
            "key": "XB",
            "title": "HALOGEN BONDS",

            "header": [
                "№",
                "Type",
                "X",
                "D",
                "A",
                "D···A (Å)",
                "Angle (°)",
                "ρ, a.u.",
                "∇²ρ, a.u.",
                "ε",
            ],

            "row_func": lambda d, i: [
                i + 1,

                f"{d.get('neighbor_symbol', '—')}-"
                f"{d.get('electrophile_symbol', '—')}···"
                f"{d.get('nucleophile_symbol', '—')}",

                atom_number(
                    d,
                    "neighbor_num"
                ),

                atom_number(
                    d,
                    "electrophile_num"
                ),

                atom_number(
                    d,
                    "nucleophile_num"
                ),

                fmt_distance(
                    d,
                    "distance"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),

                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },

        # ----------------------------------------------------------------- #
        # Chalcogen bonds
        # ----------------------------------------------------------------- #

        {
            "key": "ChB",
            "title": "CHALCOGEN BONDS",

            "header": [
                "№",
                "Type",
                "X",
                "D",
                "A",
                "D···A (Å)",
                "Angle (°)",
                "ρ, a.u.",
                "∇²ρ, a.u.",
                "ε",
            ],

            "row_func": lambda d, i: [
                i + 1,

                f"{d.get('neighbor_symbol', '—')}-"
                f"{d.get('electrophile_symbol', '—')}···"
                f"{d.get('nucleophile_symbol', '—')}",

                atom_number(
                    d,
                    "neighbor_num"
                ),

                atom_number(
                    d,
                    "electrophile_num"
                ),

                atom_number(
                    d,
                    "nucleophile_num"
                ),

                fmt_distance(
                    d,
                    "distance"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),

                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },

        # ----------------------------------------------------------------- #
        # Pnictogen bonds
        # ----------------------------------------------------------------- #

        {
            "key": "PnB",
            "title": "PNICTOGEN BONDS",

            "header": [
                "№",
                "Type",
                "X",
                "D",
                "A",
                "D···A (Å)",
                "Angle (°)",
                "ρ, a.u.",
                "∇²ρ, a.u.",
                "ε",
            ],

            "row_func": lambda d, i: [
                i + 1,

                f"{d.get('neighbor_symbol', '—')}-"
                f"{d.get('electrophile_symbol', '—')}···"
                f"{d.get('nucleophile_symbol', '—')}",

                atom_number(
                    d,
                    "neighbor_num"
                ),

                atom_number(
                    d,
                    "electrophile_num"
                ),

                atom_number(
                    d,
                    "nucleophile_num"
                ),

                fmt_distance(
                    d,
                    "distance"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),

                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },

        # ----------------------------------------------------------------- #
        # π-π stacking
        # ----------------------------------------------------------------- #

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

                atom_number(
                    d,
                    "atom_num_cycle_1"
                ),

                atom_number(
                    d,
                    "atom_num_cycle_2"
                ),

                fmt_distance(
                    d,
                    "centroid_distance"
                ),

                fmt_distance(
                    d,
                    "offset"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),
            ],
        },

        # ----------------------------------------------------------------- #
        # T-stacking
        # ----------------------------------------------------------------- #

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

                atom_number(
                    d,
                    "atom_num_cycle_1"
                ),

                atom_number(
                    d,
                    "atom_num_cycle_2"
                ),

                fmt_distance(
                    d,
                    "centroid_distance"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),
            ],
        },

        # ----------------------------------------------------------------- #
        # X-H···π
        # ----------------------------------------------------------------- #

        {
            "key": "H-pi",
            "title": "X-H···π INTERACTIONS",

            "header": [
                "№",
                "Type",
                "X",
                "H",
                "Atom of Ring",
                "H···π (Å)",
                "Angle (°)",
                "Projection (Å)",
            ],

            "row_func": lambda d, i: [
                i + 1,

                f"{d['symbol_x']}-H···π",

                atom_number(
                    d,
                    "x_num"
                ),

                atom_number(
                    d,
                    "h_num"
                ),

                atom_number(
                    d,
                    "atom_num_cycle"
                ),

                fmt_distance(
                    d,
                    "h_centroid_distance"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),

                fmt_distance(
                    d,
                    "projection_distance"
                ),
            ],
        },

        # ----------------------------------------------------------------- #
        # Lone pair···π
        # ----------------------------------------------------------------- #

        {
            "key": "n-pi",
            "title": "LONE PAIR···π",

            "header": [
                "№",
                "Type",
                "Atom",
                "Atom of Ring",
                "Distance (Å)",
                "Angle (°)",
            ],

            "row_func": lambda d, i: [
                i + 1,

                f"{d['symbol_lp']}···π",

                atom_number(
                    d,
                    "lp_num"
                ),

                atom_number(
                    d,
                    "atom_num_cycle"
                ),

                fmt_distance(
                    d,
                    "lp_centroid_distance"
                ),

                fmt_float(
                    d,
                    "angle",
                    1
                ),
            ],
        },

        # ----------------------------------------------------------------- #
        # Carbonyl n→π*
        # ----------------------------------------------------------------- #

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
                "ρ, a.u.",
                "∇²ρ, a.u.",
                "ε",
            ],

            "row_func": lambda d, i: [
                i + 1,

                "n→π*",

                atom_number(
                    d,
                    "donor_oxygen"
                ),

                atom_number(
                    d,
                    "acceptor_carbon"
                ),

                fmt_distance(
                    d,
                    "distance"
                ),

                fmt_float(
                    d,
                    "burgi_dunitz_angle",
                    1
                ),

                (
                    "—"
                    if d.get("plane_angle") is None
                    else round(
                        float(
                            d["plane_angle"]
                        ),
                        1
                    )
                ),

                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },

        # ----------------------------------------------------------------- #
        # Unclassified NCI contacts
        # ----------------------------------------------------------------- #

        {
            "key": "unclassified",
            "title": "UNCLASSIFIED NON-COVALENT CONTACTS",

            "header": [
                "№",
                "Type",
                "Atom 1",
                "Atom 2",
                "Element 1",
                "Element 2",
                "Distance (Å)",
                "ρ, a.u.",
                "∇²ρ, a.u.",
                "ε",
            ],

            "row_func": lambda d, i: [
                i + 1,

                "unclassified",

                atom_number(
                    d,
                    "atom1_num"
                ),

                atom_number(
                    d,
                    "atom2_num"
                ),

                d.get(
                    "atom1_symbol",
                    "—"
                ),

                d.get(
                    "atom2_symbol",
                    "—"
                ),

                fmt_distance(
                    d,
                    "distance"
                ),

                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },
    ]

    # ===================================================================== #
    # Write output file
    # ===================================================================== #

    output_filename = f"{filename}.nci"

    with open(
        output_filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "NCITools: Topological analysis of electron density\n"
            f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Input file: {filename}{ext}\n"
        )

        # ------------------------------------------------------------- #
        # Sections
        # ------------------------------------------------------------- #

        for sec in sections:

            interaction_edges = edges_by_type.get(
                sec["key"],
                []
            )

            if not interaction_edges:
                continue

            rows = [
                sec["row_func"](
                    edge,
                    i
                )
                for i, edge in enumerate(
                    interaction_edges
                )
            ]

            f.write("\n")
            f.write("=" * 120 + "\n")
            f.write(
                sec["title"]
            )
            f.write("\n")
            f.write("=" * 120 + "\n\n")

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

        # ------------------------------------------------------------- #
        # Footer
        # ------------------------------------------------------------- #

        WIDTH = 40

        philosopher, quote = random.choice(
            quotes
        )

        f.write("\n\n")

        f.write(
            textwrap.fill(
                quote.upper(),
                width=WIDTH,
                initial_indent=" ",
                subsequent_indent=" ",
            )
        )

        f.write(
            f"\n   -- {philosopher.upper()}\n"
        )




# ============================================================================
#  Main script
# ============================================================================

def main(
    cube_file,
    output_file=None,
    cutoff=3.0,         # in Ang!
    rho_min=0.001,
    rho_max=0.089,
    nproc=4,
):
    """
    Run the complete topological NCI analysis.

    Parameters
    ----------
    cube_file : str
        Input Gaussian cube file containing electron density.

    output_file : str or None
        Output filename without '.nci'.
        If None, the cube filename is used.

    cutoff : float
        Maximum atom-atom distance used for initial BCP search.

    rho_min, rho_max : float
        Density range used to classify non-covalent BCPs.

    nproc : int
        Number of processes used for BCP search.

    Returns
    -------
    G : nx.Graph
        Final molecular interaction graph.
    """

    # ===================================================================== #
    # Input
    # ===================================================================== #

    print("=" * 70)
    print("NCI TOPOLOGICAL ANALYSIS")
    print("=" * 70)

    print(f"\nReading electron density: {cube_file}")

    origin, spacing, density, atoms = read_cube(cube_file)

    print(f"Grid shape: {density.shape}")
    print(f"Number of atoms: {len(atoms)}")

    # ===================================================================== #
    # B-spline interpolation
    # ===================================================================== #

    print("\nBuilding B-spline interpolator...")

    interp = BSplineAIM.from_density(density, origin, spacing)

    # ===================================================================== #
    # BCP search
    # ===================================================================== #

    print("\nSearching for bond critical points...")

    bcps = find_bcp_parallel(interp, atoms, cutoff=cutoff, nproc=nproc)

    print(f"Total BCPs found: {len(bcps)}")

    # ===================================================================== #
    # NCI filtering
    # ===================================================================== #

    bcps_nci, bcps_covalent = find_nci(bcps, rho_min=rho_min, rho_max=rho_max)

    print(f"Covalent BCPs: {len(bcps_covalent)}")
    print(f"Non-covalent BCPs: {len(bcps_nci)}")

    # ===================================================================== #
    # Build covalent molecular graph
    # ===================================================================== #

    print("\nBuilding covalent molecular graph...")

    G = build_covalent_graph(bcps_covalent, atoms)

    n_atoms = sum(1 for _, data in G.nodes(data=True) if data.get("node_type") == "atom")
    n_rings = sum(1 for _, data in G.nodes(data=True) if data.get("node_type") == "aromatic_center")
    n_covalent = sum(1 for _, _, data in G.edges(data=True) if data.get("bond_type") == "covalent")

    print(f"Atoms: {n_atoms}")
    print(f"Aromatic rings: {n_rings}")
    print(f"Covalent bonds: {n_covalent}")

    # ===================================================================== #
    # Add all NCI interactions
    # ===================================================================== #

    print("\nSearching for non-covalent interactions...")

    G = build_graph_with_nci(bcps_nci, G, atoms)
    G = build_graph_with_carbonyl(bcps_nci, G, atoms)
    G = build_graph_with_unclassified(bcps_nci, G, atoms)

    # ===================================================================== #
    # Summary
    # ===================================================================== #

    print("\nDetected interactions:")


    interaction_types = [
        "HB",
        "XB",
        "ChB",
        "PnB",
        "stacking",
        "T-stacking",
        "H-pi",
        "n-pi",
        "carbonyl",
        "unclassified"
    ]

    for bond_type in interaction_types:

        count = sum(
            1
            for _, _, data in G.edges(data=True)
            if data.get("bond_type") == bond_type
        )

        if count:
            print(
                f"  {bond_type:12s}: {count}"
            )

    # ===================================================================== #
    # Output
    # ===================================================================== #

    if output_file is None:

        output_file = os.path.splitext(
            cube_file
        )[0]

    input_ext = os.path.splitext(
        cube_file
    )[1]

    print(
        f"\nWriting results to: {output_file}.nci"
    )

    output(
        G,
        filename=output_file,
        ext=input_ext
    )

    print(
        "\nAnalysis completed."
    )

    return G

if __name__ == "__main__":
    G = main(r"C:\Users\User\Navuka\Proteins_NCI_analysis\SCF for manual\H_optimized\GFN2-xTB\SCF\HF\1ubq_HF-pcseg-1_opt_xtb_250_grid.cube", nproc=8)