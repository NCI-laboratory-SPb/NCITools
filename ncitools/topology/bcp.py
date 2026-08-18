"""
NCITools
========

Fast brute-force QTAIM-like analysis of electron-density CUBE files.

Pipeline
--------
1. Read Gaussian/compatible CUBE.
2. Detect coordinate units from the CUBE unit flag.
3. Convert all geometry internally to Bohr.
4. Build cubic B-spline representation of rho.
5. Generate candidate atom pairs within user-defined radius.
6. Single-start Newton search for stationary points of rho (midpoint seed).
7. Keep only non-degenerate (3,-1) critical points.
8. Verify that CP belongs to the considered atom pair (geometry).
9. Deduplicate CPs globally by position.
10. Filter out NCI candidates with negative Laplacian and other artifacts.
11. Separate likely covalent and non-covalent contacts.
12. Classify NCI contacts geometrically:
       HB
       XB
       ChB
       PnB
       H-pi
       n-pi
       pi-pi stacking
       T-stacking
       n->pi*
       unclassified
13. Write .nci output.

IMPORTANT
---------
This is a fast screening algorithm, not a full QTAIM atomic-basin /
bond-path tracer.

A (3,-1) CP is a topological object. Its chemical interpretation
requires additional chemical/geometrical criteria. The code therefore
deliberately uses conservative "unclassified" output rather than
forcing every CP into a chemical category.
"""

from __future__ import annotations

import os
import random
import textwrap
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional, Union, List, Dict, Any, Tuple

import networkx as nx
import numpy as np
from numpy.linalg import eigvalsh, norm, solve
from scipy.ndimage import spline_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.quotes import quotes
from ncitools.correlations import correlation_1
from ncitools.utils import get_symbol
from ncitools.constants import bohr_to_angstrom, RADII, BONDI


ANGSTROM_TO_BOHR: float = 1.0 / bohr_to_angstrom


# ============================================================================
# 1. Atomic radii
# ============================================================================

def covalent_radius(symbol: str) -> float:
    """Return covalent radius in Angstrom."""
    return RADII.get(symbol, 1.50)


def bond_cutoff_angstrom(
    symbol1: str,
    symbol2: str,
    scale: float = 1.25,
) -> float:
    """
    Maximum distance used for a geometry-based covalent assignment.

    The value is deliberately somewhat generous because the purpose
    is screening, not rigorous bond-order determination.
    """
    return scale * (covalent_radius(symbol1) + covalent_radius(symbol2))


# ============================================================================
# 2. CUBE reader
# ============================================================================

class CubeData:
    """
    Container for a parsed CUBE file.

    All coordinates and grid vectors are converted to Bohr internally.
    """

    def __init__(
        self,
        origin: np.ndarray,
        axes: np.ndarray,
        density: np.ndarray,
        atoms: List[Tuple[int, np.ndarray]],
        coordinate_units: str,
        comments: List[str],
    ) -> None:
        self.origin = np.asarray(origin, dtype=float)
        self.axes = np.asarray(axes, dtype=float)
        self.density = np.asarray(density, dtype=float)
        self.atoms = atoms
        self.coordinate_units = coordinate_units
        self.comments = comments
        self.shape = self.density.shape

    @property
    def spacing(self) -> np.ndarray:
        """Voxel-vector lengths in Bohr."""
        return np.linalg.norm(self.axes, axis=1)


def _detect_cube_units(
    voxel_counts: np.ndarray,
    requested: str = "auto",
) -> str:
    """
    Detect coordinate units from the CUBE grid-count sign convention.

    Traditional Gaussian-style CUBE convention:
        positive -> Bohr
        negative -> Angstrom

    Mixed signs are ambiguous for a shared origin and are rejected
    in auto mode.
    """
    requested = requested.lower()
    if requested in {"bohr", "angstrom"}:
        return requested
    if requested != "auto":
        raise ValueError("cube_units must be 'auto', 'bohr' or 'angstrom'")

    signs = np.sign(voxel_counts)
    if np.all(signs > 0):
        return "bohr"
    if np.all(signs < 0):
        return "angstrom"
    raise ValueError(
        "CUBE file contains mixed-sign voxel counts. "
        "The coordinate units cannot be inferred safely. "
        "Use cube_units='bohr' or cube_units='angstrom'."
    )


def read_cube(
    filename: str,
    cube_units: str = "auto",
) -> CubeData:
    """
    Read a Gaussian-style CUBE file.

    Parameters
    ----------
    filename
        Input CUBE filename.
    cube_units
        'auto', 'bohr' or 'angstrom'.

    Returns
    -------
    CubeData
        Coordinates and grid vectors are stored internally in Bohr.
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError("File is too short to be a valid CUBE file.")

    comments = lines[:2]
    header = lines[2].split()

    natoms_raw = int(float(header[0]))
    natoms = abs(natoms_raw)

    origin = np.array(list(map(float, header[1:4])), dtype=float)

    # Standard CUBE: Nx vx vy vz, Ny ..., Nz ...
    grid_counts = []
    axes = []
    for line_number in range(3, 6):
        parts = lines[line_number].split()
        count = int(float(parts[0]))
        vector = np.array(list(map(float, parts[1:4])), dtype=float)
        grid_counts.append(count)
        axes.append(vector)

    grid_counts = np.asarray(grid_counts, dtype=int)
    axes = np.asarray(axes, dtype=float)

    units = _detect_cube_units(grid_counts, cube_units)
    scale = 1.0 if units == "bohr" else ANGSTROM_TO_BOHR

    origin *= scale
    axes *= scale

    # Atom records
    atoms = []
    atom_start = 6
    for i in range(natoms):
        parts = lines[atom_start + i].split()
        if len(parts) < 5:
            raise ValueError(f"Malformed atom record at line {atom_start + i + 1}.")
        Z = int(float(parts[0]))
        coord = np.array(list(map(float, parts[2:5])), dtype=float) * scale
        atoms.append((Z, coord))

    # Volumetric data
    data_start = atom_start + natoms
    data_text = " ".join(lines[data_start:])
    density = np.fromstring(data_text, sep=" ", dtype=float)
    expected_size = int(np.prod(np.abs(grid_counts)))
    if density.size < expected_size:
        raise ValueError(f"CUBE contains only {density.size} grid values, expected {expected_size}.")
    density = density[:expected_size]
    shape = tuple(np.abs(grid_counts))
    density = density.reshape(shape)

    return CubeData(
        origin=origin,
        axes=axes,
        density=density,
        atoms=atoms,
        coordinate_units=units,
        comments=comments,
    )


# ============================================================================
# 3. Cubic B-spline basis
# ============================================================================

def basis(t: np.ndarray) -> np.ndarray:
    """Centered cubic B-spline basis β(t)."""
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)

    m1 = (t >= -2.0) & (t <= -1.0)
    m2 = (t > -1.0) & (t <= 0.0)
    m3 = (t > 0.0) & (t <= 1.0)
    m4 = (t > 1.0) & (t <= 2.0)

    t1 = t[m1]
    t2 = t[m2]
    t3 = t[m3]
    t4 = t[m4]

    result[m1] = ((t1 + 2.0) ** 3) / 6.0
    result[m2] = (-3.0 * t2**3 - 6.0 * t2**2 + 4.0) / 6.0
    result[m3] = (3.0 * t3**3 - 6.0 * t3**2 + 4.0) / 6.0
    result[m4] = ((2.0 - t4) ** 3) / 6.0

    return result


def basis_deriv1(t: np.ndarray) -> np.ndarray:
    """First derivative of centered cubic B-spline."""
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)

    m1 = (t >= -2.0) & (t <= -1.0)
    m2 = (t > -1.0) & (t <= 0.0)
    m3 = (t > 0.0) & (t <= 1.0)
    m4 = (t > 1.0) & (t <= 2.0)

    t1 = t[m1]
    t2 = t[m2]
    t3 = t[m3]
    t4 = t[m4]

    result[m1] = 0.5 * (t1 + 2.0) ** 2
    result[m2] = 0.5 * (-3.0 * t2**2 - 4.0 * t2)
    result[m3] = 0.5 * (3.0 * t3**2 - 4.0 * t3)
    result[m4] = -0.5 * (2.0 - t4) ** 2

    return result


def basis_deriv2(t: np.ndarray) -> np.ndarray:
    """Second derivative of centered cubic B-spline."""
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)

    m1 = (t >= -2.0) & (t <= -1.0)
    m2 = (t > -1.0) & (t <= 0.0)
    m3 = (t > 0.0) & (t <= 1.0)
    m4 = (t > 1.0) & (t <= 2.0)

    t1 = t[m1]
    t2 = t[m2]
    t3 = t[m3]
    t4 = t[m4]

    result[m1] = t1 + 2.0
    result[m2] = -3.0 * t2 - 2.0
    result[m3] = 3.0 * t3 - 2.0
    result[m4] = 2.0 - t4

    return result


# ============================================================================
# 4. B-spline interpolator
# ============================================================================

class BSplineAIM:
    """
    Cubic B-spline interpolator for a 3D density.

    Internal coordinates are Cartesian Bohr.

    The CUBE grid vectors may be non-orthogonal:

        r = origin + g0 * a0 + g1 * a1 + g2 * a2

    Therefore grid -> Cartesian derivatives use the full Jacobian.
    """

    def __init__(self, coeffs: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> None:
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        self.axes = np.asarray(axes, dtype=float)
        if self.axes.shape != (3, 3):
            raise ValueError("axes must have shape (3, 3)")

        self.axes_inv = np.linalg.inv(self.axes)
        self.jacobian = self.axes.T
        self.inv_jacobian = np.linalg.inv(self.jacobian)
        self.grad_transform = self.inv_jacobian.T
        self.hess_transform = self.inv_jacobian.T

    @classmethod
    def from_density(
        cls,
        density: np.ndarray,
        origin: np.ndarray,
        axes: np.ndarray,
    ) -> BSplineAIM:
        """Build spline coefficients using scipy's spline_filter."""
        coeffs = spline_filter(np.asarray(density, dtype=float), order=3, mode="mirror")
        return cls(coeffs, origin, axes)

    def world_to_grid(self, r: np.ndarray) -> np.ndarray:
        """Convert Cartesian Bohr coordinates to grid coordinates."""
        r = np.asarray(r, dtype=float)
        return self.inv_jacobian @ (r - self.origin)

    def _local_data(self, g: np.ndarray) -> Optional[Tuple]:
        """
        Return the local 4x4x4 spline stencil.

        Points too close to the grid boundary are rejected instead
        of silently clipping indices. This avoids introducing an
        artificial density derivative caused by the old clipping
        strategy.
        """
        shape = self.coeffs.shape
        if np.any(g < 1.0) or np.any(g > np.asarray(shape) - 3.0):
            return None

        i0 = int(np.floor(g[0]))
        j0 = int(np.floor(g[1]))
        k0 = int(np.floor(g[2]))

        i = np.arange(i0 - 1, i0 + 3, dtype=int)
        j = np.arange(j0 - 1, j0 + 3, dtype=int)
        k = np.arange(k0 - 1, k0 + 3, dtype=int)

        ti = g[0] - i
        tj = g[1] - j
        tk = g[2] - k

        return (
            self.coeffs[np.ix_(i, j, k)],
            basis(ti), basis(tj), basis(tk),
            basis_deriv1(ti), basis_deriv1(tj), basis_deriv1(tk),
            basis_deriv2(ti), basis_deriv2(tj), basis_deriv2(tk),
        )

    def compute_all(self, r: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Return (rho, gradient, Hessian) at Cartesian coordinate r.
        """
        g = self.world_to_grid(r)
        local = self._local_data(g)
        if local is None:
            raise ValueError("Requested point lies too close to the CUBE boundary for the spline stencil.")

        (
            c,
            Bx, By, Bz,
            dBx, dBy, dBz,
            d2Bx, d2By, d2Bz,
        ) = local

        val = np.einsum("ijk,i,j,k->", c, Bx, By, Bz)
        gx = np.einsum("ijk,i,j,k->", c, dBx, By, Bz)
        gy = np.einsum("ijk,i,j,k->", c, Bx, dBy, Bz)
        gz = np.einsum("ijk,i,j,k->", c, Bx, By, dBz)
        grad_grid = np.array([gx, gy, gz], dtype=float)

        hxx = np.einsum("ijk,i,j,k->", c, d2Bx, By, Bz)
        hyy = np.einsum("ijk,i,j,k->", c, Bx, d2By, Bz)
        hzz = np.einsum("ijk,i,j,k->", c, Bx, By, d2Bz)
        hxy = np.einsum("ijk,i,j,k->", c, dBx, dBy, Bz)
        hxz = np.einsum("ijk,i,j,k->", c, dBx, By, dBz)
        hyz = np.einsum("ijk,i,j,k->", c, Bx, dBy, dBz)
        hess_grid = np.array([[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]], dtype=float)

        grad = self.grad_transform @ grad_grid
        hess = self.grad_transform @ hess_grid @ self.inv_jacobian
        hess = 0.5 * (hess + hess.T)

        return float(val), grad, hess

    def value(self, r: np.ndarray) -> float:
        return self.compute_all(r)[0]

    def gradient(self, r: np.ndarray) -> np.ndarray:
        return self.compute_all(r)[1]

    def hessian(self, r: np.ndarray) -> np.ndarray:
        return self.compute_all(r)[2]


# ============================================================================
# 5. Critical-point classification
# ============================================================================

def classify_hessian(
    eigs: np.ndarray,
    eig_tol: float = 1.0e-8,
) -> Tuple[int, Optional[int], str]:
    """
    Classify a rank-3 Hessian.

    Returns
    -------
    (rank, signature, cp_type)

    cp_type:
        (3,-3) nuclear maximum
        (3,-1) BCP/LCP
        (3,+1) RCP
        (3,+3) CCP
        degenerate
    """
    eigs = np.asarray(eigs, dtype=float)
    if eigs.size != 3:
        raise ValueError("Hessian must have three eigenvalues.")

    significant = np.abs(eigs) > eig_tol
    rank = int(np.count_nonzero(significant))
    if rank != 3:
        return rank, None, "degenerate"

    signs = np.sign(eigs)
    signature = int(np.sum(signs))
    mapping = {-3: "(3,-3)", -1: "(3,-1)", +1: "(3,+1)", +3: "(3,+3)"}
    return 3, signature, mapping.get(signature, "unknown")


# ============================================================================
# 6. Robust Newton solver
# ============================================================================

def newton_stationary_point(
    interp: BSplineAIM,
    r0: np.ndarray,
    grad_tol: float = 1.0e-6,
    max_iter: int = 60,
    damping: bool = True,
    max_step_bohr: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """
    Find a stationary point of rho.

    A damped Newton method with backtracking is used.

    Returns
    -------
    dict or None
        Contains position, gradient norm, iterations, eigenvalues.
    """
    r = np.asarray(r0, dtype=float).copy()
    best_r = None
    best_norm = float("inf")

    for iteration in range(max_iter):
        try:
            rho, grad, hess = interp.compute_all(r)
        except (ValueError, np.linalg.LinAlgError):
            return None

        gnorm = norm(grad)
        if gnorm < best_norm:
            best_norm = gnorm
            best_r = r.copy()

        if not np.isfinite(gnorm) or not np.all(np.isfinite(hess)):
            return None

        if gnorm <= grad_tol:
            eigs = eigvalsh(hess)
            return {
                "position": r.copy(),
                "rho": float(rho),
                "gradient": grad.copy(),
                "grad_norm": float(gnorm),
                "hessian": hess.copy(),
                "eigenvalues": eigs,
                "iterations": iteration + 1,
            }

        try:
            step = solve(hess, grad)
        except np.linalg.LinAlgError:
            return None

        step_norm = norm(step)
        if not np.isfinite(step_norm) or step_norm == 0.0:
            return None

        if step_norm > max_step_bohr:
            step *= max_step_bohr / step_norm

        if not damping:
            r_new = r - step
        else:
            current_norm = gnorm
            alpha = 1.0
            accepted = False
            for _ in range(12):
                candidate = r - alpha * step
                try:
                    _, candidate_grad, _ = interp.compute_all(candidate)
                    candidate_norm = norm(candidate_grad)
                except (ValueError, np.linalg.LinAlgError):
                    candidate_norm = float("inf")
                if np.isfinite(candidate_norm) and candidate_norm < current_norm:
                    r_new = candidate
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return None

        r = r_new

    # No convergence.
    return None


# ============================================================================
# 7. Pair geometry
# ============================================================================

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """Angle ABC in degrees."""
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    v1 = a - b
    v2 = c - b
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 <= 0.0 or n2 <= 0.0:
        return None
    cosine = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pair_geometry(
    r: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Optional[Dict[str, float]]:
    """
    Calculate geometry of point r relative to atom pair A-B.

    Returns
    -------
    dict
        distance_a
        distance_b
        atom_distance
        angle
        projection
        line_distance
        projection_fraction
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    d = norm(ab)
    if d <= 1.0e-12:
        return None

    u = ab / d
    ar = r - a
    projection = np.dot(ar, u)
    closest = a + projection * u
    line_distance = norm(r - closest)
    fraction = projection / d
    angle = _angle(a, r, b)

    return {
        "distance_a": norm(r - a),
        "distance_b": norm(r - b),
        "atom_distance": d,
        "angle": angle,
        "projection": projection,
        "projection_fraction": fraction,
        "line_distance": line_distance,
    }


def valid_bcp_pair_geometry(
    geometry: Dict[str, float],
    min_angle: float = 140.0,
    projection_margin: float = 0.20,
    max_line_fraction: float = 0.35,
) -> bool:
    """
    Decide whether a CP is geometrically associated with A-B.

    This is deliberately permissive enough to allow curved/anisotropic
    density paths, but rejects obviously unrelated stationary points.
    """
    if geometry is None:
        return False
    if geometry["angle"] is None:
        return False
    if geometry["angle"] < min_angle:
        return False
    fraction = geometry["projection_fraction"]
    if fraction < -projection_margin or fraction > 1.0 + projection_margin:
        return False
    if geometry["line_distance"] > max_line_fraction * geometry["atom_distance"]:
        return False
    return True


# ============================================================================
# 8. Single seed (midpoint)
# ============================================================================

def make_midpoint_seed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the midpoint of the segment AB as the only starting point."""
    return 0.5 * (np.asarray(a) + np.asarray(b))


# ============================================================================
# 9. Worker-side BCP search (single seed)
# ============================================================================

_WORKER_INTERP: Optional[BSplineAIM] = None


def _init_worker(coeffs: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> None:
    """ProcessPool initializer: transfer density coefficients once per worker."""
    global _WORKER_INTERP
    _WORKER_INTERP = BSplineAIM(coeffs, origin, axes)


def _search_pair_worker(args: Tuple[int, int, np.ndarray, np.ndarray, Dict]) -> List[Dict]:
    """
    Worker function.

    Performs exactly one Newton search starting from the midpoint.
    Returns a single candidate if a valid (3,-1) CP is found.
    """
    atom1_idx, atom2_idx, atom1_position, atom2_position, settings = args
    interp = _WORKER_INTERP
    if interp is None:
        raise RuntimeError("Worker interpolator was not initialized.")

    a = np.asarray(atom1_position, dtype=float)
    b = np.asarray(atom2_position, dtype=float)
    seed = make_midpoint_seed(a, b)

    result = newton_stationary_point(
        interp, seed,
        grad_tol=settings["grad_tol"],
        max_iter=settings["max_iter"],
        damping=True,
        max_step_bohr=settings["max_step_bohr"],
    )
    if result is None:
        return []

    geometry = pair_geometry(result["position"], a, b)
    if not valid_bcp_pair_geometry(
        geometry,
        min_angle=settings["min_bcp_angle"],
        projection_margin=settings["projection_margin"],
        max_line_fraction=settings["max_line_fraction"],
    ):
        return []

    eigs = result["eigenvalues"]
    rank, signature, cp_type = classify_hessian(eigs, eig_tol=settings["eig_tol"])
    if cp_type != "(3,-1)":
        return []

    candidate = {
        "position": result["position"].copy(),
        "rho": float(result["rho"]),
        "grad_norm": float(result["grad_norm"]),
        "eigenvalues": eigs.copy(),
        "laplacian": float(np.sum(eigs)),
        "ellipticity": (float(eigs[0] / eigs[1] - 1.0) if abs(eigs[1]) > settings["eig_tol"] else 0.0),
        "atom1": int(atom1_idx),
        "atom2": int(atom2_idx),
        "pair_distance": float(geometry["atom_distance"]),
        "pair_angle": float(geometry["angle"]),
        "line_distance": float(geometry["line_distance"]),
        "projection_fraction": float(geometry["projection_fraction"]),
        "distance_a": float(geometry["distance_a"]),
        "distance_b": float(geometry["distance_b"]),
    }
    return [candidate]


# ============================================================================
# 10. Global BCP deduplication
# ============================================================================

def _compute_bcp_score(bcp: Dict) -> float:
    """
    Compute a geometry score for a BCP candidate.
    Lower is better.
    """
    distance_penalty = bcp["distance_a"] + bcp["distance_b"]
    angle = bcp["pair_angle"]
    if angle is None:
        return float("inf")
    angle_penalty = (180.0 - angle) / 40.0
    line_penalty = bcp["line_distance"] / max(bcp["pair_distance"], 1.0e-8)
    proj = bcp["projection_fraction"]
    proj_penalty = 0.0
    if proj < 0.0:
        proj_penalty += abs(proj)
    elif proj > 1.0:
        proj_penalty += proj - 1.0
    return distance_penalty + line_penalty + angle_penalty + proj_penalty


def deduplicate_bcps(
    candidates: List[Dict],
    position_tol_bohr: float = 0.05,
) -> List[Dict]:
    """
    Globally deduplicate BCP candidates by position.

    For each cluster of CPs within position_tol_bohr, keep the one
    with the smallest geometry score (best pair assignment).

    Returns a list of unique BCPs.
    """
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (x["position"][0], x["position"][1], x["position"][2])
    )

    clusters = []
    for cand in sorted_candidates:
        pos = cand["position"]
        found = False
        for cluster in clusters:
            ref_pos = cluster[0]["position"]
            if norm(pos - ref_pos) <= position_tol_bohr:
                cluster.append(cand)
                found = True
                break
        if not found:
            clusters.append([cand])

    unique = []
    for cluster in clusters:
        best = min(cluster, key=_compute_bcp_score)
        unique.append(best)

    return sorted(
        unique,
        key=lambda x: (x["atom1"], x["atom2"], x["position"][0], x["position"][1], x["position"][2])
    )


# ============================================================================
# 11. Candidate atom pairs
# ============================================================================

def generate_candidate_pairs(
    atoms: List[Tuple[int, np.ndarray]],
    bcp_search_radius: float = 5.0,  # Angstrom
) -> List[Tuple[int, int]]:
    """
    Generate all atom pairs within the search radius.

    KDTree avoids O(N^2) distance checking for large systems.
    """
    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    radius_bohr = bcp_search_radius * ANGSTROM_TO_BOHR
    tree = cKDTree(coordinates)
    pairs = tree.query_pairs(radius_bohr, output_type="set")
    return sorted(pairs)


# ============================================================================
# 12. Main BCP search
# ============================================================================

def find_bcps_parallel(
    interp: BSplineAIM,
    atoms: List[Tuple[int, np.ndarray]],
    bcp_search_radius: float = 5.0,
    nproc: int = 4,
    grad_tol: float = 1.0e-6,
    max_iter: int = 60,
    eig_tol: float = 1.0e-8,
    min_bcp_angle: float = 140.0,
    projection_margin: float = 0.20,
    max_line_fraction: float = 0.35,
    max_step_angstrom: float = 0.75,
    dedup_tol_angstrom: float = 0.03,
) -> List[Dict]:
    """
    Fast single-start BCP search.

    Parameters are expressed in chemically intuitive units where
    possible; internally all geometry is converted to Bohr.
    """
    pairs = generate_candidate_pairs(atoms, bcp_search_radius)
    if not pairs:
        return []

    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)

    settings = {
        "grad_tol": grad_tol,
        "max_iter": max_iter,
        "eig_tol": eig_tol,
        "min_bcp_angle": min_bcp_angle,
        "projection_margin": projection_margin,
        "max_line_fraction": max_line_fraction,
        "max_step_bohr": max_step_angstrom * ANGSTROM_TO_BOHR,
    }

    tasks = [
        (int(i), int(j), coordinates[i], coordinates[j], settings)
        for i, j in pairs
    ]

    if nproc <= 1:
        global _WORKER_INTERP
        _WORKER_INTERP = interp
        results = []
        iterator = (_search_pair_worker(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Searching BCPs"):
            results.extend(result)
    else:
        with ProcessPoolExecutor(
            max_workers=nproc,
            initializer=_init_worker,
            initargs=(interp.coeffs, interp.origin, interp.axes),
        ) as executor:
            results = []
            iterator = executor.map(
                _search_pair_worker,
                tasks,
                chunksize=max(1, len(tasks) // (nproc * 8)),
            )
            for result in tqdm(iterator, total=len(tasks), desc="Searching BCPs"):
                results.extend(result)

    deduped = deduplicate_bcps(
        results,
        position_tol_bohr=dedup_tol_angstrom * ANGSTROM_TO_BOHR,
    )
    return deduped


# Backward-compatible name.
find_bcp_parallel = find_bcps_parallel


# ============================================================================
# 13. Covalent / non-covalent separation + filtering
# ============================================================================

def classify_covalent_like(
    bcp: Dict,
    atoms: List[Tuple[int, np.ndarray]],
    rho_threshold: float = 0.089,
    covalent_scale: float = 1.25,
    require_negative_laplacian: bool = False,
) -> bool:
    """
    Decide whether a BCP is likely covalent.

    This is intentionally NOT based on rho alone.

    A BCP is considered covalent-like if:

        rho >= rho_threshold

    AND

        atom distance <=
        covalent_scale * (r_cov(A) + r_cov(B))

    Optionally the Laplacian can also be required to be negative.

    This is a screening classifier, not a universal QTAIM rule.
    """
    i = int(bcp["atom1"])
    j = int(bcp["atom2"])
    symbol_i = get_symbol(atoms[i][0])
    symbol_j = get_symbol(atoms[j][0])
    distance_angstrom = bcp["pair_distance"] * bohr_to_angstrom
    cutoff = bond_cutoff_angstrom(symbol_i, symbol_j, scale=covalent_scale)

    if bcp["rho"] < rho_threshold:
        return False
    if distance_angstrom > cutoff:
        return False
    if require_negative_laplacian and bcp["laplacian"] >= 0.0:
        return False
    return True


def filter_nci_artifacts(
    bcps: List[Dict],
    max_rho: float = 0.15,  # empirical upper limit for NCI
) -> List[Dict]:
    """
    Remove BCPs that are clearly artefacts for NCI.

    Criteria:
      - Laplacian must be positive (∇²ρ > 0)
      - rho must not exceed a conservative upper bound
    """
    filtered = []
    for bcp in bcps:
        if bcp["laplacian"] < 0.0:
            continue
        if bcp["rho"] > max_rho:
            continue
        filtered.append(bcp)
    return filtered


def find_nci(
    bcps: List[Dict],
    atoms: List[Tuple[int, np.ndarray]],
    rho_threshold: float = 0.089,
    covalent_scale: float = 1.25,
    require_negative_laplacian: bool = False,
    filter_artifacts: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split BCPs into likely covalent and NCI sets.

    NCI set is additionally filtered to remove artefacts
    (negative Laplacian, excessive rho).
    """
    bcps_nci = []
    bcps_covalent = []
    for bcp in bcps:
        if classify_covalent_like(
            bcp, atoms,
            rho_threshold=rho_threshold,
            covalent_scale=covalent_scale,
            require_negative_laplacian=require_negative_laplacian,
        ):
            bcps_covalent.append(bcp)
        else:
            bcps_nci.append(bcp)

    if filter_artifacts:
        bcps_nci = filter_nci_artifacts(bcps_nci)

    return bcps_nci, bcps_covalent


# ============================================================================
# 14. Graph helpers (for nx.Graph)
# ============================================================================

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


# ============================================================================
# 15. Aromatic/ring detection
# ============================================================================

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


# ============================================================================
# 16. Aromatic helpers
# ============================================================================

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


# ============================================================================
# 17. BCP property helper
# ============================================================================

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


# ============================================================================
# 18. Aromatic interaction classifier
# ============================================================================

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
                "ring_1": ring1,
                "ring_2": ring2,
                "atom_num_cycle_1": int(min(G.nodes[ring1]["cycle"])),
                "atom_num_cycle_2": int(min(G.nodes[ring2]["cycle"])),
                "centroid_distance": float(centroid_distance),
                "offset": float(offset),
                "angle": float(ring_angle),
            }
            attrs.update(_bcp_properties(bcp))
            return {"u": ring1, "v": ring2, "attrs": attrs}

        # T stacking
        if ring_angle >= 80.0:
            attrs = {
                "bond_type": "T-stacking",
                "ring_1": ring1,
                "ring_2": ring2,
                "atom_num_cycle_1": int(min(G.nodes[ring1]["cycle"])),
                "atom_num_cycle_2": int(min(G.nodes[ring2]["cycle"])),
                "centroid_distance": float(centroid_distance),
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
            "ring": ring,
            "atom_num_cycle": int(min(G.nodes[ring]["cycle"])),
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
        "ring": ring,
        "atom_num_cycle": int(min(G.nodes[ring]["cycle"])),
        "lp_centroid_distance": float(distance),
        "angle_to_normal": float(angle_to_normal),
        "angle": float(best_angle),
        "lp_neighbour_num": int(best_neighbour),
    }
    attrs.update(_bcp_properties(bcp))
    return {"u": lp_idx, "v": ring, "attrs": attrs}


# ============================================================================
# 19. Ordinary NCI classifier
# ============================================================================

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


# ============================================================================
# 20. Carbonyl n -> pi*
# ============================================================================

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


# ============================================================================
# 21. Unclassified NCI contacts
# ============================================================================

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


# ============================================================================
# 22. Output
# ============================================================================

def output(
    G: nx.Graph,
    filename: str = "default",
    ext: str = ".cube",
    correlation=correlation_1,
) -> str:
    """
    Write .nci report.

    All distances are written in Angstrom.
    """
    now = datetime.now()

    edges_by_type = {}
    for _, _, data in G.edges(data=True):
        bond_type = data.get("bond_type")
        if bond_type is None:
            continue
        edges_by_type.setdefault(bond_type, []).append(data)

    def atom_number(data: Dict, key: str) -> str:
        value = data.get(key)
        return "—" if value is None else str(int(value) + 1)

    def fmt_float(data: Dict, key: str, digits: int = 3) -> str:
        value = data.get(key)
        return "—" if value is None else str(round(float(value), digits))

    def fmt_distance(data: Dict, key: str, digits: int = 3) -> str:
        value = data.get(key)
        return "—" if value is None else str(round(float(value), digits))

    def fmt_rho(data: Dict) -> str:
        return fmt_float(data, "rho", 5)

    def fmt_laplacian(data: Dict) -> str:
        return fmt_float(data, "laplacian", 5)

    def fmt_ellipticity(data: Dict) -> str:
        return fmt_float(data, "ellipticity", 3)

    sections = [
        {
            "key": "HB",
            "title": "HYDROGEN BONDS",
            "header": [
                "№", "Type", "D", "H", "A",
                "D-H (Å)", "H···A (Å)", "D···A (Å)", "Angle (°)",
                "ρ", "∇²ρ", "ε", "Energy"
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['donor_symbol']}-H···{d['acceptor_symbol']}",
                atom_number(d, "donor_num"),
                atom_number(d, "hydrogen_num"),
                atom_number(d, "acceptor_num"),
                fmt_distance(d, "h_donor_distance"),
                fmt_distance(d, "distance"),
                fmt_distance(d, "heavy_atom_distance"),
                fmt_float(d, "angle", 1),
                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
                round(float(correlation(float(d["rho"]), f"{d['donor_symbol']}-H...{d['acceptor_symbol']}")), 1),
            ],
        },
        {"key": "XB", "title": "HALOGEN BONDS"},
        {"key": "ChB", "title": "CHALCOGEN BONDS"},
        {"key": "PnB", "title": "PNICTOGEN BONDS"},
        {
            "key": "stacking",
            "title": "π-π STACKING",
            "header": ["№", "Type", "Ring 1", "Ring 2", "Centroid (Å)", "Offset (Å)", "Angle (°)"],
            "row_func": lambda d, i: [
                i + 1, "parallel",
                atom_number(d, "atom_num_cycle_1"),
                atom_number(d, "atom_num_cycle_2"),
                fmt_distance(d, "centroid_distance"),
                fmt_distance(d, "offset"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "T-stacking",
            "title": "T-SHAPED STACKING",
            "header": ["№", "Type", "Ring 1", "Ring 2", "Centroid (Å)", "Angle (°)"],
            "row_func": lambda d, i: [
                i + 1, "T-shaped",
                atom_number(d, "atom_num_cycle_1"),
                atom_number(d, "atom_num_cycle_2"),
                fmt_distance(d, "centroid_distance"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "H-pi",
            "title": "X-H···π INTERACTIONS",
            "header": ["№", "Type", "X", "H", "Ring", "H···π (Å)", "Angle (°)", "Projection (Å)"],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_x']}-H···π",
                atom_number(d, "x_num"),
                atom_number(d, "h_num"),
                atom_number(d, "atom_num_cycle"),
                fmt_distance(d, "h_centroid_distance"),
                fmt_float(d, "angle", 1),
                fmt_distance(d, "projection_distance"),
            ],
        },
        {
            "key": "n-pi",
            "title": "LONE PAIR···π",
            "header": ["№", "Type", "Atom", "Ring", "Distance (Å)", "Angle (°)"],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_lp']}···π",
                atom_number(d, "lp_num"),
                atom_number(d, "atom_num_cycle"),
                fmt_distance(d, "lp_centroid_distance"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "carbonyl",
            "title": "CARBONYL n→π*",
            "header": ["№", "Type", "Donor", "Acceptor C", "C=O", "Distance (Å)", "BD angle (°)", "Plane angle (°)", "ρ", "∇²ρ", "ε"],
            "row_func": lambda d, i: [
                i + 1, "n→π*",
                atom_number(d, "donor_oxygen"),
                atom_number(d, "acceptor_carbon"),
                atom_number(d, "carbonyl_oxygen"),
                fmt_distance(d, "distance"),
                fmt_float(d, "burgi_dunitz_angle", 1),
                fmt_float(d, "plane_angle", 1),
                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },
        {
            "key": "unclassified",
            "title": "UNCLASSIFIED NON-COVALENT CONTACTS",
            "header": ["№", "Type", "Atom 1", "Atom 2", "Element 1", "Element 2", "Distance (Å)", "ρ", "∇²ρ", "ε"],
            "row_func": lambda d, i: [
                i + 1, "unclassified",
                atom_number(d, "atom1_num"),
                atom_number(d, "atom2_num"),
                d.get("atom1_symbol", "—"),
                d.get("atom2_symbol", "—"),
                fmt_distance(d, "distance"),
                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },
    ]

    # Add heavy-atom sections headers
    heavy_headers = ["№", "Type", "X", "E", "Nu", "E···Nu (Å)", "Angle (°)", "ρ", "∇²ρ", "ε"]
    for key, title in [("XB", "HALOGEN BONDS"), ("ChB", "CHALCOGEN BONDS"), ("PnB", "PNICTOGEN BONDS")]:
        for section in sections:
            if section["key"] == key:
                section["header"] = heavy_headers
                section["row_func"] = lambda d, i: [
                    i + 1,
                    f"{d.get('neighbor_symbol', '—')}-{d.get('electrophile_symbol', '—')}···{d.get('nucleophile_symbol', '—')}",
                    atom_number(d, "neighbor_num"),
                    atom_number(d, "electrophile_num"),
                    atom_number(d, "nucleophile_num"),
                    fmt_distance(d, "distance"),
                    fmt_float(d, "angle", 1),
                    fmt_rho(d),
                    fmt_laplacian(d),
                    fmt_ellipticity(d),
                ]

    output_filename = f"{filename}.nci"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("NCITools: Topological analysis of electron density\n")
        f.write(f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {filename}{ext}\n")

        for section in sections:
            interaction_edges = edges_by_type.get(section["key"], [])
            if not interaction_edges:
                continue
            f.write("\n" + "=" * 120 + "\n")
            f.write(section["title"])
            f.write("\n" + "=" * 120 + "\n\n")

            rows = [section["row_func"](edge, i) for i, edge in enumerate(interaction_edges)]
            from tabulate import tabulate
            table = tabulate(rows, headers=section["header"], tablefmt="github", numalign="right", stralign="center")
            f.write(table)
            f.write("\n")

        philosopher, quote = random.choice(quotes)
        f.write("\n\n")
        f.write(textwrap.fill(quote.upper(), width=40, initial_indent=" ", subsequent_indent=" "))
        f.write(f"\n   -- {philosopher.upper()}\n")

    return output_filename


# ============================================================================
# 23. Main pipeline
# ============================================================================

def main(
    cube_file: str,
    output_file: Optional[str] = None,
    cube_units: str = "auto",
    bcp_search_radius: float = 5.0,
    nproc: int = 4,
    grad_tol: float = 1.0e-6,
    max_iter: int = 60,
    eig_tol: float = 1.0e-8,
    min_bcp_angle: float = 140.0,
    projection_margin: float = 0.20,
    max_line_fraction: float = 0.35,
    rho_covalent_threshold: float = 0.089,
    covalent_scale: float = 1.25,
    require_negative_laplacian: bool = False,
    aromatic_mode: str = "heuristic",
    correlation=correlation_1,
) -> nx.Graph:
    """
    Run complete analysis.

    All distance-related public parameters are in Angstrom.

    Internally:
        coordinates -> Bohr
        grid vectors -> Bohr
        distances -> Bohr
        density derivatives -> atomic units
    """
    print("=" * 70)
    print("NCI TOPOLOGICAL ANALYSIS")
    print("=" * 70)

    print(f"\nReading electron density: {cube_file}")
    cube = read_cube(cube_file, cube_units=cube_units)
    print(f"Detected coordinate units: {cube.coordinate_units}")
    print(f"Grid shape: {cube.shape}")
    print(f"Number of atoms: {len(cube.atoms)}")
    print("Internal coordinate system: Bohr")
    print("Grid spacing (Å): " + ", ".join(f"{x * bohr_to_angstrom:.4f}" for x in cube.spacing))

    print("\nBuilding B-spline interpolator...")
    interp = BSplineAIM.from_density(cube.density, cube.origin, cube.axes)

    print("\nSearching for (3,-1) critical points...")
    bcps = find_bcps_parallel(
        interp, cube.atoms,
        bcp_search_radius=bcp_search_radius,
        nproc=nproc,
        grad_tol=grad_tol,
        max_iter=max_iter,
        eig_tol=eig_tol,
        min_bcp_angle=min_bcp_angle,
        projection_margin=projection_margin,
        max_line_fraction=max_line_fraction,
    )
    print(f"Unique (3,-1) BCPs found: {len(bcps)}")

    bcps_nci, bcps_covalent = find_nci(
        bcps, cube.atoms,
        rho_threshold=rho_covalent_threshold,
        covalent_scale=covalent_scale,
        require_negative_laplacian=require_negative_laplacian,
        filter_artifacts=True,
    )
    print(f"Covalent-like BCPs: {len(bcps_covalent)}")
    print(f"NCI BCPs (after filtering): {len(bcps_nci)}")

    print("\nBuilding covalent molecular graph...")
    G = build_covalent_graph(bcps_covalent, cube.atoms, aromatic_mode=aromatic_mode)

    n_atoms = sum(1 for _, data in G.nodes(data=True) if data.get("node_type") == "atom")
    n_rings = sum(1 for _, data in G.nodes(data=True) if data.get("node_type") == "aromatic_center" and data.get("aromatic", False))
    n_covalent = sum(1 for _, _, data in G.edges(data=True) if data.get("bond_type") == "covalent")
    print(f"Atoms: {n_atoms}")
    print(f"Aromatic rings: {n_rings}")
    print(f"Covalent BCP edges: {n_covalent}")

    print("\nSearching for non-covalent interactions...")
    G = build_graph_with_nci(bcps_nci, G, cube.atoms)
    G = build_graph_with_carbonyl(bcps_nci, G, cube.atoms)
    G = build_graph_with_unclassified(bcps_nci, G, cube.atoms)

    print("\nDetected interactions:")
    interaction_types = ["HB", "XB", "ChB", "PnB", "stacking", "T-stacking", "H-pi", "n-pi", "carbonyl", "unclassified"]
    for bond_type in interaction_types:
        count = sum(1 for _, _, data in G.edges(data=True) if data.get("bond_type") == bond_type)
        if count:
            print(f"  {bond_type:12s}: {count}")

    if output_file is None:
        output_file = os.path.splitext(cube_file)[0]
    input_ext = os.path.splitext(cube_file)[1]
    print(f"\nWriting results to: {output_file}.nci")
    output(G, filename=output_file, ext=input_ext, correlation=correlation)

    print("\nAnalysis completed.")
    return G


# ============================================================================
# 24. Script entry point
# ============================================================================

if __name__ == "__main__":
    G = main(
        r"C:\Users\User\PycharmProjects\NCITools\tests\data\carbonyl\carbonyl_1.cub",
        nproc=8,
        bcp_search_radius=3.7,
        cube_units="auto",
        rho_covalent_threshold=0.089,
        covalent_scale=1.25,
        aromatic_mode="heuristic",
    )