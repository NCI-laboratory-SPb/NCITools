"""
Topological analysis of electron density: CUBE reading, B-spline interpolation,
BCP search, and critical point classification.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from numpy.linalg import eigvalsh, norm, solve
from scipy.ndimage import spline_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from ncitools.constants import bohr_to_angstrom, RADII, BONDI
from ncitools.utils import get_symbol

ANGSTROM_TO_BOHR: float = 1.0 / bohr_to_angstrom


# ----------------------------------------------------------------------------
# Covalent radii helpers (using RADII from constants)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# CUBE reader
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Cubic B-spline basis
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# B-spline interpolator
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Critical-point classification
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Pair geometry
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Single seed (midpoint)
# ----------------------------------------------------------------------------

def make_midpoint_seed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the midpoint of the segment AB as the only starting point."""
    return 0.5 * (np.asarray(a) + np.asarray(b))


# ----------------------------------------------------------------------------
# Robust Newton solver
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Worker-side BCP search (single seed)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Global BCP deduplication
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Candidate atom pairs
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Main BCP search
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Covalent / non-covalent separation + filtering
# ----------------------------------------------------------------------------

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