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

from ncitools.correlations import correlation_1   # default energy estimator
from ncitools.utils import get_symbol
from ncitools.constants import bohr_to_angstrom

# -------------------------------------------------------------
#  Поиск атома-донора для водорода (ближайший атом, кроме акцептора)
# -------------------------------------------------------------
def find_donor(hydrogen_idx, atoms, exclude_idx):
    """
    Находит атом, ковалентно связанный с водородом (индекс hydrogen_idx),
    исключая атом с индексом exclude_idx (акцептор) и сам водород.
    Возвращает индекс донора или None, если не найден (например, если молекула состоит только из водорода).
    """
    h_pos = atoms[hydrogen_idx][1]
    best_dist = np.inf
    best_idx = None
    for i, (z, pos) in enumerate(atoms):
        # Пропускаем сам водород и атом-акцептор
        if i == hydrogen_idx or i == exclude_idx:
            continue
        dist = norm(h_pos - pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx

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
        This is the most efficient way to obtain all three.
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

def enrich_bcps_with_types(bcps, atoms):
    """For each BCP, identify the two nearest atoms and assign a contact type."""

    if not bcps:
        return []
    coords = np.array([atom[1] for atom in atoms])
    tree = cKDTree(coords)

    enriched = []
    for bcp in bcps:
        pos = bcp['position']
        # два ближайших атома
        dists, indices = tree.query(pos, k=2)
        atom1 = atoms[indices[0]]
        atom2 = atoms[indices[1]]

        sym1 = get_symbol(atom1[0])
        sym2 = get_symbol(atom2[0])

        has_H1 = (atom1[0] == 1)
        has_H2 = (atom2[0] == 1)

        bcp_copy = bcp.copy()
        bcp_copy['atom1'] = indices[0]
        bcp_copy['atom2'] = indices[1]

        # Обрабатываем все возможные комбинации наличия водорода
        if has_H1 and has_H2:
            continue
        elif has_H1 and not has_H2:
            # атом1 – водород, атом2 – акцептор
            donor_idx = find_donor(indices[0], atoms, exclude_idx=indices[1])
            if donor_idx is not None:
                donor_sym = get_symbol(atoms[donor_idx][0])
                contact_type = f"{donor_sym}-H...{sym2}"
            else:
                contact_type = f"{sym1}-{sym2}"   # fallback (донор не найден)
        elif not has_H1 and has_H2:
            # атом2 – водород, атом1 – акцептор
            donor_idx = find_donor(indices[1], atoms, exclude_idx=indices[0])
            if donor_idx is not None:
                donor_sym = get_symbol(atoms[donor_idx][0])
                contact_type = f"{donor_sym}-H...{sym1}"
            else:
                contact_type = f"{sym1}-{sym2}"
        else:
            # нет водорода – обычный нековалентный контакт между двумя тяжёлыми атомами
            contact_type = f"{sym1}-{sym2}"

        bcp_copy['contact_type'] = contact_type
        enriched.append(bcp_copy)

    return enriched

def find_nci(bcps, atoms, rho_min=0.001, rho_max=0.1):
    """Filter BCPs by density and add contact type information."""
    bcps_filtered = [bcp for bcp in bcps if rho_min <= bcp['rho'] <= rho_max]
    return enrich_bcps_with_types(bcps_filtered, atoms)

def build_graph_with_nci(bcps, atoms):
    """Build a NetworkX graph where edges are NCI (from bcps)."""
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    for bcp in tqdm(bcps, desc='Building graph from BCPs'):
        G.add_edge(bcp['atom1'], bcp['atom2'],
                   bond_type=bcp['contact_type'],
                   position=bcp['position'],
                   rho=bcp['rho'],
                   laplacian=bcp['laplacian'],
                   eigenvalues=bcp['eigenvalues'],
                   ellipticity=bcp['ellipticity'])
    return G


def output(G: nx.classes.graph.Graph, filename='default', ext='.default', correlation=correlation_1):
    """Write NCI results to a .nci file (topological analysis)."""

    now = datetime.now()
    nci_list = [i[1] for i in G.edges.items() if i[1]['bond_type'] != 'covalent']

    with open(f'{filename}.nci', 'w', encoding='utf-8') as f:
        f.write(f'AIM-NCI: non-covalent interactions.              Date: {now}\n')
        f.write(f'Uploaded file: {filename}{ext}\n\n')

        # Заголовок: добавлен столбец Type после номера BCP
        header = (f"{'BCP':>5} {'Type':>13} {'X (Å)':>12} {'Y (Å)':>12} {'Z (Å)':>12} "
                  f"{'ρ (e/bohr³)':>16} {'∇²ρ (e/bohr⁵)':>12} "
                  f"{'λ1':>6} {'λ2':>12} {'λ3':>12} {'ε':>10}    {'Energy (kcal/mol)':>10}")
        f.write(header + '\n')

        if not nci_list:
            f.write('No non-covalent interactions found.\n')
            f.write('Buy developer a coffee!\n')
        else:
            for index, bcp in enumerate(nci_list):
                x = bcp['position'][0] * bohr_to_angstrom
                y = bcp['position'][1] * bohr_to_angstrom
                z = bcp['position'][2] * bohr_to_angstrom
                rho = bcp['rho']
                lap = bcp['laplacian']
                eigs = bcp['eigenvalues']
                eps = bcp['ellipticity']
                typ = bcp['bond_type']
                energy = correlation(rho, typ)

                line = (f"{index+1:5d} {typ:>15} {x:12.7f} {y:12.7f} {z:12.7f} "
                        f"{rho:12.7f} {lap:12.7f} "
                        f"{eigs[0]:12.7f} {eigs[1]:12.7f} {eigs[2]:12.7f} {eps:10.6f} {energy:10.3f}")
                f.write(line + '\n')

            f.write('Buy developer a coffee!\n')

# ============================================================================
#  Main script
# ============================================================================

if __name__ == "__main__":
    cube_file = input("Enter the path to the cube file: ").strip().replace('"','')
    filename = os.path.basename(cube_file).split('.')[0]
    abs_dir = os.path.dirname(cube_file)
    ext = os.path.splitext(cube_file)[1]
    os.chdir(abs_dir)

    # Чтение
    print('Reading electron density...')
    origin, spacing, density, atoms = read_cube(cube_file)
    print('Reading is done!')

    # Интерполятор
    print('Starting interpolation of grid data')
    interp = BSplineAIM.from_density(density, origin, spacing)
    print('Interpolation completed')

    # Поиск всех BCP
    print('Starting topological analysis...')
    bcps = find_bcp_parallel(
        interp,
        atoms,
        cutoff=5.0,
        nproc=8
    )


    # Отбор NCI по плотности и определение типов
    nci_list = find_nci(bcps, atoms, rho_min=1e-3)
    G = build_graph_with_nci(nci_list, atoms)
    print(f"\nNon-covalent interactions found: {len(nci_list)}")

    # Запись в файл
    output(G, filename=filename, ext=ext)
