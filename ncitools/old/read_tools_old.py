import os
import numpy as np
import re
from ase import Atoms
from ase.io.proteindatabank import read_proteindatabank
from ase.io.gaussian import read_gaussian_in

# Radii used to determine connectivity in symmetry corrections
# Covalent radii taken from Cambridge Structural Database
RADII = {'H': 0.32, 'He': 0.93, 'Li': 1.23, 'Be': 0.90, 'B': 0.82, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.72,
         'Ne': 0.71, 'Na': 1.54, 'Mg': 1.36, 'Al': 1.18, 'Si': 1.11, 'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.98,
         'K': 2.03, 'Ca': 1.74, 'Sc': 1.44, 'Ti': 1.32, 'V': 1.22, 'Cr': 1.18, 'Mn': 1.17, 'Fe': 1.17, 'Co': 1.16,
         'Ni': 1.15, 'Cu': 1.17, 'Zn': 1.25, 'Ga': 1.26, 'Ge': 1.22, 'As': 1.20, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.12,
         'Rb': 2.16, 'Sr': 1.91, 'Y': 1.62, 'Zr': 1.45, 'Nb': 1.34, 'Mo': 1.30, 'Tc': 1.27, 'Ru': 1.25, 'Rh': 1.25,
         'Pd': 1.28, 'Ag': 1.34, 'Cd': 1.48, 'In': 1.44, 'Sn': 1.41, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
         'Cs': 2.35, 'Ba': 1.98, 'La': 1.69, 'Lu': 1.60, 'Hf': 1.44, 'Ta': 1.34, 'W': 1.30, 'Re': 1.28, 'Os': 1.26,
         'Ir': 1.27, 'Pt': 1.30, 'Au': 1.34, 'Hg': 1.49, 'Tl': 1.48, 'Pb': 1.47, 'Bi': 1.46, 'X': 0}
# Bondi van der Waals radii for all atoms from: Bondi, A. J. Phys. Chem. 1964, 68, 441-452,
# except hydrogen, which is taken from Rowland, R. S.; Taylor, R. J. Phys. Chem. 1996, 100, 7384-7391.
# Radii unavailable in either of these publications are set to 2 Angstrom
# (Unfinished)
BONDI = {'H': 1.09, 'He': 1.40, 'Li': 1.82, 'Be': 2.00, 'B': 2.00, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
         'Ne': 1.54, 'P': 1.80, 'S': 1.80, 'Se': 1.90}

bohr_to_angstrom = 0.52918

def read_pdb(file):

    ext = os.path.splitext(file)[1]
    if ext == ".pdb":

        pdb_file = read_proteindatabank(file)

        return pdb_file

    else:
        print("Unrecognised file type", ext)


def angle(atom1, atom2, atom3):
    """

    Parameters
    ----------
    atom1 : int
        First atom number.
    atom2 : int
        Second atom number.
    atom3 : int
        Third atom number.

    Returns
    -------
    float
        Angle value in degrees.

    """
    v1 = atom1 - atom2
    v2 = atom3 - atom2

    try:
        ang = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
        return ang
    except RuntimeWarning:
        print(v1, v2)

def read_CPprop(file):

    with open(file,'r') as f:
        raw_text = f.read()

    BCPs = []
    pattern = r"----------------\s+CP\s+\d+,\s+Type\s+\(3,\s*-1\)\s+----------------.*?(?=----------------\s+CP|\Z)"
    text_BCPs = re.findall(pattern, raw_text, re.DOTALL)

    for bcp in text_BCPs:
        BCP = dict()

        # coordinates (Angs)
        coords = re.search(r'Position \(Angstrom\).*?\n', bcp).group()
        coords = np.array(coords.split()[2:]).astype(float)
        BCP['coordinates'] = coords

        # Density of electrons
        Dens = re.search(r'User-defined real space function.*?\n', bcp).group()
        Dens = float(Dens.split()[4])
        BCP['ED'] = Dens

        #Laplacian of density
        Lapl_dens = re.search(r'Total.*?\n', bcp).group()
        Lapl_dens = float(Lapl_dens.split()[1])
        BCP['Laplacian_density'] = Lapl_dens

        # eigenvalues of Laplacian
        Lapl_dens = re.findall(r'Components of Laplacian.*?Total', bcp, re.DOTALL)[0]
        Lapl_dens = np.array(Lapl_dens.split()[6:9]).astype(float)
        BCP['Laplacian_eigenvalues'] = Lapl_dens

        BCPs.append(BCP)

    return BCPs

def read_gjf(file_gjf):

    with open(file_gjf,'r') as f:

        mol = read_gaussian_in(f)

    return mol

def what_NCI(BCPs, mol_geometry):

    symbols = mol_geometry.get_chemical_symbols()
    positions = mol_geometry.get_positions()

    for bcp in BCPs:
        distances = []
        indexes = []

        if 0.001 <= bcp['ED'] <= 0.05 and 0.02 <= bcp['Laplacian_density'] <= 0.15:
            bcp['is_NCI'] = True

            for index, position in enumerate(positions):
                dist = np.linalg.norm(bcp['coordinates'] - position)
                distances.append(dist)
                indexes.append(index)

            sorted_pairs = sorted(zip(distances, indexes, symbols))
            distances_sorted, indexes_sorted, symboles_sorted = zip(*sorted_pairs)
            bcp['NCI_type'] = symboles_sorted[0] + symboles_sorted[1]
            bcp['dist_DA'] = np.linalg.norm(positions[indexes_sorted[0]] - positions[indexes_sorted[1]])

            symbols = mol_geometry.get_chemical_symbols()
            if symboles_sorted[0] == 'H' and symboles_sorted[1] != 'H':
                bcp['dist_HA'] = np.linalg.norm(positions[indexes_sorted[0]] - positions[indexes_sorted[1]])
                bcp['H_index'] = indexes_sorted[0] + 1
                bcp['A_index'] = indexes_sorted[1] + 1
                distances_from_H = []
                i_H = indexes_sorted[0]
                for position in positions:
                    dist_H = np.linalg.norm(positions[i_H] - position)
                    distances_from_H.append(dist_H)

                sorted_pairs = sorted(zip(distances_from_H, indexes, symbols))
                distances_sorted_H, indexes_sorted_H, symboles_sorted_H = zip(*sorted_pairs)
                bcp['NCI_type'] = symboles_sorted_H[1] + 'H' + symboles_sorted[1]
                bcp['dist_DH'] = np.linalg.norm(positions[indexes_sorted_H[1]] - positions[indexes_sorted[0]])
                bcp['dist_DA'] = np.linalg.norm(positions[indexes_sorted_H[1]] - positions[indexes_sorted[1]])
                bcp['angle_DHA'] = angle(positions[indexes_sorted_H[1]], positions[indexes_sorted[0]], positions[indexes_sorted[1]])
                bcp['D_index'] = indexes_sorted_H[1] + 1

            elif symboles_sorted[1] == 'H' and symboles_sorted[0] != 'H':
                bcp['dist_HA'] = np.linalg.norm(positions[indexes_sorted[0]] - positions[indexes_sorted[1]])
                bcp['H_index'] = indexes_sorted[1] + 1
                bcp['A_index'] = indexes_sorted[0] + 1
                distances_from_H = []
                i_H = indexes_sorted[1]
                for position in positions:
                    dist_H = np.linalg.norm(positions[i_H] - position)
                    distances_from_H.append(dist_H)

                sorted_pairs = sorted(zip(distances_from_H, indexes, symbols))
                distances_sorted_H, indexes_sorted_H, symboles_sorted_H = zip(*sorted_pairs)
                bcp['NCI_type'] = symboles_sorted_H[1] + 'H' + symboles_sorted[0]
                bcp['dist_DH'] = np.linalg.norm(positions[indexes_sorted_H[1]] - positions[indexes_sorted[1]])
                bcp['dist_DA'] = np.linalg.norm(positions[indexes_sorted_H[1]] - positions[indexes_sorted[0]])
                bcp['angle_DHA'] = angle(positions[indexes_sorted_H[1]], positions[indexes_sorted[1]], positions[indexes_sorted[0]])
                bcp['D_index'] = indexes_sorted_H[1] + 1


            else:
                bcp['NCI_type'] = symboles_sorted[0] + symboles_sorted[1]
                bcp['D_index'] = indexes_sorted[0] + 1
                bcp['H_index'] = 'None'
                bcp['A_index'] = indexes_sorted[1] + 1

            if bcp['D_index'] == bcp['A_index'] or abs(bcp['D_index'] - bcp['A_index']) == 1:
                bcp['is_NCI'] = False
            else:
                continue

        else:
            bcp['is_NCI'] = False
            bcp['NCI_type'] = None

    return BCPs

def HB_energy_Rozenberg(bcp):
    # Common correlation from M. Rozenberg, RSC Advances, 2014, 4(51), 26928 DOI: 10.1039/c4ra03889d publication

    energy = (-6.6 + 1215 * bcp['ED']) / 4.184
    return energy

def HB_enthalpy_Rozenberg_geom(bcp):
    # Common correlation from M. Rozenberg, PCCP, 2000, 2(12), 2699–2702. doi:10.1039/b002216k

    if 'HO' in bcp['NCI_type']:
        energy = 0.134 * (bcp['dist_HA'] / 10) ** -3.05 / 4.184
    else:
        energy = 0
    return energy

def HB_energy_Afonin_linear_ED(bcp):
    # General correlation Andrei V.Afonin, 2024, J. Mol Model., 30(1), 18. DOI: 10.1007/s00894-023-05811-1
    #

    energy = 192 * bcp['ED'] - 0.7
    return energy

def HB_energy_Afonin_linear_LaplED(bcp):

    energy = 70.5 * bcp['Laplacian_density'] - 2.38
    return energy

def HB_energy_Afonin_quadratic_LaplED(bcp):

    energy = 573.7 * bcp['Laplacian_density'] ** 2 - 59.9 * bcp['Laplacian_density'] + 4.24
    return energy

def HB_energy_Emamian_general(bcp):
    # Common correlation S. Emamian, 2019, J. Comp. Chem., 40(32), 2868–2881. doi:10.1002/jcc.2606
    # for entire set of complexes: R^2 = 0.9716, Num = 42

    energy = -357.73 * bcp['ED'] + 2.6182
    return -energy

def HB_energy_Nikolaienko(bcp):
    # T. Nikolaienko, 2012, PCCP, 14(20), 7441–0. doi:10.1039/c2cp40176b
    # Differentiate by hydrogen bond type

    if bcp['NCI_type'] == 'OHO':
        # R^2 = 0.93, std = 0.69, Num = 1949
        energy = 239 * bcp['ED'] + -3.09

    elif bcp['NCI_type'] == 'OHN':
        # R^2 = 0.97, std = 0.50, Num = 269
        energy = 142 * bcp['ED'] + 1.72

    elif bcp['NCI_type'] == 'NHO':
        # R^2 = 0.85, std = 0.76, Num = 150
        energy = 225 * bcp['ED'] + -2.03

    elif bcp['NCI_type'] == 'OHC':
        # R^2 = 0.86, std = 0.35, Num = 81
        energy = 288 * bcp['ED'] + -0.29

    else:
        energy = 0

    return energy

def HB_energy_Mata_ED(bcp):
    # General correlation by I. Mata et al., 2011, Chem. Phys. Lett., 507, 185–189. DOI: 10.1016/j.cplett.2011.03.055
    # R^2 = 0.980, Num = 37

    energy = 186 * (bcp['ED'] / bohr_to_angstrom ** 3) - 2.3
    return energy / 4.184

def HB_energy_Mata_LaplED(bcp):
    # General correlation by I. Mata et al., 2011, Chem. Phys. Lett., 507, 185–189. DOI: 10.1016/j.cplett.2011.03.055
    # R^2 = 0.992, Num = 37

    energy = 2.52 * (bcp['Laplacian_density'] / bohr_to_angstrom ** 5) ** 2 + 5.2
    return energy / 4.184


file = r"C:\Users\User\Navuka\Proteins_NCI_analysis\SCF for manual\H_optimized\GFN2-xTB\SCF\HF\CPprop_1ubq_HF-pcseg-1_opt_xtb_250_grid.txt"
file_gjf = r"C:\Users\User\Navuka\Proteins_NCI_analysis\SCF for manual\H_optimized\GFN2-xTB\SCF\HF\1ubq_HF-6-31Gd_opt_xtb.gjf"

u = read_CPprop(file)
geom = read_gjf(file_gjf)
BCPs = what_NCI(u, geom)


header_columns = [' Type_NCI', 'Index H', 'Index D','Index A', 'dist_DH','dist_HA', 'dist_DA', 'angle_DHA', 'Electron_density', 'Laplacian_of_ED', 'Enthalpy_Rozenberg_geom',
                  'Rozenberg', 'Afonin_linear_ED', 'Afonin_linear_LaplED', 'Emamian', 'Nikolaienko', 'Mata_ED',
                  'Afonin_Quadratic_on_Lapl_of_ED', 'Mata_LaplED']
column_widths = [len(header_columns[i]) for i in range(len(header_columns))]

formatted_header = "".join(
    f"{header_columns[i].center(column_widths[i] + 3)}"
    for i in range(len(header_columns))
)

print(formatted_header)

for bcp in BCPs:
    ed = round(bcp['ED'], 4)
    lapl_ed = round(bcp['Laplacian_density'], 4)
    type = bcp['NCI_type']

    if bcp['is_NCI'] and bcp['NCI_type'] != 'HH' and 'H' in bcp['NCI_type'] and bcp['angle_DHA'] > 120:
        e0 = round(HB_enthalpy_Rozenberg_geom(bcp), 3)
        e1 = round(HB_energy_Emamian_general(bcp), 3)
        e2 = round(HB_energy_Nikolaienko(bcp), 3)
        e3 = round(HB_energy_Mata_ED(bcp), 3)
        e4 = round(HB_energy_Afonin_quadratic_LaplED(bcp), 3)
        e5 = round(HB_energy_Mata_LaplED(bcp), 3)
        e6 = round(HB_energy_Rozenberg(bcp), 3)
        e7 = round(HB_energy_Afonin_linear_ED(bcp), 3)
        e8 = round(HB_energy_Afonin_linear_LaplED(bcp), 3)

        d1 = round(bcp['dist_DH'], 3)
        d2 = round(bcp['dist_HA'], 3)
        d3 = round(bcp['dist_DA'], 3)
        a1 = round(bcp['angle_DHA'], 1)

        ind_H = bcp['H_index']
        ind_D = bcp['D_index']
        ind_A = bcp['A_index']

        data_values = [type, ind_H, ind_D, ind_A, d1, d2, d3, a1, ed, lapl_ed, e0, e6, e7, e8, e1, e2, e3, e4, e5]
        formatted_data = "".join(
            f"{str(data_values[i]).center(column_widths[i] + 2)} "
            for i in range(len(data_values))
        )

        print(formatted_data)

    elif bcp['is_NCI'] and not ('H' in bcp['NCI_type']):
        e0 = 0
        e1 = 0
        e2 = 0
        e3 = 0
        e4 = 0
        e5 = 0
        e6 = 0
        e7 = 0
        e8 = 0

        d1 = 0
        d2 = 0
        d3 = round(bcp['dist_DA'], 3)
        a1 = 0

        ind_H = bcp['H_index']
        ind_D = bcp['D_index']
        ind_A = bcp['A_index']

        data_values = [type, ind_H, ind_D, ind_A, d1, d2, d3, a1, ed, lapl_ed, e0, e6, e7, e8, e1, e2, e3, e4, e5]
        formatted_data = "".join(
            f"{str(data_values[i]).center(column_widths[i] + 2)} "
            for i in range(len(data_values))
        )

       # print(formatted_data)

    else:
        continue
"""
header_columns = [' Type_NCI', 'Index D','Index A', 'dist_DA', 'ED', 'LaplED']
column_widths = [len(header_columns[i]) for i in range(len(header_columns))]

formatted_header = "".join(
    f"{header_columns[i].center(column_widths[i] + 3)}"
    for i in range(len(header_columns))
)

print(formatted_header)

for bcp in BCPs:
    type = bcp['NCI_type']

    if bcp['is_NCI'] and not ('H' in bcp['NCI_type']) and bcp['dist_DA'] > 1.5:
        ind_D = bcp['D_index']
        ind_A = bcp['A_index']
        d1 = round(bcp['dist_DA'], 3)

        data_values = [type, ind_D, ind_A, d1, bcp['ED'], bcp['Laplacian_density']]
        formatted_data = "".join(
            f"{str(data_values[i]).center(column_widths[i] + 2)} "
            for i in range(len(data_values))
        )

        print(formatted_data)

    else:
        continue

"""

