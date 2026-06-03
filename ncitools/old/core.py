import os
import numpy as np
import pandas as pd
from datetime import datetime
from progress.bar import IncrementalBar
from ase.io import read
from ase.io.proteindatabank import read_proteindatabank
import hydrogens

# Bondi van der Waals radii for all atoms from: Bondi, A. J. Phys. Chem. 1964, 68, 441-452,
# except hydrogen, which is taken from Rowland, R. S.; Taylor, R. J. Phys. Chem. 1996, 100, 7384-7391.,
# Radii unavailable in either of these publications are set to 2 Angstrom
BONDI = {'H': 1.09, 'He': 1.40, 'Li': 1.82, 'Be': 2.00, 'B': 2.00, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
         'Ne': 1.54, 'P': 1.80, 'S': 1.80, 'Se': 1.90, 'Pt': 1.75}

bohr_to_angstrom = 0.52918

def read_input(file):
    filename = os.path.basename(file).split('.')[0]
    abs_dir = os.path.dirname(file)
    ext = os.path.splitext(file)[1]

    if ext == ".pdb":
        pdb_file = read_proteindatabank(file)
        return filename, ext, abs_dir, pdb_file

    else:
        in_file = read(file)

    return filename, ext, abs_dir, in_file


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
    v1 = np.array(atom1) - np.array(atom2)
    v2 = np.array(atom3) - np.array(atom2)

    ang = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
    return ang

def distance(atom1, atom2):
    dist = np.linalg.norm(np.array(atom1) - np.array(atom2))
    return dist

def hb_energy(hb):
    # Common correlation from M. Rozenberg, PCCP, 2000, 2(12), 2699–2702. doi:10.1039/b002216k

    if 'HO' in hb['Type']:
        energy = 0.134 * (hb['dist_HA'] / 10) ** -3.05 / 4.184
    else:
        energy = 0
    return energy

def find_HB(positions, symboles):
    molecule = pd.DataFrame(data = {'symboles': symboles, 'positions': list(positions)}, dtype=object)

    hydrogens = molecule[molecule['symboles'] == 'H']
    hydrogens_indexes = hydrogens.index.values
    non_hydrogens = molecule[molecule['symboles'] != 'H']
    hydrogen_bonds = []

    bar = IncrementalBar('Searching for each hydrogen', max=len(hydrogens_indexes))

    for index_h, coord_h in enumerate(hydrogens['positions']):
        bar.next()
        distances = []
        indexes = []

        for index_atom, atom in non_hydrogens.iterrows():

            dist = distance(coord_h, atom['positions'])

            a_symb = atom['symboles']
            if 0.9 <= dist <= (BONDI['H'] + BONDI[f'{a_symb}']):
                distances.append(dist)
                indexes.append(index_atom)
            else:
                continue

        if len(distances) == 0:  # Зона с тритментом свободных H, которые возникает из-за PBC в cif
            continue

        sorted_pairs = sorted(zip(distances, indexes))
        distances_sorted, indexes_sorted = zip(*sorted_pairs)

        if len(distances_sorted) > 1:
            index_d = indexes_sorted[0]
            coord_d = molecule['positions'][index_d]

            for index, atom in enumerate(indexes_sorted[1:]):
                if non_hydrogens['symboles'][atom] == 'C':
                    continue
                hb = {}
                coord_a = molecule['positions'][atom]
                hb_angle = angle(coord_d, coord_h, coord_a)

                if hb_angle > 120:
                    hb['Type'] = non_hydrogens['symboles'][index_d] + 'H' + non_hydrogens['symboles'][atom]
                    hb['Index_H'] = hydrogens_indexes[index_h] + 1
                    hb['Index_D'] = index_d + 1
                    hb['Index_A'] = atom + 1
                    hb['dist_DH'] = np.round(distances_sorted[0], 4)
                    hb['dist_HA'] = np.round(distances_sorted[index + 1], 4)
                    hb['dist_DA'] = np.round(distance(coord_d, coord_a), 4)
                    hb['angle_DHA'] = np.round(hb_angle, 1)
                    hb['Energy'] = np.round(hb_energy(hb), 2)

                    hydrogen_bonds.append(hb)

        else:
            break

    bar.finish()
    return hydrogen_bonds

header_columns = ['Type of NCI', 'Index H', 'Index D','Index A', 'dist DH','dist HA', 'dist DA', 'angle DHA', 'Energy']
column_widths = [len(header_columns[i]) for i in range(len(header_columns))]
formatted_header = "".join(
    f"{header_columns[i].center(column_widths[i] + 3)}"
    for i in range(len(header_columns))
)

def output(hb_data, filename='default', ext='.default'):
    now = datetime.now()

    with open(f'{filename}.nci', 'w') as f:
        f.write('NCITools: hydrogen bonds finding based on geometry.' + 10*' ' + f'Date: {now}' + '\n')
        f.write(f'Uploaded file: {filename}{ext}' + '\n')
        f.write('\n')
        f.write(formatted_header + '\n')

        if len(hb_data) == 0:
            f.write('There is no hydrogen bonds to describe!\n')
            f.write('Buy developer a coffee!')
        else:
            for hb in hb_data:
                data_values = [hb['Type'], hb['Index_H'], hb['Index_D'], hb['Index_A'],
                               hb['dist_DH'], hb['dist_HA'], hb['dist_DA'], hb['angle_DHA'], hb['Energy']]

                formatted_data = "".join(
                    f"{str(data_values[i]).center(column_widths[i] + 2)} "
                    for i in range(len(data_values))
                )
                f.write(formatted_data)
                f.write('\n')
            f.write('Buy developer a coffee!')

def analyze_structure(file_path):
    filename, ext, abs_dir, struct = read_input(file_path)

    pos = struct.get_positions()
    sym = struct.get_chemical_symbols()

    hbs = find_HB(pos, sym)

    return {
        "filename": filename,
        "extension": ext,
        "n_hbonds": len(hbs),
        "hbonds": hbs
    }


