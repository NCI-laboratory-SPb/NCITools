"""
Geometry‑based detection of hydrogen bonds using covalent + van der Waals radii.
"""

import ase
import numpy as np
import networkx as nx
from tqdm import tqdm
from datetime import datetime
from ase.neighborlist import neighbor_list
from ncitools.constants import RADII, BONDI
from ncitools.correlations import Rozenberg_2000
from ncitools.utils import read_input   # if needed in main

def find_covalent_pairs(atoms: ase.Atoms, radii=RADII, tolerance=0.1) -> list:
    """Return list of (i,j) atom indices that are covalently bonded
    based on sum of covalent radii + tolerance.

    Parameters
    ----------
    atoms : ase.Atoms
        Объект ASE Atoms.

    radii : dict
        Словарь ковалентных радиусов:
        {'H': 0.31, 'C': 0.76, ...}

    tolerance : float
        Допуск к сумме радиусов.

    Returns
    -------
    list[tuple[int, int]]
        Список уникальных пар (i, j), где i < j.
    """
    cutoffs = [radii[s] + tolerance for s in atoms.get_chemical_symbols()]
    i_list, j_list = neighbor_list('ij', atoms, cutoffs)
    pairs = []

    for i, j in zip(i_list, j_list):

        if i >= j:
            continue

        cutoff = (radii[atoms[i].symbol] + radii[atoms[j].symbol]  + tolerance)
        dist = atoms.get_distance(i, j, mic=True)

        if dist <= cutoff:
            pairs.append((i, j))

    return pairs


def build_graph_with_hbs(atoms: ase.Atoms, covalent_pairs: list, radii=BONDI, angle_tol=120) -> nx.classes.graph.Graph:
    """ Build a NetworkX graph with covalent edges and hydrogen‑bond edges (NCI).
    H‑bonds are identified when H···A distance ≤ sum of van der Waals radii
    and D–H···A angle ≥ angle_tol.

    Parameters
    ----------
    atoms : ase.Atoms
        Объект ASE Atoms.

    covalent_pairs: list
        Лист кортежей из уникальных ковалентно связанных пар

    radii : dict
        Словарь радиусов по Бонди:

    angle_tol : float
        Допуск по углам.

    Returns
    -------
    nx.classes.graph.Graph
        Граф связанных атомов.
    """

    chemical_symbols = atoms.get_chemical_symbols()
    hydrogen_indices = [index for (index, sym) in enumerate(chemical_symbols) if sym == 'H']
    heavy_atoms_indices = [index for (index, sym) in enumerate(chemical_symbols) if sym != 'H']
    n_atoms = len(atoms)

    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))

    for i, j in covalent_pairs:
        G.add_edge(i, j, bond_type='covalent')

    for h in tqdm(hydrogen_indices, desc='Checking every hydrogen atom'):
        d = list(G.adj[h])[0]
        el_d = chemical_symbols[d]

        distances_with_h = atoms.get_distances(h, heavy_atoms_indices, mic=True)
        zipped = zip(distances_with_h, heavy_atoms_indices)
        zipped_sorted = sorted(zipped, key=lambda x: x[0])
        distances_sorted, indices_ha_sorted = zip(*zipped_sorted)

        for index, dist in enumerate(distances_sorted[1:]):
            a = indices_ha_sorted[index + 1]
            el = chemical_symbols[a]
            if dist <= (BONDI['H'] + BONDI[el]):
                angle = atoms.get_angle(d, h, a, mic=True)
                if angle >= angle_tol:
                    da_dist = atoms.get_distance(a, d, mic=True)
                    cov_dist = atoms.get_distance(h, d, mic=True)
                    G.add_edge(h, a, bond_type='nci', symbol_d=el_d, symbol_a=el,length_hb=dist, length_cov=cov_dist, length_da=da_dist, angle=angle)
            else:
                break

    return G


def output(G: nx.classes.graph.Graph, filename='default': str, ext='.default': str, correlation=Rozenberg_2000):
    """Write geometry‑based NCI results to a .nci file."""

    now = datetime.now()

    header_columns = [
        ' Number', 'Type NCI', 'covalent bond length (D-H), Å',
        'hydrogen bond length (H...A), Å', 'distance between heavy atoms (D-A), Å',
        'hydrogen bond angle (DHA)', 'Energy, kcal/mol'
    ]

    column_widths = [len(header_columns[i]) for i in range(len(header_columns))]

    formatted_header = "".join(
        f"{header_columns[i].center(column_widths[i] + 3)}"
        for i in range(len(header_columns))
    )

    nci_data = [i[1] for i in G.edges.items() if i[1]['bond_type'] == 'nci']

    with open(f'{filename}.nci', 'w', encoding='utf-8') as f:
        f.write('NCITools: geometry-based analysis.' + 10 * ' ' + f'Date: {now}' + '\n')
        f.write(f'Uploaded file: {filename}{ext}' + '\n')
        f.write('\n')
        f.write(formatted_header + '\n')
        if len(nci_data) == 0:
            f.write('There is no hydrogen bonds to describe!\n')
            f.write('Buy developer a coffee!')
        else:
            for index, hb in enumerate(nci_data):
                D = hb['symbol_d']
                A = hb['symbol_a']
                bond_type = f'{D}-H...{A}'
                Energy = np.float64(correlation(hb['length_hb'], bond_type))
                data_values = [str(index + 1), bond_type, hb['length_cov'].round(5),
                               hb['length_hb'].round(5), hb['length_da'].round(5),
                               hb['angle'].round(3), Energy.round(3)]
                formatted_data = "".join(
                    f"{str(data_values[i]).center(column_widths[i] + 2)} "
                    for i in range(len(data_values))
                )
                f.write(formatted_data)
                f.write('\n')
            f.write('Buy developer a coffee!')


def main():
    file = r"C:\Users\User\PycharmProjects\NCITools\ncitools\tests\1IU6_conf_A_protonated_PyMOL.pdb"
    _, _, _, atoms = read_input(file)
    print(atoms)
    cov_p = find_covalent_pairs(atoms)
    G = build_graph_with_hbs(atoms, cov_p)
    output(G, filename='1IU6_conf_A_protonated_PyMOL')



if __name__ == "__main__":
    main()


