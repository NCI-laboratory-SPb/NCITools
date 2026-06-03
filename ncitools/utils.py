"""Utility functions for file I/O, geometry, and element symbols."""

import os
import numpy as np
from ase.io import read
from ase.io.proteindatabank import read_proteindatabank

def read_input(file):
    """Read molecular structure file (PDB or other ASE formats).
    Returns (filename, extension, directory, ASE Atoms object).
    """

    filename = os.path.basename(file).split('.')[0]
    abs_dir = os.path.dirname(file)
    ext = os.path.splitext(file)[1]

    if ext == ".pdb":
        pdb_file = read_proteindatabank(file)
        return filename, ext, abs_dir, pdb_file
    else:
        in_file = read(file)
        return filename, ext, abs_dir, in_file

#Delete in future versions due to ASE module
def angle(atom1, atom2, atom3):
    """Angle (degrees) between three points (each as [x,y,z])."""

    v1 = np.array(atom1) - np.array(atom2)
    v2 = np.array(atom3) - np.array(atom2)
    ang = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
    return ang

#Delete in future versions due to ASE module
def distance(atom1, atom2):
    """Euclidean distance between two points."""

    dist = np.linalg.norm(np.array(atom1) - np.array(atom2))
    return dist

# Delete in future versions due to correlations.py module
def hb_energy(hb):
    """Compute H‑bond energy from distance using Rozenberg 2000.
    hb dict must contain 'dist_HA' (Å) and 'Type' (e.g. 'NHO').
    """
    if 'NHO' in hb['Type'] or 'OHO' in hb['Type']:
        energy = 0.134 * (hb['dist_HA'] / 10) ** -3.05 / 4.184
    else:
        energy = 0
    return energy

def get_symbol(z):
    """Return element symbol for atomic number Z (common elements)."""

    symbols = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 35: 'Br', 53: 'I'
    }
    return symbols.get(z, 'X')
