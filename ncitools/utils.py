"""Utility functions for file I/O, geometry, and element symbols."""

import os

import networkx as nx
from .constants import METALS, HALOGENS
from ase import Atoms
from ase.io import read
from ase.io.proteindatabank import read_proteindatabank
from pathlib import Path
import tempfile

def read_input(file):
    """Read molecular structure file (PDB or other ASE formats).
    Returns (filename, extension, directory, ASE Atoms object).
    """

    ext = os.path.splitext(file)[1]

    if ext == ".pdb":
        try:
            atoms = read_proteindatabank(file)
            return atoms
        except KeyError:
            atoms = read_pdb_normalized(file)
            return atoms
    else:
        atoms = read(file)
        return atoms

def read_pdb_normalized(filename):
    filename = Path(filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    fixed_lines = []

    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 78:
            element = line[76:78].strip().upper()

            if element == "D":
                line = line[:76] + " H" + line[78:]

        fixed_lines.append(line)

    # Temporal copy of loaded PDB
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pdb",
        delete=False
    ) as tmp:
        tmp.writelines(fixed_lines)
        tmp_path = Path(tmp.name)

    try:
        atoms = read_proteindatabank(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return atoms



def check_atomic_connectivity(atoms: Atoms, G: nx.Graph) -> list[str]:
    """Warn if non-noble-gas nonmetals have unusual connectivity."""

    warnings = []

    allowed_connectivity = {
        "H": {1},
        "C": {1, 2, 3, 4},
        "N": {1, 2, 3, 4},
        "O": {1, 2, 3},

        # Halogens
        "F": {0, 1},
        "Cl": {0, 1, 2, 3, 4, 5, 6},
        "Br": {0, 1, 2, 3, 4, 5, 6},
        "I": {0, 1, 2, 3, 4, 5, 6, 7},
    }

    for atom_idx, atom in enumerate(atoms):
        symbol = atom.symbol

        count = sum(
            1
            for nbr in G.neighbors(atom_idx)
            if not isinstance(nbr, str)
            and G.edges[atom_idx, nbr].get("bond_type")
            in ("covalent", "coordination")
        )

        # All other nonmetals must have at least one neighbour
        if count == 0 and not (symbol in METALS or symbol in HALOGENS):
            message = (
                f"WARNING: {symbol} atom {atom_idx + 1} "
                "has no covalent or coordination neighbours."
            )
            warnings.append(message)
            continue

        # Check typical connectivity for selected elements
        if symbol in allowed_connectivity:
            if count not in allowed_connectivity[symbol]:
                message = (
                    f"WARNING: {symbol} atom {atom_idx + 1} "
                    f"has an unusual number of covalent and/or "
                    f"coordination neighbours: {count}."
                )
                warnings.append(message)

    return warnings



def get_symbol(z):
    """Return element symbol for atomic number Z (common elements)."""

    symbols = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
        58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
        93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',
        100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db',
        106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg',
        112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts',
        118: 'Og'
    }

    return symbols.get(z, 'X')
