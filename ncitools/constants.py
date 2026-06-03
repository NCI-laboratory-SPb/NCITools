"""
Atomic radii and conversion constants for non‑covalent interaction analysis.

RADII      : Covalent radii (Cambridge Structural Database)
BONDI      : van der Waals radii (Bondi, with H from Rowland & Taylor)
bohr_to_angstrom : Conversion factor 0.52918 Å/Bohr
"""

RADII = {'H': 0.32, 'He': 0.93, 'Li': 1.23, 'Be': 0.90, 'B': 0.82, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.72,
         'Ne': 0.71, 'Na': 1.54, 'Mg': 1.36, 'Al': 1.18, 'Si': 1.11, 'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.98,
         'K': 2.03, 'Ca': 1.74, 'Sc': 1.44, 'Ti': 1.32, 'V': 1.22, 'Cr': 1.18, 'Mn': 1.17, 'Fe': 1.17, 'Co': 1.16,
         'Ni': 1.15, 'Cu': 1.17, 'Zn': 1.25, 'Ga': 1.26, 'Ge': 1.22, 'As': 1.20, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.12,
         'Rb': 2.16, 'Sr': 1.91, 'Y': 1.62, 'Zr': 1.45, 'Nb': 1.34, 'Mo': 1.30, 'Tc': 1.27, 'Ru': 1.25, 'Rh': 1.25,
         'Pd': 1.28, 'Ag': 1.34, 'Cd': 1.48, 'In': 1.44, 'Sn': 1.41, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
         'Cs': 2.35, 'Ba': 1.98, 'La': 1.69, 'Lu': 1.60, 'Hf': 1.44, 'Ta': 1.34, 'W': 1.30, 'Re': 1.28, 'Os': 1.26,
         'Ir': 1.27, 'Pt': 1.30, 'Au': 1.34, 'Hg': 1.49, 'Tl': 1.48, 'Pb': 1.47, 'Bi': 1.46, 'X': 0}

BONDI = {'H': 1.09, 'He': 1.40, 'Li': 1.82, 'Be': 2.00, 'B': 2.00, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
         'Ne': 1.54, 'P': 1.80, 'S': 1.80, 'Se': 1.90}

bohr_to_angstrom = 0.529177249