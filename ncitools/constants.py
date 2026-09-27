"""
Atomic radii and conversion constants for non‑covalent interaction analysis.

RADII      : Covalent radii (Cambridge Structural Database)
BONDI      : van der Waals radii (Bondi, with H from Rowland & Taylor)
HALOGENS   : ionic anions halogens radii
CHARRY_TKATCHENKO : vdw radii of Charry and Tkatchenko
METALS     : Ionic radii for metals only
HOMA_PARAMS: HOMA parameters for aromaticity identification
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

BONDI = {'H': 1.09, 'He': 1.40, 'Li': 1.82, 'Be': 2.00, 'B': 2.00, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
        'Al': 2.51, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'As': 1.85, 'Sb': 2.12, 'Se': 1.90, 'Br': 1.85,
         'Kr': 2.02, 'I': 1.98}

HALOGENS = {'H': 0.45, 'F': 1.33, 'Cl': 1.81,'Br': 1.96,'I': 2.20}

CHARRY_TKATCHENKO = {
    'H': 1.67,  'He': 1.41,
    'Li': 2.80, 'Be': 2.27, 'B': 2.08,  'C': 1.91,  'N': 1.80,  'O': 1.71,  'F': 1.63,  'Ne': 1.55,
    'Na': 2.80, 'Mg': 2.48, 'Al': 2.41, 'Si': 2.27, 'P': 2.14,  'S': 2.06,  'Cl': 1.98, 'Ar': 1.91,
    'K': 3.04,  'Ca': 2.79, 'Sc': 2.60, 'Ti': 2.61, 'V': 2.56,  'Cr': 2.54, 'Mn': 2.47, 'Fe': 2.44,
    'Co': 2.39, 'Ni': 2.36, 'Cu': 2.34, 'Zn': 2.28, 'Ga': 2.36, 'Ge': 2.29, 'As': 2.20, 'Se': 2.19,
    'Br': 2.09, 'Kr': 2.02,
    'Rb': 3.08, 'Sr': 2.87, 'Y': 2.79, 'Zr': 2.65, 'Nb': 2.60, 'Mo': 2.56, 'Tc': 2.52, 'Ru': 2.49,
    'Rh': 2.46, 'Pd': 2.15, 'Ag': 2.39, 'Cd': 2.33, 'In': 2.45, 'Sn': 2.38, 'Sb': 2.31, 'Te': 2.27,
    'I': 2.23, 'Xe': 2.17,
    'Cs': 3.18, 'Ba': 3.01,
    'La': 2.91, 'Ce': 2.89, 'Pr': 2.91, 'Nd': 2.90, 'Pm': 2.88, 'Sm': 2.86, 'Eu': 2.85, 'Gd': 2.78,
    'Tb': 2.81, 'Dy': 2.80, 'Ho': 2.78, 'Er': 2.76, 'Tm': 2.75, 'Yb': 2.73, 'Lu': 2.73,
    'Hf': 2.62, 'Ta': 2.50, 'W': 2.47, 'Re': 2.44, 'Os': 2.41, 'Ir': 2.39, 'Pt': 2.35, 'Au': 2.25,
    'Hg': 2.23, 'Tl': 2.36, 'Pb': 2.34, 'Bi': 2.35, 'Po': 2.32, 'At': 2.30, 'Rn': 2.24,
    'Fr': 3.08, 'Ra': 2.97,
    'Ac': 2.89, 'Th': 2.92, 'Pa': 2.77, 'U': 2.70, 'Np': 2.77, 'Pu': 2.71, 'Am': 2.71, 'Cm': 2.75,
    'Bk': 2.69, 'Cf': 2.68, 'Es': 2.67, 'Fm': 2.66, 'Md': 2.64, 'No': 2.64, 'Lr': 3.08,
    'Rf': 2.65, 'Db': 2.30, 'Sg': 2.29, 'Bh': 2.27, 'Hs': 2.25, 'Mt': 2.24, 'Ds': 2.22, 'Rg': 2.22,
    'Cn': 2.17, 'Nh': 2.19, 'Fl': 2.21, 'Mc': 2.48, 'Ts': 2.51, 'Og': 2.41,
}

BONDI_num = {
    1: 1.09,    # H
    2: 1.40,    # He
    3: 1.82,    # Li
    4: 2.00,    # Be
    5: 2.00,    # B
    6: 1.70,    # C
    7: 1.55,    # N
    8: 1.52,    # O
    9: 1.47,    # F
    10: 1.54,   # Ne
    13: 2.51,   # Al
    14: 2.10,   # Si
    15: 1.80,   # P
    16: 1.80,   # S
    17: 1.75,   # Cl
    18: 1.88,   # Ar
    33: 1.85,   # As
    51: 2.12,   # Sb
    34: 1.90,   # Se
    35: 1.85,   # Br
    36: 2.02,   # Kr
    53: 1.98    # I
}

# --------------------------------------------------------------------------- #
#  Metal ionic radii
# --------------------------------------------------------------------------- #
# Values are representative ionic radii in Å. For elements with several
# common oxidation states, a commonly occurring state is used.
# For uncommon/heavy elements without a useful default, a generic value
# is assigned so that the element can still be processed.
METALS = {
    # Alkali metals
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Fr": 1.80,

    # Alkaline-earth metals
    "Be": 0.59, "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35, "Ra": 1.48,

    # Transition metals
    "Sc": 0.75, "Ti": 0.79, "V": 0.79, "Cr": 0.80, "Mn": 0.83, "Fe": 0.78, "Co": 0.75, "Ni": 0.69, "Cu": 0.73, "Zn": 0.74,
    "Y": 0.90, "Zr": 0.72, "Nb": 0.74, "Mo": 0.73, "Tc": 0.75, "Ru": 0.68, "Rh": 0.67, "Pd": 0.86, "Ag": 1.15, "Cd": 0.95,
    "Hf": 0.71, "Ta": 0.72,  "W": 0.66, "Re": 0.63, "Os": 0.63, "Ir": 0.68, "Pt": 0.80, "Au": 1.37, "Hg": 1.02,

    # Post-transition metals
    "Al": 0.54,  "Ga": 0.62,  "In": 0.80, "Tl": 1.02, "Sn": 0.83, "Pb": 1.19, "Bi": 1.03, "Po": 0.94,

    # Lanthanides
    "La": 1.03, "Ce": 1.01, "Pr": 0.99, "Nd": 0.98, "Pm": 0.97, "Sm": 0.96, "Eu": 1.17, "Gd": 0.94, "Tb": 0.92,
    "Dy": 0.91, "Ho": 0.90, "Er": 0.89, "Tm": 0.88, "Yb": 0.87, "Lu": 0.86,

    # Actinides
    "Ac": 1.12, "Th": 1.05, "Pa": 1.04, "U": 1.03, "Np": 1.01, "Pu": 1.00, "Am": 0.98, "Cm": 0.97, "Bk": 0.96,
    "Cf": 0.95, "Es": 0.94, "Fm": 0.93, "Md": 0.92, "No": 0.91, "Lr": 0.90,

    # Superheavy metals
    "Rf": 0.80, "Db": 0.80, "Sg": 0.80, "Bh": 0.80, "Hs": 0.80, "Mt": 0.80, "Ds": 0.80, "Rg": 0.80, "Cn": 0.80,
    "Nh": 0.80, "Fl": 0.80, "Mc": 0.80, "Lv": 0.80,}


# --------------------------------------------------------------------------- #
#  HOMA parameters
# --------------------------------------------------------------------------- #
# Ropt is the optimal bond length in Å and alpha is the HOMA normalization
# coefficient in Å^-2.
#
# The main parameters follow the extended HOMA93 parametrization.
# CSe and NS are included to improve coverage of heavier heterocycles.
HOMA_PARAMS = {
    "CC":  (1.388, 257.70),
    "CN":  (1.334,  93.52),
    "CO":  (1.265, 157.38),
    "CP":  (1.698, 118.91),
    "CS":  (1.677,  94.09),
    "CSe": (1.8217, 84.9144),
    "NN":  (1.309, 130.33),
    "NO":  (1.248,  57.21),
    "NS":  (1.616,  71.875),
}

bohr_to_angstrom = 0.529177249