"""
Energy estimation correlations for hydrogen bonds.
All functions return energy in **kcal/mol**.
"""

def Rozenberg_2000(dist, contact_type):
    """Energy from H···A distance (Å) for O–H···O or N–H···O.
    Eq. from Rozenberg, PCCP 2000; returns kcal/mol.
    """

    if 'H...O' in contact_type:
        energy = 0.134 * (dist / 10) ** -3.05 / 4.184
    else:
        energy = 0.
    return energy

def Rozenberg_2014(rho):
    """Energy from electron density ρ (e/bohr³) at BCP.
    Rozenberg, RSC Adv. 2014; returns kcal/mol.
    """

    energy = (-6.6 + 1215 * rho) / 4.184
    return energy

def Mata_ED_2011(rho):
    """Energy from ρ (e/bohr³) – general correlation.
    Mata et al., Chem. Phys. Lett. 2011; returns kcal/mol.
    """

    energy = 186 * (rho / bohr_to_angstrom ** 3) - 2.3
    return energy / 4.184

def Mata_LaplED_2011(laplacian):
    """Energy from Laplacian ∇²ρ (e/bohr⁵) – general correlation.
    returns kcal/mol.
    """

    energy = 2.52 * (laplacian / bohr_to_angstrom ** 5) ** 2 + 5.2
    return energy / 4.184

def Nikolaienko_2012(rho, contact_type='N-H...O'):
    """Energy from ρ, differentiated by H‑bond type.
    Contact types: O-H...O, O-H...N, N-H...O, O-H...C.
    Returns kcal/mol.
    """

    if contact_type == 'O-H...O':
        # R^2 = 0.93, std = 0.69, Num = 1949
        energy = 239 * rho + -3.09

    elif contact_type == 'O-H...N':
        # R^2 = 0.97, std = 0.50, Num = 269
        energy = 142 * rho + 1.72

    elif contact_type == 'N-H...O':
        # R^2 = 0.85, std = 0.76, Num = 150
        energy = 225 * rho + -2.03

    elif contact_type == 'O-H...C':
        # R^2 = 0.86, std = 0.35, Num = 81
        energy = 288 * rho + -0.29

    else:
        energy = 0

    return energy

def Afonin_ED_2024(rho):
    """Energy from ρ – linear fit (Afonin, J. Mol. Model. 2024)."""

    energy = 192 * rho - 0.7
    return energy

def Afonin_linear_LaplED_2024(laplacian):
    """Energy from ∇²ρ – linear fit (Afonin)."""

    energy = 70.5 * laplacian - 2.38
    return energy

def Afonin_quadratic_LaplED_2024(laplacian):
    """Energy from ∇²ρ – quadratic fit (Afonin)."""

    energy = 573.7 * laplacian ** 2 - 59.9 * laplacian + 4.24
    return energy

def Emamian_2019(rho):
    """Energy from ρ – general correlation (Emamian, J. Comp. Chem. 2019).
    Returns positive kcal/mol.
    """

    energy = -357.73 * rho + 2.6182
    return -energy

def Rudenko_2026_ED(rho, contact_type='C-H...O'):
    """Energy from ρ for C–H···O and C–H···N H‑bonds (NCI lab).
    Returns positive kcal/mol.
    """

    if contact_type=='C-H...O':
        return -323.0 * rho + 0.33

    elif contact_type=='C-H...N':
        return -239.55 * rho + 0.49

    else:
        return None

def correlation_1(rho, contact_type):
    """Default energy estimator: uses Rudenko for C–H···X, else Emamian."""

    if contact_type=='C-H...O' or contact_type=='C-H...N':
        return -Rudenko_2026_ED(rho, contact_type)

    else:
        return Emamian_2019(rho)

