"""
Energy estimation correlations from geometry.
All functions return energy in kcal/mol.
"""
import numpy as np
from ncitools.constants import BOHR_TO_ANGSTROM


##################
# HYDROGEN BONDS #
##################

def hb_1(r, contact_type):
    """Rozenberg, M., Loewenschuss, A., & Marcus, Y. (2000).
    Phys. Chem. Chem. Phys., 2(12), 2699-2702.
    """

    if 'H...O' in contact_type:
        energy = 0.134 * (r / 10) ** -3.05 / 4.184
    else:
        energy = 0
    return np.maximum(0, energy)

def hb_2(r, contact_type):
    """Zou, J. W., Huang, M., Hu, G. X., & Jiang, Y. J. (2017).
    RSC advances, 7(17), 10295-10305. DOI:  10.1039/c6ra27590g
    """

    if 'C-H...N' in contact_type:
        energy = -12.98 * r + 34.86     # R^2 = 0.972
    else:
        energy = 0
    return np.maximum(0, energy)

def hb_3(r, contact_type):
    """Tupikina, E. Y., Sigalov, M., Shenderovich, I. G., Mulloyarova, V. V., Denisov, G. S., & Tolstoy, P. M. (2019).
    J. Chem. Phys., 150(11). DOI: 10.1063/1.5090180
    """

    if 'N-H...N' in contact_type:
        return np.maximum(0, 601.7 * (r ** -6.333)) # R^2 = 0.972
    else:
        return 0

def hb_4(r, contact_type):
    """Kaplanskiy, M. V., & Tupikina, E. Y. (2025).
    J. Comp. Chem., 46(31), e70282. DOI: 10.1002/jcc.70282
    """

    if 'N-H...N' in contact_type:
        return np.maximum(0, 1034.33 * np.exp(-0.0987 * r)) # R^2 = 0.978
    elif 'O-H...O' in contact_type:
        return np.maximum(0, -37.12 * r + 77.16) # R^2 = 0.973
    elif 'O-H...N' in contact_type:
        return np.maximum(0, 292.1 * (r ** -5.563))  # R^2 = 0.988
    else:
        return 0

def hb_5(r, contact_type):
    """Kostin, M. A., Pylaeva, S. A., & Tolstoy, P. M. (2022).
    Phys. Chem. Chem. Phys., 24(11), 7121-7133. DOI: 10.1039/d1cp05939d
    """

    if 'N-H...O' in contact_type:
        return 1740.1 * np.exp(-3.021 * r) # R^2 = 0.977
    elif 'O-H...O' in contact_type:
        return np.maximum(0, -23.6 * r + 48.9) # R^2 = 0.954
    elif 'C-H...O' in contact_type:
        return 0, 2181.6 * np.exp(-3.175 * r) # R^2 = 0.967
    else:
        return 0


def hb_uni_1(r):
    """Wendler, K., Thar, J., Zahn, S., & Kirchner, B. (2010).
    J. Phys. Chem. A, 114(35), 9529-9536. DOI: 10.1021/jp103470e
    """

    return np.maximum(0, 79.0 / (r ** 3.78)) # R^2 = 966, RMSD = 1.31 kcal/mol


#################
# HALOGEN BONDS #
#################

def xb_1(r, contact_type):
    """Zou, J. W., Huang, M., Hu, G. X., & Jiang, Y. J. (2017).
     RSC advances, 7(17), 10295-10305. DOI:  10.1039/c6ra27590g
     """

    if contact_type == 'ClN':
        return np.maximum(0, 75.66 - 39.56 * r + 5.20 * r ** 2)  # R^2 = 0.965
    elif contact_type == 'BrN':
        return np.maximum(0, 121.1 - 67.41 * r + 9.45 * r ** 2)  # R^2 = 0.964
    elif contact_type == 'IN':
        return np.maximum(0, 205.4 - 119.5 * r + 17.63 * r ** 2) # R^2 = 0.960

def xb_2(r, contact_type):
    """Ostras’, A. S., Ivanov, D. M., Novikov, A. S., & Tolstoy, P. M. (2020).
    Molecules, 25(6), 1406. DOI: 10.3390/molecules25061406
    """

    if contact_type == 'FO':
        return np.maximum(0, 1312.1 * np.exp(-2.4 * r))  # R^2 = 0.36
    elif contact_type == 'ClO':
        return np.maximum(0, 351.3 * np.exp(-1.5 * r))  # R^2 = 0.80
    elif contact_type == 'BrO':
        return np.maximum(0, 980.9 * np.exp(-1.8 * r))  # R^2 = 0.83
    elif contact_type == 'IO':
        return np.maximum(0, 3895.3 * np.exp(-2.2 * r)) # R^2 = 0.89
    else:
        return 0


###################
# CHALCOGEN BONDS #
###################

def chb_1(r, contact_type):
    """de Azevedo Santos, L., van Der Lubbe, S. C., Hamlin, T. A., Ramalho, T. C., & Matthias Bickelhaupt, F., (2021).
    ChemistryOpen, 10(4), 391-401. DOI: 10.1002/open.202000323
    """

    if contact_type in ('SF', 'SCl', 'SBr', 'SI'):
        return np.maximum(0, 203.33 * (r ** -2.415))  # R^2 = 0.995, RMSE = 0.94 kcal/mol
    elif contact_type in ('SeF', 'SeCl', 'SeBr', 'SeI'):
        return np.maximum(0, 263.09 * (r ** -2.224))  # R^2 = 0.995, RMSE = 1.98 kcal/mol
    elif contact_type in ('TeF', 'TeCl', 'TeBr', 'TeI'):
        return np.maximum(0, 343.11 * (r ** -2.178))  # R^2 = 0.997, RMSE = 0.76 kcal/mol
    else:
        return 0


###################
# PNICTOGEN BONDS #
###################

def pnb_1(r, contact_type):
    """de Azevedo Santos, L., Hamlin, T. A., Ramalho, T. C., & Bickelhaupt, F. M. (2021).
    Phys. Chem. Chem. Phys., 23(25), 13842-13852. DOI: 10.1039/d1cp01571k
    """

    if contact_type in ('PF', 'PCl', 'PBr', 'PI'):
        return np.maximum(0, 249.36 * (r ** -2.654))  # R^2 = 0.979, RMSE = 3.03 kcal/mol
    elif contact_type in ('SbF', 'SbCl', 'SbBr', 'SbI'):
        return np.maximum(0, 382.43 * (r ** -2.303))  # R^2 = 0.994, RMSE = 1.33 kcal/mol
    elif contact_type in ('NF', 'NCl', 'NBr', 'NI'):
        return np.maximum(0, 61.36 * (r ** -2.374))  # R^2 = 0.952, RMSE = 2.54 kcal/mol
    else:
        return 0


################
# TETREL BONDS #
################

def tetb_1(r, contact_type):
    """
    """

    return 0


#######################
# UNIVERSAL FUNCTIONS #
#######################

def universal_1_hb(r, contact_type):
    """Default HB energy estimator."""

    if 'O-H···O' in contact_type:
        """Energy from H···A distance (Å) for O–H···O or N–H···O.
           Eq. from Rozenberg, PCCP 2000; returns kcal/mol.
        """
        return np.maximum(0, 35.93462544848542 * (r ** -3.05)), hb_1.__doc__

    elif 'N-H...O' in contact_type:
        """ Kostin, M. A., Pylaeva, S. A., & Tolstoy, P. M. (2022).
            PCCP, 24(11), 7121-7133. DOI: 10.1039/d1cp05939d
        """
        return 1740.1 * np.exp(-3.021 * r), hb_5.__doc__ # R^2 = 0.976

    elif 'C-H...O' in contact_type:
        """ Kostin, M. A., Pylaeva, S. A., & Tolstoy, P. M. (2022).
            PCCP, 24(11), 7121-7133. DOI: 10.1039/d1cp05939d
        """
        return 2181.6 * np.exp(-3.175 * r), hb_5.__doc__ # R^2 = 0.976

    elif 'O-H...N' in contact_type:
        """ Kaplanskiy, M. V., & Tupikina, E. Y. (2025).
            J. Comp. Chem., 46(31), e70282. DOI: 10.1002/jcc.70282
        """
        return np.maximum(0, 292.1 * (r ** -5.563)), hb_4.__doc__  # R^2 = 0.988

    elif 'C-H...N' in contact_type:
        """Zou, J. W., Huang, M., Hu, G. X., & Jiang, Y. J. (2017).
         RSC advances, 7(17), 10295-10305. DOI:  10.1039/c6ra27590g"""
        return np.maximum(0, 12.98 * r - 34.86), hb_2.__doc__ # R^2 = 0.972

    elif 'N-H...N' in contact_type:
        """ Tupikina, E. Y., Sigalov, M., Shenderovich, I. G., Mulloyarova, V. V.,
         Denisov, G. S., & Tolstoy, P. M. (2019). J. Chem. Phys., 150(11). DOI: 10.1063/1.5090180
        """
        return np.maximum(0, 601.7 * (r ** -6.333)), hb_3.__doc__ # R^2 = 0.972

    else:
        """Wendler, K., Thar, J., Zahn, S., & Kirchner, B. (2010).
        J. Phys. Chem. A, 114(35), 9529-9536. DOI: 10.1021/jp103470e"""
        return np.maximum(0, 79.0 / (r ** 3.78)), hb_uni_1.__doc__  # R^2 = 966, RMSD = 1.31 kcal/mol;


def universal_1_sb(r, contact_type):
    """Default sigma-bond energy estimator."""

    if contact_type in ('ClN', 'BrN', 'IN'):
        return np.maximum(0, xb_1(r, contact_type)), xb_1.__doc__
    elif contact_type in ('FO', 'ClO', 'BrO', 'IO'):
        return np.maximum(0, xb_2(r, contact_type)), xb_2.__doc__
    elif contact_type in ('SF', 'SCl', 'SBr', 'SI',
                          'SeF', 'SeCl', 'SeBr', 'SeI',
                          'TeF', 'TeCl', 'TeBr', 'TeI'):
        return np.maximum(0, chb_1(r, contact_type)), chb_1.__doc__
    elif contact_type in ('PF', 'PCl', 'PBr', 'PI',
                          'NF', 'NCl', 'NBr', 'NI',
                          'SbF', 'SbCl', 'SbBr', 'SbI'):
        return np.maximum(0, pnb_1(r, contact_type)), pnb_1.__doc__
    else:
        return 0, None
        
        

"""
Energy estimation correlations from electron density.
All functions return energy in kcal/mol.
"""
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

    energy = 186 * (rho / BOHR_TO_ANGSTROM ** 3) - 2.3
    return energy / 4.184

def Mata_LaplED_2011(laplacian):
    """Energy from Laplacian ∇²ρ (e/bohr⁵) – general correlation.
    returns kcal/mol.
    """

    energy = 2.52 * (laplacian / BOHR_TO_ANGSTROM ** 5) ** 2 + 5.2
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

