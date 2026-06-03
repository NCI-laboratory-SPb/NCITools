"""
DFT calculations with PySCF to generate electron density cube files.
"""

import os
import pyscf
from pyscf import gto, dft, tools
from pyscf.lib import num_threads


def write_cube_pyscf(pdb_file, functional='b3lyp', basis='def2-svp',
                     aux_basis='def2-universal-jfit', charge=0, multiplicity=1,
                     ncores=12, cube_name='default.cube'):
    """Run DFT (RKS) with density fitting and write density cube.

    Perform DFT calculation with PySCF and write electron density cube.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file.
    functional : str, optional
        DFT functional (default: 'b3lyp').
    basis : str, optional
        Orbital basis set (default: 'def2-svp').
    aux_basis : str, optional
        Auxiliary basis set for density fitting (default: 'def2-universal-jfit').
    charge : int, optional
        Molecular charge (default: 0).
    ncores : int, optional
        Number of CPU cores to use (default: 12).
    cube_name : str, optional
        Name of the output cube file (default: 'default.cube').

    Returns
    -------
    str
        Path to the generated cube file.
    """
    # Set number of threads for OpenMP (PySCF uses OpenMP for many operations)
    num_threads(ncores)

    # Build molecule from PDB file
    mol = gto.Mole(max_memory=120000)
    mol.atom = pdb_file  # PySCF can directly read PDB files
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.basis = basis
    mol.verbose = 4  # Increase verbosity to see progress (optional)
    mol.build()

    # Set up DFT with density fitting
    mf = dft.RKS(mol)
    mf.xc = functional
    # Enable density fitting (RI-J) using the specified auxiliary basis
    mf = mf.density_fit(auxbasis=aux_basis)
    mf.max_cycle = 250  # Maximum SCF iterations

    # Run SCF calculation
    energy = mf.kernel()
    print(f"SCF energy: {energy:.12f} Hartree")

    # Get the converged density matrix
    dm = mf.make_rdm1()

    # Write cube file of the electron density
    tools.cubegen.density(mol, cube_name, dm, nx=250, ny=250, nz=250)

    return cube_name


def main():
    # Get PDB file path from user input
    pdb_file = input("Enter the path to the pdb file: ").strip().replace('"', '')

    if not os.path.isfile(pdb_file):
        print(f"Error: File not found: {pdb_file}")
        return

    # Determine output cube name based on input filename
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    cube_name = base_name + '.cube'
    abs_dir = os.path.dirname(pdb_file)

    # Change to the directory of the input file (so the cube file is created there)
    os.chdir(abs_dir)

    # Perform calculation
    write_cube_pyscf(pdb_file, cube_name=cube_name)

    print(f"Cube file written: {os.path.join(abs_dir, cube_name)}")


if __name__ == '__main__':
    main()