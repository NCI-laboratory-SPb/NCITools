from pyfock import Basis, Mol, DFT
from pyfock import Utils
import os

def write_cube_cpu(xyz, functional=None, basis='def2-SVP', aux_basis='def2-universal-jfit',
                 charge=0, ncores=12, cube_name='default.cube'):
    mol = Mol(coordfile=xyz, charge=charge)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis)})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=aux_basis)})

    dftObj = DFT(mol, basis, auxbasis, xc=[101, 130])
    dftObj.max_itr = 250
    dftObj.ncores = ncores

    # Run SCF calculation
    energy, dmat = dftObj.scf()
    Utils.write_density_cube(mol, basis, dftObj.dmat, cube_name, nx=250, ny=250, nz=250, ncores=ncores)

    return cube_name

def main():
    pdb_file = input("Enter the path to the pdb file: ").strip().replace('"','')
    filename = os.path.basename(pdb_file).split('.')[0]
    abs_dir = os.path.dirname(pdb_file)
    ext = os.path.splitext(pdb_file)[1]
    os.chdir(abs_dir)

    pdb2xyz(pdb_file)
    get_cube_cpu(filename + '.xyz')

if __name__ == '__main__':
    main()
