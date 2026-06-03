"""
PDB fixing (PDBFixer) and hydrogen optimisation (xTB/ASE).
"""

import os

def fixPDB(pdb, add_hydrogens, pH=7.4):
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    filename = os.path.basename(pdb).split('.')[0]
    filename_fixed = filename + '_fixed.pdb'

    fixer = PDBFixer(filename=pdb)
    #fixer.findMissingResidues()
    #fixer.findNonstandardResidues()
    #fixer.replaceNonstandardResidues()
    #fixer.removeHeterogens(keepWater=False)
    #fixer.findMissingAtoms()
    #fixer.addMissingAtoms()

    if add_hydrogens:
        fixer.addMissingHydrogens(pH)


    PDBFile.writeFile(fixer.topology, fixer.positions, open(filename_fixed, 'w'))

    return filename_fixed


def optimize_hydrogens(filename, charge=0):
    from ase.io import read, write
    from ase.constraints import FixAtoms
    from xtb.ase.calculator import XTB
    from ase.optimize import FIRE

    molecule = read(filename)
    c = FixAtoms(indices=[atom.index for atom in molecule if atom.symbol != 'H'])
    molecule.set_constraint(c)

    calc = XTB(method="GFN2-xTB", charge=charge, max_iterations=1000)
    molecule.calc = calc
    opt = FIRE(molecule)
    opt.run(fmax=1e-3)

    write(filename, molecule, format='xyz')
