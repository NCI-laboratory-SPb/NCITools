import os
from ase.io.proteindatabank import read_proteindatabank

def read_pdb(file):

    ext = os.path.splitext(file)[1]
    if ext == ".pdb":

        pdb_file = read_proteindatabank(file)

        return pdb_file

    else:
        print("Unrecognised file type", ext)

