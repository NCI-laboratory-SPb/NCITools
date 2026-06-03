import os
from ase.io import write
from ase.io.proteindatabank import read_proteindatabank

def pdb2xyz(file):

    ext = os.path.splitext(file)[1]
    name = os.path.splitext(file)[0]
    if ext == ".pdb":
        pdb = read_proteindatabank(file)
        write(filename=f'{name}' + '.xyz', images=pdb, format='xyz')

    else:
        print("Unrecognised file type", ext)