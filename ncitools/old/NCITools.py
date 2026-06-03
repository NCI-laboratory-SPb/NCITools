import os
from ncitools.ncitools.constants import BONDI

"""import QC
from DensTopology import *
from ReadTools import pdb2xyz, fixPDB, optimize_hydrogens

def header():
    pass

def output():
    pass

pdb_file = input("Enter the path to the pdb file: ").strip().replace('"','')
abs_dir = os.path.dirname(os.path.abspath(pdb_file))
ext = os.path.splitext(pdb_file)[1]
os.chdir(abs_dir)

is_add_H = [False, True][int(input("Add protons? (0 = no, 1 = yes) "))]

if is_add_H == True:
    pH = float(input("Choose pH: "))
    filename = fixPDB(pdb_file, add_hydrogens=is_add_H, pH=pH)
else:
    pH = 7.4
    filename = pdb_file

pdb2xyz(filename)
filename_xyz = filename.split('.')[0] + '.xyz'

is_opt_hydrogens = [False, True][int(input("Optimize protons coordinates? (0 = no, 1 = yes) "))]
charge = int(input("Give me a charge of your molecule: "))
multiplicity = int(input("Give me a multiplicity of your molecule: "))

if is_opt_hydrogens:
    optimize_hydrogens(filename_xyz, charge=charge)

cube_file = QC.write_cube_pyscf(filename_xyz, charge=charge, multiplicity=multiplicity, ncores=6)

# Чтение
print('Reading electron density...')
origin, spacing, density, atoms = read_cube(cube_file)
print('Reading is done!')

# Интерполятор
print('Starting interpolation of grid data')
interp = BSplineAIM.from_density(density, origin, spacing)
print('Interpolation completed')

# Поиск всех BCP
print('Starting topological analysis...')
bcps = find_bcp_parallel(
    interp,
    atoms,
    cutoff=5.0,
    nproc=8
)

# Отбор NCI по плотности и определение типов
nci_list = find_nci(bcps, atoms, rho_min=1e-3)

print(f"\nNon-covalent interactions found: {len(nci_list)}")

# Запись в файл
output_nci(nci_list, filename=filename.split('.')[0], ext=ext)


# добавь функцию оптимизации координат протонов
# пиши функции header и output. Оформляй код в прилежный вид
# Добавляй исключения для функций
# Создавай сайт
"""