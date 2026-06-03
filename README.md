```markdown
# NCITools – Non‑Covalent Interaction Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NCITools** is a Python package that automates the detection and energy estimation of non‑covalent interactions (especially hydrogen bonds) from molecular structures. It provides two complementary approaches:

1. **Geometry‑based analysis** – uses only atomic coordinates (distances and angles) to identify hydrogen bonds.  
2. **Topology‑based analysis** – uses the electron density (from a `.cube` file) to locate bond critical points (BCPs) and characterise non‑covalent interactions (NCI) according to the quantum theory of atoms in molecules (QTAIM).

The package also includes utilities to prepare PDB files (add missing hydrogens, fix residues) and to generate electron density cube files via PySCF.

---

## Key Features

- **Hydrogen bond detection from geometry**  
  - Covalent bonds identified using covalent radii.  
  - H‑bonds found when H···A distance ≤ sum of van der Waals radii **and** D–H···A angle ≥ threshold (default 120°).  
  - Energy estimation using empirical correlations (Rozenberg 2000, etc.).  
  - *Requires explicit hydrogen atoms in the input structure.*

- **Topological (AIM/NCI) analysis**  
  - Reads Gaussian‑format cube files (electron density).  
  - Interpolates density with cubic B‑splines for analytical derivatives.  
  - Parallel search for bond critical points (BCPs) using Newton’s method.  
  - Filters BCPs by electron density to isolate non‑covalent interactions.  
  - Assigns contact types (e.g., `O-H...O`, `C-H...N`) and computes interaction energies using multiple published correlations.

- **PDB preparation and fixing**  
  - Add missing residues/atoms and hydrogens (using PDBFixer).  
  - Optimise only hydrogen positions with GFN2‑xTB (via ASE and xTB).

- **Electron density cube generation**  
  - Run DFT calculations with PySCF (B3LYP, wB97X‑D3, etc.) and write `.cube` files for later topological analysis.

- **Command‑line interface (CLI)**  
  - Easy‑to‑use subcommands: `geom`, `top`, `cube`, `fix`, `convert`.

---

## Installation

NCITools requires **Python ≥ 3.8**. It is recommended to use a virtual environment.

```bash
# Clone the repository (or download the source)
git clone https://github.com/yourusername/ncitools.git
cd ncitools

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or .\venv\Scripts\activate (Windows)

# Install the package in editable mode
pip install -e .
```

Optional dependencies for specific features (DFT, PDB fixing, xTB optimisation) can be installed with:

```bash
pip install -e .[all]   # install everything
# or individually:
pip install -e .[dft]   # only PySCF
pip install -e .[fix]   # only PDBFixer + OpenMM
pip install -e .[xtb]   # only xTB
```

After installation the command `ncitools` will be available in your terminal.

---

## Quick Examples

### 1. Geometry‑based hydrogen bond detection

```bash
ncitools geom my_structure.pdb --angle-tol 120 -o output
```

Output is written to `output.nci` with a human‑readable table.

### 2. Generate electron density cube from a PDB file

```bash
ncitools cube complex.pdb --functional b3lyp --basis def2-svp --ncores 8
```

Creates `complex.cube`.

### 3. Topological NCI analysis from cube file

```bash
ncitools top complex.cube --rho-min 0.002 --cutoff 4.0 --nproc 8
```

Produces a detailed `.nci` file containing BCP coordinates, density, Laplacian, eigenvalues, ellipticity, and estimated interaction energies.

### 4. Fix a PDB, add hydrogens, and optimise them

```bash
ncitools fix raw.pdb --add-h --ph 7.4 -o fixed.pdb
ncitools fix fixed.pdb --optimize-h --charge 0 -o opt.xyz
```

### 5. Convert PDB to XYZ

```bash
ncitools convert protein.pdb --to xyz
```

---

## Output Format (`.nci` file)

For geometry‑based analysis, the output table includes:

| # | Type NCI | D‑H (Å) | H···A (Å) | D···A (Å) | DHA (°) | Energy (kcal/mol) |
|---|----------|---------|-----------|-----------|---------|-------------------|

For topological analysis, the output includes:

| BCP | Type | X (Å) | Y (Å) | Z (Å) | ρ (e/bohr³) | ∇²ρ (e/bohr⁵) | λ1 | λ2 | λ3 | ε | Energy (kcal/mol) |
|-----|------|-------|-------|-------|-------------|---------------|----|----|----|---|------------------|

---

## Current Limitations

The following limitations are known and will be addressed in future versions:

- **Geometry‑based analysis** detects only **hydrogen bonds**. Other NCI types (π‑stacking, halogen bonds, etc.) are not identified.
- **Topological analysis** does **not** unambiguously assign the physical nature of a non‑covalent contact. For example, an `N···O` contact could be a chalcogen bond, a pnictogen bond, or a fortuitous interaction. **Manual verification is required** for such contacts.
- **No XYZ → PDB converter** is implemented yet (only PDB → XYZ).
- **Deuterium atoms (`D`)** are **not recognised** in PDB files. Deuterium should be renamed to `H` or removed before use.
- **Hydrogen position optimisation** (without fixing heavy atoms) is **not yet integrated** into the CLI (only available via the `fix` subcommand with `--optimize-h`).
- **Automatic determination of molecular charge and spin multiplicity** is **not implemented**. The user must provide these values for DFT calculations.

---

## Dependencies

Core dependencies (automatically installed):
- `numpy`, `scipy`, `ase`, `networkx`, `tqdm`, `click`

Optional (for specific subcommands):
- `pyscf` – for `cube` subcommand
- `pdbfixer`, `openmm` – for `fix` subcommand
- `xtb-python` – for hydrogen optimisation in `fix --optimize-h`

---

## Contributing

Bug reports, feature requests, and pull requests are welcome. Please open an issue on GitHub before submitting major changes.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Citation

If you use NCITools in scientific work, please cite the original methodological papers (see the docstrings in `correlations.py` for each correlation). A dedicated Zenodo DOI will be added after the first stable release.
```