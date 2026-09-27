# NCITools — Non-Covalent Interaction Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**NCITools** is a Python package for the automatic detection of
non-covalent interactions (NCIs) from molecular structures.  The current
release focuses on a **geometry-based** approach: all interactions are
inferred from atomic coordinates and covalent connectivity alone, without
any quantum-chemical input.

A future release will add a complementary **topology-based** approach
using QTAIM bond critical points from electron-density cube files.

---

## Key features (geometry-based)

For every structure NCITools detects:

- hydrogen bonds (D–H···A)
- halogen bonds (D–X···A, X = Cl, Br, I)
- chalcogen bonds (X = S, Se, Te)
- pnictogen bonds (X = P, As, Sb, Bi)
- tetrel bonds (X = Si, Ge, Sn, Pb)
- π···π stacking between aromatic rings
- X–H···π contacts
- lone-pair···π (n···π) contacts
- ion···π contacts
- metallophilic-like contacts (Au, Ag, Cu, Hg, Ir, Pt, Pd, Ni)
- carbonyl n→π* contacts (Bürgi–Dunitz geometry)

Each contact is scored by a **soft (fuzzy) confidence value** in
`[0, 1]`, obtained as the product of smooth geometric criteria
(distance, two angles).  A contact is reported only if its score
exceeds a user-supplied threshold.

Aromatic rings are identified by combining

1. cycle detection on the covalent graph,
2. an RMS planarity test on the ring atoms, and
3. a HOMA aromaticity index above a threshold.

Interaction energies are estimated with published empirical
correlations (Rozenberg, Espinosa, etc.); the correlations live in
`ncitools/correlations.py`.

---

## Installation

NCITools requires **Python ≥ 3.9**.

```bash
git clone https://github.com/NCI-laboratory-SPb/NCITools.git
cd ncitools

python -m venv venv
source venv/bin/activate    # Linux/macOS
# or .\venv\Scripts\activate  (Windows)

pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

---

## Quick start

### Command line

```bash
ncitools geom my_structure.xyz
```

This writes a human-readable report to `my_structure.nci` and prints a
per-family summary to the terminal:

```
Loaded 24 atoms from my_structure.xyz
  covalent bonds: 24
  aromatic rings: 1

Detected interactions:
  Hydrogen bonds         3
  π···π stacking         1
  X–H···π                2

Report written to my_structure.nci
```

Useful options:

```bash
ncitools geom complex.pdb \
    --angle-tol 120 \
    --confidence 0.8 \
    --radii bondi \
    -o complex_analysis
```

`ncitools geom --help` lists everything.

### Python API

```python
from ase.io import read
from ncitools.geometry import (
    build_graph_with_covalent_pairs,
    find_nci_bonds,
    find_nci_with_aromatic,
    output,
)

atoms = read("complex.xyz")

G = build_graph_with_covalent_pairs(atoms)
G = find_nci_bonds(atoms, G, ["H"], 110.0, "HB")
G = find_nci_with_aromatic(atoms, G)

output(G, filename="complex", ext=".xyz")
```

---

## Output format

The `.nci` file is plain text with one table per interaction family.
Columns depend on the family; a hydrogen-bond table looks like this:

```
| № | Type     | Confidence score | D | H | A | D-H (Å) | H···A (Å) | D···A (Å) | Angle DHA (°) | Angle RAH (°) | Energy (kcal/mol) |
|---|----------|------------------|---|---|---|---------|-----------|-----------|---------------|---------------|-------------------|
| 1 | O-H···O  | 0.94             | 1 | 2 | 6 | 0.970   | 1.812     | 2.771     | 170.2         | 118.5         |  4.8              |
```

---

## Package layout

```
ncitools/
├── cli.py                    # click-based command line interface
├── constants.py              # radii, HOMA parameters, element sets
├── utils.py                  # input readers
├── quotes.py                 # signature quotes for the report
├── correlations.py           # empirical geometry → energy correlations
└── geometry/
    ├── soft.py               # smooth (fuzzy) threshold functions
    ├── geom.py               # HOMA, best-fit planes, planarity metrics
    ├── graph_builder.py      # molecular graph from coordinates
    ├── output.py             # formatted .nci report writer
    └── detectors/
        ├── _common.py        # shared acceptor-angle rules
        ├── sigma_hole.py     # HB, XB, ChB, PnB, TetB
        ├── aromatic.py       # π···π, H···π, n···π, ion···π
        ├── carbonyl.py       # n→π* with carbonyls
        └── metallophilic.py  # metal···metal contacts
```

---

## Roadmap

Not yet in this release — planned for the next ones:

- **Topology-based analysis** (QTAIM) from electron-density cube files.
- **Quantum-chemical helpers**: DFT cube generation with PySCF.
- **Structure preparation**: PDB fixing, hydrogen addition, xTB calculator
- **Format conversion** utilities.

The code paths for these modules exist in the repository but are not
shipped in the first release.

---

## Known limitations

- **Hydrogen atoms must be explicit** in the input structure for
  hydrogen-bond detection.
- **Deuterium (`D`) is not recognised**; rename it to `H` before use.
- The detector reports geometric contacts, not chemical truths.  A
  short N···O contact may be a chalcogen bond, a pnictogen bond, or an
  artefact — **manual verification of borderline cases is expected**.

---

## Contributing

Bug reports, feature requests and pull requests are welcome.  Please
open an issue before submitting a large change.

---

## License

Distributed under the MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use NCITools in scientific work, please cite the original
methodological papers; each empirical correlation is documented in the
docstring of `ncitools/correlations.py`.  A Zenodo DOI will be added
after the first stable release.

