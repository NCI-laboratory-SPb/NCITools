# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

NCITools is a Python package for detecting **non-covalent interactions (NCIs)** in
molecular structures. The supported path is **geometry-based**: interactions are
inferred from atomic coordinates and covalent connectivity (ASE + NetworkX), with
no quantum-chemical input.

A **topology-based** path (QTAIM bond critical points) lives in
`ncitools/topology/` and is experimental: it is not wired into the packaged CLI.

- Python: `>=3.9`
- Package: `ncitools`
- CLI entry point: `ncitools = ncitools.cli:cli`
- Report output: human-readable `<stem>.nci`

## Setup

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

```bash
# Full test suite
python -m pytest

# One file / one test
python -m pytest tests/test_cli.py
python -m pytest tests/test_cli.py::test_cli_help

# CLI
ncitools geom structure.xyz
ncitools geom structure.pdb --confidence 0.8 --angle-tol 120 -o analysis
ncitools geom --help
```

There is no linter or formatter configured in `pyproject.toml`. Match the
existing style instead of introducing a new tool without agreement.

## Repository layout

```text
ncitools/
  cli.py              # Click CLI (currently only the `geom` subcommand)
  constants.py        # vdW/covalent radii tables, unit conversions
  correlations.py     # empirical energy correlations
  quotes.py
  utils.py            # input reading, connectivity warnings
  geometry/           # supported geometry-based pipeline
    geom.py
    graph_builder.py  # covalent graph construction
    soft.py           # smooth (fuzzy) scoring functions
    output.py         # .nci report writer
    detectors/        # one module per interaction family
      _common.py      # shared element sets and angle logic
  topology/           # experimental QTAIM path, not in the CLI yet
    topology.py
    OLD_graph_builder.py  # legacy, do not delete without asking
tests/
  conftest.py         # ASE Atoms fixtures + tmp_xyz helper
  test_*.py
  data/               # real structures (currently empty)
```

## Architecture notes

The geometry pipeline is:

```text
atoms -> build_graph_with_covalent_pairs -> detectors -> output
```

- A molecular graph is a `networkx.Graph` whose nodes are atom indices.
- Covalent edges carry `bond_type="covalent"`; each detector adds NCI edges with
  its own `bond_type`.
- Detectors accept `(atoms, G, ...)` and **return the same graph**, annotated.
- Aromatic centres are graph nodes with `node_type="aromatic_center"` and
  `aromatic=True`.
- Confidence scores are products of smooth criteria in `[0, 1]`; a contact is
  kept only if it reaches `confidence_threshold` (CLI default `0.75`).
- Coordinates are in ångström internally; `--bohr` converts on input via
  `BOHR_TO_ANGSTROM`.

Interaction codes used by the CLI, reports and `--skip`:

```text
HB, XB, ChB, PnB, TetB, stacking, H-pi, X-pi, n-pi, ion-pi,
metallophilic, carbonyl
```

## Adding a detector

1. Put the implementation in `ncitools/geometry/detectors/<family>.py`; reuse
   helpers from `_common.py` where possible.
2. Re-export it from `ncitools/geometry/__init__.py`.
3. Wire it into `ncitools/cli.py`: the call, `_INTERACTION_LABELS`, and the
   `--skip` choices.
4. Make sure `ncitools/geometry/output.py` can render the new `bond_type`.
5. Add fixtures in `tests/conftest.py` with both a positive geometry and a
   negative control, plus a test in `tests/`.
6. Update the README feature list and CLI help if user-visible behaviour changes.

## Testing expectations

- Run `python -m pytest` before finishing a change.
- Tests use `ase.Atoms` fixtures from `tests/conftest.py`; `tmp_xyz(atoms)` writes
  a temporary XYZ file.
- CLI tests use `click.testing.CliRunner` and assert on `<output>.nci`.
- Do not weaken existing assertions to make a change pass; fix the behaviour or
  explain why the expectation changed.

## Git and release hygiene

- `develop` is the integration branch; `origin` is
  `https://github.com/NCI-laboratory-SPb/NCITools.git`.
- Never commit `*.nci`, `__pycache__`, `.idea`, `.venv` or other generated files;
  they are already covered by `.gitignore`.
- The version lives in **two places**: `[project].version` in `pyproject.toml`
  and `click.version_option(version=...)` in `ncitools/cli.py`. Bump both
  together.
- Commit messages: short and imperative, e.g. `Add tetrel-bond detector`,
  `Fix angle tolerance in sigma-hole scoring`.
- Do not push, force-push, rewrite history or change remotes without explicit
  approval from the repository owner.

## Known gotchas

- `ncitools/io/` and `ncitools/qc/` are listed in `.gitignore` as "not shipped in
  the first release"; only stale `__pycache__` directories remain. Do not add
  source there unless the packaging plan changes.
- `ncitools/topology/cli.py` exists but is not registered in the `ncitools`
  entry point.
- Several detector defaults differ from the CLI defaults (for example carbonyl
  uses `confidence_threshold=0.8`); always check the detector signature.
