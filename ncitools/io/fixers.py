"""
pdb_preparation.py

Preparation of protein PDB structures for subsequent
non-covalent interaction analysis.

Dependencies
------------
pdbfixer
openmm
propka

No external executables are required.

Pipeline
--------
PDB
 |
 +--> PDBFixer
 |      |
 |      +--> missing residues       (optional)
 |      +--> nonstandard residues   (optional)
 |      +--> missing heavy atoms
 |
 +--> fixed PDB
 |
 +--> PROPKA
 |      |
 |      +--> pKa prediction
 |      +--> pKa report
 |
 +--> PDBFixer
        |
        +--> missing hydrogens at requested pH
        |
        +--> protonated PDB

The resulting protonated PDB is intended to be passed
to downstream structure-analysis modules.
"""

from pathlib import Path
import argparse
import io


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _check_input_file(pdb):
    """Return Path object and check that the input file exists."""
    pdb = Path(pdb)

    if not pdb.is_file():
        raise FileNotFoundError(
            f"PDB file not found: {pdb}"
        )

    return pdb


def _write_pdb(topology, positions, output):
    """Write an OpenMM topology and positions to a PDB file."""
    from openmm.app import PDBFile

    with open(output, "w") as f:
        PDBFile.writeFile(
            topology,
            positions,
            f,
            keepIds=True,
        )


# ---------------------------------------------------------------------------
# PDBFixer
# ---------------------------------------------------------------------------

def fix_structure(
    pdb,
    output=None,
    add_missing_residues=False,
    replace_nonstandard=True,
    keep_heterogens=True,
):
    """
    Repair a PDB structure using PDBFixer.
    """

    from pdbfixer import PDBFixer

    pdb = _check_input_file(pdb)

    if output is None:
        output = pdb.with_name(
            pdb.stem + "_fixed.pdb"
        )

    output = Path(output)

    fixer = PDBFixer(filename=str(pdb))

    # -------------------------------------------------------
    # Missing residues
    # -------------------------------------------------------
    #
    # findMissingAtoms() expects missingResidues to exist,
    # so this method must be called even when we don't want
    # to reconstruct missing residues.
    #
    fixer.findMissingResidues()

    if not add_missing_residues:
        fixer.missingResidues = {}

    # -------------------------------------------------------
    # Non-standard residues
    # -------------------------------------------------------

    if replace_nonstandard:
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

    # -------------------------------------------------------
    # Heterogens
    # -------------------------------------------------------

    if not keep_heterogens:
        fixer.removeHeterogens(
            keepWater=False
        )

    # -------------------------------------------------------
    # Missing heavy atoms
    # -------------------------------------------------------

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # -------------------------------------------------------
    # Write PDB
    # -------------------------------------------------------

    _write_pdb(
        fixer.topology,
        fixer.positions,
        output,
    )

    return str(output)



# ---------------------------------------------------------------------------
# PROPKA
# ---------------------------------------------------------------------------

def calculate_pka(
    pdb,
    pH=7.4,
    write_pka=True,
):
    """
    Run PROPKA directly through its Python API.

    No external executable is called.

    Parameters
    ----------
    pdb : str or Path
        Input PDB file.

    pH : float
        pH used for the PROPKA calculation.

    write_pka : bool
        Whether PROPKA should write its .pka report.

    Returns
    -------
    MolecularContainer
        PROPKA molecular container.

    Notes
    -----
    PROPKA's Python API is not guaranteed to remain stable
    between minor releases.
    """

    from propka.run import single

    pdb = _check_input_file(pdb)

    if not 0.0 <= pH <= 14.0:
        raise ValueError(
            f"pH must be between 0 and 14, got {pH}"
        )

    # PROPKA's --pH option is passed through optargs.
    options = [
        "--pH",
        str(pH),
    ]

    molecule = single(
        str(pdb),
        optargs=options,
        write_pka=write_pka,
    )

    return molecule


def get_pka_report_path(pdb):
    """
    Return the conventional PROPKA .pka filename.
    """
    pdb = Path(pdb)

    return pdb.with_suffix(".pka")


# ---------------------------------------------------------------------------
# PROPKA results
# ---------------------------------------------------------------------------

def extract_pka_values(molecule):
    """
    Extract a simple dictionary of pKa values from a PROPKA
    MolecularContainer.

    Returns
    -------
    dict
        Keys are residue labels and values are predicted pKa.

    Notes
    -----
    PROPKA's internal data model is subject to API changes.
    This function intentionally exposes only a simple representation.
    """

    result = {}

    for name in molecule.conformation_names:

        conformation = molecule.conformations[name]

        for group in conformation.groups:

            try:
                atom = group.atom

                residue = atom.res_name
                residue_number = atom.res_num
                chain = atom.chain_id

                label = (
                    f"{residue}{residue_number}"
                )

                if chain:
                    label = (
                        f"{chain}:{residue}{residue_number}"
                    )

                result[label] = group.pka_value

            except AttributeError:
                # Ignore groups that do not expose the expected
                # pKa-related attributes.
                continue

    return result


# ---------------------------------------------------------------------------
# PDBFixer protonation
# ---------------------------------------------------------------------------

def protonate_structure(
    pdb,
    pH=7.4,
    output=None,
):
    """Add missing hydrogen atoms using PDBFixer."""

    from pdbfixer import PDBFixer

    pdb = _check_input_file(pdb)

    if output is None:
        output = pdb.with_name(
            pdb.stem + "_protonated.pdb"
        )

    output = Path(output)

    fixer = PDBFixer(filename=str(pdb))

    fixer.addMissingHydrogens(pH)

    _write_pdb(
        fixer.topology,
        fixer.positions,
        output,
    )

    return str(output)



# ---------------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------------

def prepare_pdb(
    pdb,
    output_dir=None,
    pH=7.4,
    add_missing_residues=False,
    replace_nonstandard=True,
    keep_heterogens=True,
    calculate_propka=True,
):
    """
    Prepare a PDB structure for non-covalent interaction analysis.

    Parameters
    ----------
    pdb : str or Path
        Input PDB file.

    output_dir : str or Path, optional
        Directory for generated files.

    pH : float
        Target pH.

    add_missing_residues : bool
        Whether to reconstruct missing residues.

    replace_nonstandard : bool
        Whether to replace non-standard residues.

    keep_heterogens : bool
        Whether to preserve ligands, ions and water.

    calculate_propka : bool
        Whether to run PROPKA.

    Returns
    -------
    dict
        Paths and PROPKA information.
    """

    pdb = _check_input_file(pdb)

    # -------------------------------------------------------
    # Output directory
    # -------------------------------------------------------

    if output_dir is None:
        output_dir = pdb.parent / (
            pdb.stem + "_prepared"
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------
    # Step 1: Fix heavy-atom structure
    # -------------------------------------------------------

    fixed_pdb = output_dir / (
        pdb.stem + "_fixed.pdb"
    )

    fixed_pdb = fix_structure(
        pdb=pdb,
        output=fixed_pdb,
        add_missing_residues=add_missing_residues,
        replace_nonstandard=replace_nonstandard,
        keep_heterogens=keep_heterogens,
    )

    # -------------------------------------------------------
    # Step 2: PROPKA
    # -------------------------------------------------------

    molecule = None
    pka_file = None
    pka_values = {}

    if calculate_propka:

        molecule = calculate_pka(
            fixed_pdb,
            pH=pH,
            write_pka=True,
        )

        pka_file = get_pka_report_path(
            fixed_pdb
        )

        try:
            pka_values = extract_pka_values(
                molecule
            )
        except (AttributeError, TypeError):
            pka_values = {}

    # -------------------------------------------------------
    # Step 3: Add hydrogens
    # -------------------------------------------------------

    protonated_pdb = output_dir / (
        pdb.stem + "_protonated.pdb"
    )

    protonated_pdb = protonate_structure(
        fixed_pdb,
        pH=pH,
        output=protonated_pdb,
    )

    # -------------------------------------------------------
    # Return results
    # -------------------------------------------------------

    return {
        "input": str(pdb),
        "fixed_pdb": str(fixed_pdb),
        "protonated_pdb": str(protonated_pdb),
        "pka_file": (
            str(pka_file)
            if pka_file is not None
            and pka_file.is_file()
            else None
        ),
        "pka_values": pka_values,
        "propka": molecule,
        "pH": pH,
    }


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def main():
    """
    Demonstration of the preparation pipeline.

    Usage:

        python pdb_preparation.py 1abc.pdb

    or:

        python pdb_preparation.py 1abc.pdb --ph 6.5

    This function is intentionally simple so that the module
    can also be tested manually before integrating it into
    the rest of the project.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Prepare a PDB structure using "
            "PDBFixer and PROPKA."
        )
    )

    parser.add_argument(
        "pdb",
        help="Input PDB file",
    )

    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="Target pH (default: 7.4)",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated files",
    )

    parser.add_argument(
        "--add-missing-residues",
        action="store_true",
        help="Reconstruct missing residues",
    )

    parser.add_argument(
        "--remove-heterogens",
        action="store_true",
        help=(
            "Remove heterogens instead of preserving them"
        ),
    )

    parser.add_argument(
        "--no-replace-nonstandard",
        action="store_true",
        help="Do not replace non-standard residues",
    )

    parser.add_argument(
        "--no-propka",
        action="store_true",
        help="Do not run PROPKA",
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Run preparation
    # -------------------------------------------------------

    result = prepare_pdb(
        pdb=args.pdb,
        output_dir=args.output_dir,
        pH=args.ph,
        add_missing_residues=args.add_missing_residues,
        replace_nonstandard=(
            not args.no_replace_nonstandard
        ),
        keep_heterogens=(
            not args.remove_heterogens
        ),
        calculate_propka=(
            not args.no_propka
        ),
    )

    # -------------------------------------------------------
    # Display result
    # -------------------------------------------------------

    print()
    print("PDB preparation completed.")
    print()

    print(
        f"Input PDB:       {result['input']}"
    )

    print(
        f"Fixed PDB:       {result['fixed_pdb']}"
    )

    print(
        f"Protonated PDB:  {result['protonated_pdb']}"
    )

    print(
        f"PROPKA .pka:     {result['pka_file']}"
    )

    print(
        f"pH:               {result['pH']}"
    )

    print()

    if result["pka_values"]:

        print("Predicted pKa values:")
        print()

        for residue, pka in sorted(
            result["pka_values"].items()
        ):
            print(
                f"  {residue:12s} {pka:6.2f}"
            )

        print()

from openmm import unit
from openmm import CustomExternalForce, VerletIntegrator
from openmm.app import PDBFile, ForceField, Simulation


def optimize_hydrogens(
    pdb,
    output=None,
    max_iterations=200,
    tolerance=10,
):
    """
    Optimize only hydrogen coordinates using OpenMM
    molecular mechanics.

    Heavy atoms remain fixed.
    """
    pass

if __name__ == "__main__":
    main()
