#!/usr/bin/env python3
"""
Command-line interface for NCITools.

First-release scope
-------------------
Only the geometry-based analysis is exposed:

    ncitools geom INPUT_FILE [OPTIONS]

Future releases will add the topological (``top``), quantum-chemical
(``cube``), structure-preparation (``fix``) and format-conversion
(``convert``) subcommands.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from ase.io import read

from ncitools.constants import BONDI, RADII, CHARRY_TKATCHENKO, BOHR_TO_ANGSTROM
from ncitools.utils import read_input
from ncitools.geometry import (
    build_graph_with_covalent_pairs,
    find_carbonyl_interactions,
    find_intermetalls_contact,
    find_nci_bonds,
    find_nci_with_aromatic,
    output as geom_output,
)


# ---------------------------------------------------------------------- #
#  Human-readable labels for the report summary
# ---------------------------------------------------------------------- #

_INTERACTION_LABELS = {
    "HB":            "Hydrogen bonds",
    "XB":            "Halogen bonds",
    "ChB":           "Chalcogen bonds",
    "PnB":           "Pnictogen bonds",
    "TetB":          "Tetrel bonds",
    "stacking":      "π···π stacking",
    "H-pi":          "X–H···π",
    "n-pi":          "Lone pair···π",
    "ion-pi":        "Ion···π",
    "metallophilic": "Metallophilic",
    "carbonyl":      "Carbonyl n→π*",
}


# ---------------------------------------------------------------------- #
#  geom subcommand
# ---------------------------------------------------------------------- #

@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", default=None,
    help="Output basename (without .nci). Defaults to the input file stem.",
)
@click.option(
    "--angle-tol-hb", default=110.0, show_default=True,
    help="Minimum D–X···A angle (degrees) for hydrogen bonds.",
)
@click.option(
    "--angle-tol", default=150.0, show_default=True,
    help="Minimum D–X···A angle (degrees) for  σ-hole bonds.",
)
@click.option(
    "--confidence", default=0.75, show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum confidence score to accept an interaction.",
)
@click.option(
    "--tolerance", default=0.2, show_default=True,
    help="Extra slack (Å) added to covalent radii for bond detection.",
)
@click.option(
    "--radii", default="bondi", show_default=True,
    type=click.Choice(["bondi", "charry-tkachenko"]),
    help="Van der Waals radii set used for contact detection.",
)
@click.option(
    "--skip", multiple=True,
    type=click.Choice(list(_INTERACTION_LABELS)),
    help="Interaction type(s) to skip (repeatable).",
)
@click.option(
    "--bohr", is_flag=True, default=False,
    help="Specifying coordinates units. In default, coordinates are assumed in angstroms.",
)
def geom(input_file, output, angle_tol_hb, angle_tol, confidence, tolerance, radii, skip, bohr):
    """
    Detect non-covalent interactions from atomic coordinates only.

    INPUT_FILE must be a structure file readable by ASE
    (XYZ, PDB, CIF, …).  A human-readable report is written to
    ``<output>.nci``.
    """
    if output is None:
        output = Path(input_file).stem

    vdw_radii = BONDI if radii == "bondi" else CHARRY_TKATCHENKO

    atoms = read_input(input_file)
    click.echo(f"Loaded {len(atoms)} atoms from {input_file}")

    if bohr:
        atoms.positions *= BOHR_TO_ANGSTROM

    # ------------------------------------------------------------------ #
    # Covalent graph
    # ------------------------------------------------------------------ #
    G = build_graph_with_covalent_pairs(
        atoms,
        radii=RADII,
        tolerance=tolerance,
    )

    click.echo(f"  covalent bonds: {_count_edges(G, 'covalent')}")
    click.echo(f"  aromatic rings: {_count_aromatic_rings(G)}")
    click.echo()

    # ------------------------------------------------------------------ #
    # Non-covalent interactions
    # ------------------------------------------------------------------ #
    skipped = set(skip)

    if "HB" not in skipped:
        G = find_nci_bonds(
            atoms, G, ["H"], angle_tol_hb, "HB",
            desc="Searching for hydrogen bonds",
            radii=vdw_radii,
            confidence_threshold=confidence,
        )

    if "XB" not in skipped:
        G = find_nci_bonds(
            atoms, G, ["Cl", "Br", "I"], angle_tol, "XB",
            desc="Searching for halogen bonds",
            radii=vdw_radii,
            confidence_threshold=confidence,
        )

    if "ChB" not in skipped:
        G = find_nci_bonds(
            atoms, G, ["S", "Se", "Te"], angle_tol, "ChB",
            desc="Searching for chalcogen bonds",
            radii=vdw_radii,
            confidence_threshold=confidence,
        )

    if "PnB" not in skipped:
        G = find_nci_bonds(
            atoms, G, ["P", "As", "Sb", "Bi"], angle_tol, "PnB",
            desc="Searching for pnictogen bonds",
            radii=vdw_radii,
            confidence_threshold=confidence,
        )

    if "TetB" not in skipped:
        G = find_nci_bonds(
            atoms, G, ["Si", "Ge", "Sn", "Pb"], angle_tol, "TetB",
            desc="Searching for tetrel bonds",
            radii=vdw_radii,
            confidence_threshold=confidence,
        )

    aromatic_family = {"stacking", "H-pi", "n-pi", "ion-pi"}
    if not aromatic_family.issubset(skipped):
        G = find_nci_with_aromatic(
            atoms, G,
            radii=vdw_radii,
            confidence_threshold=confidence,
        )

    if "carbonyl" not in skipped:
        G = find_carbonyl_interactions(
            atoms, G,
            confidence_threshold=confidence,
        )

    if "metallophilic" not in skipped:
        G = find_intermetalls_contact(
            atoms, G,
            confidence_threshold=confidence,
        )

    # ------------------------------------------------------------------ #
    # Terminal summary and report
    # ------------------------------------------------------------------ #
    _print_summary(G)

    ext = os.path.splitext(input_file)[1]
    geom_output(G, filename=output, ext=ext)
    click.echo(f"Report written to {output}.nci")


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #

def _count_edges(G, bond_type: str) -> int:
    """Number of edges of a given ``bond_type``."""
    return sum(
        1 for _, _, data in G.edges(data=True)
        if data.get("bond_type") == bond_type
    )


def _count_aromatic_rings(G) -> int:
    """Number of aromatic-centre nodes in the graph."""
    return sum(
        1 for _, data in G.nodes(data=True)
        if data.get("node_type") == "aromatic_center"
        and data.get("aromatic", False)
    )


def _print_summary(G) -> None:
    """Print a short per-family count of detected interactions."""
    click.echo("Detected interactions:")
    any_found = False

    for btype, label in _INTERACTION_LABELS.items():
        count = _count_edges(G, btype)
        if count:
            click.echo(f"  {label:<22s} {count}")
            any_found = True

    if not any_found:
        click.echo("  (none)")

    click.echo()


# ---------------------------------------------------------------------- #
#  Entry point
# ---------------------------------------------------------------------- #

@click.group()
@click.version_option(version="0.1.0", prog_name="ncitools")
def cli():
    """
    NCITools — analysis of non-covalent interactions.

    This release ships the geometry-based analysis only;
    see `ncitools geom --help`.
    """
    pass


cli.add_command(geom)


if __name__ == "__main__":
    cli()