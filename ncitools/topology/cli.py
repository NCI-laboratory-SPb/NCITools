#!/usr/bin/env python3
"""Test CLI for the topology module.

This file is intended to be merged into the package-level ``cli.py``.
It exposes only the ``topology`` subcommand.
"""

from __future__ import annotations

import os

import click
import networkx as nx
from ase import Atoms as AseAtoms

from ncitools.constants import RADII, BOHR_TO_ANGSTROM
from ncitools.utils import get_symbol
from ncitools.geometry.graph_builder import build_graph_with_covalent_pairs

from ncitools.topology import (
    BSplineAIM,
    find_bcps_parallel,
    find_nci,
    read_cube,
    find_nci_with_aromatic,
    find_sigma_hole_interactions,
    find_carbonyl_interactions,
    find_dihydrogen_interactions,
    find_unclassified_interactions,
    prepare_graph_for_nci,
    output as topology_output,
)


_INTERACTION_TYPES = [
    "HB", "XB", "ChB", "PnB",
    "stacking", "H-pi", "n-pi",
    "carbonyl", "dihydrogen", "unclassified",
]


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #

def _count_edges(G: nx.Graph, bond_type: str) -> int:
    return sum(
        1 for _, _, data in G.edges(data=True)
        if data.get("bond_type") == bond_type
    )


def _count_aromatic_rings(G: nx.Graph) -> int:
    return sum(
        1 for _, data in G.nodes(data=True)
        if data.get("node_type") == "aromatic_center"
        and data.get("aromatic", False)
    )


def _print_summary(G: nx.Graph) -> None:
    click.echo("Detected interactions:")
    any_found = False
    for bond_type in _INTERACTION_TYPES:
        count = _count_edges(G, bond_type)
        if count:
            click.echo(f"  {bond_type:<14s} {count}")
            any_found = True
    if not any_found:
        click.echo("  (none)")
    click.echo()


# ---------------------------------------------------------------------- #
#  topology subcommand
# ---------------------------------------------------------------------- #

@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", default=None,
    help="Output basename (without .nci). Defaults to input file stem.",
)
@click.option(
    "--cube-units", default="auto",
    type=click.Choice(["auto", "bohr", "angstrom"]),
    show_default=True,
    help="Units used in the CUBE file.",
)
@click.option(
    "--nproc", default=4, show_default=True, type=int,
    help="Number of parallel workers for the BCP search.",
)
@click.option(
    "--bcp-search-radius", default=5.0, show_default=True, type=float,
    help="Atom-pair search radius for BCP search (Å).",
)
@click.option("--grad-tol", default=1.0e-6, show_default=True, type=float)
@click.option("--max-iter", default=60, show_default=True, type=int)
@click.option("--eig-tol", default=1.0e-8, show_default=True, type=float)
@click.option(
    "--min-bcp-angle", default=140.0, show_default=True, type=float,
    help="Minimum A–BCP–B angle (°) to accept a BCP.",
)
@click.option("--projection-margin", default=0.20, show_default=True, type=float)
@click.option("--max-line-fraction", default=0.35, show_default=True, type=float)
@click.option(
    "--min-alignment", default=0.9, show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum |cos| between the A–B direction and the positive "
         "Hessian eigenvector (hard gate). Use 0 to disable it.",
)
@click.option(
    "--rho-covalent-threshold", default=0.089,
    show_default=True, type=float,
    help="ρ threshold for covalent-like BCPs (a.u.).",
)
@click.option(
    "--require-negative-laplacian", is_flag=True, default=False,
    help="Require ∇²ρ < 0 to call a BCP covalent-like.",
)
@click.option(
    "--tolerance", default=0.2, show_default=True, type=float,
    help="Extra slack (Å) for covalent bond detection and "
         "covalent-like BCP screening.",
)
@click.option(
    "--skip", multiple=True, type=click.Choice(_INTERACTION_TYPES),
    help="Interaction type(s) to skip (repeatable).",
)
def topology(
    input_file,
    output,
    cube_units,
    nproc,
    bcp_search_radius,
    grad_tol,
    max_iter,
    eig_tol,
    min_bcp_angle,
    projection_margin,
    max_line_fraction,
    min_alignment,
    rho_covalent_threshold,
    require_negative_laplacian,
    tolerance,
    skip,
):
    """
    Topological analysis of electron density from a CUBE file.

    The pipeline is:

        1. read CUBE and build a B-spline density model;
        2. locate (3,-1) BCPs and split them into covalent / NCI;
        3. build the covalent graph via geometry.build_graph_with_covalent_pairs;
        4. classify NCI BCPs with the detectors in ``topology.detectors``;
        5. write an .nci report.

    INPUT_FILE must be a Gaussian-style CUBE file.
    """
    click.echo("=" * 70)
    click.echo("NCI TOPOLOGICAL ANALYSIS")
    click.echo("=" * 70)

    click.echo(f"\nReading electron density: {input_file}")
    cube = read_cube(input_file, cube_units=cube_units)
    click.echo(f"Detected coordinate units: {cube.coordinate_units}")
    click.echo(f"Grid shape: {cube.shape}")
    click.echo(f"Number of atoms: {len(cube.atoms)}")
    click.echo("Internal coordinate system: Bohr")

    click.echo("\nBuilding B-spline interpolator...")
    interp = BSplineAIM.from_density(cube.density, cube.origin, cube.axes)

    click.echo("\nSearching for (3,-1) critical points...")
    bcps = find_bcps_parallel(
        interp, cube.atoms,
        bcp_search_radius=bcp_search_radius,
        nproc=nproc,
        grad_tol=grad_tol,
        max_iter=max_iter,
        eig_tol=eig_tol,
        min_bcp_angle=min_bcp_angle,
        projection_margin=projection_margin,
        max_line_fraction=max_line_fraction,
        min_alignment=min_alignment,
    )
    click.echo(f"Unique (3,-1) BCPs found: {len(bcps)}")

    bcps_nci, bcps_covalent = find_nci(
        bcps, cube.atoms,
        rho_threshold=rho_covalent_threshold,
        tolerance=tolerance,
        require_negative_laplacian=require_negative_laplacian,
        filter_artifacts=True,
    )
    click.echo(f"Covalent-like BCPs: {len(bcps_covalent)}")
    click.echo(f"NCI BCPs (after filtering): {len(bcps_nci)}")

    # ------------------------------------------------------------------ #
    # Covalent graph — built by geometry.graph_builder
    # ------------------------------------------------------------------ #
    click.echo("\nBuilding covalent graph (geometry.graph_builder)...")

    symbols = [get_symbol(z) for z, _ in cube.atoms]
    positions_angstrom = [
        pos * BOHR_TO_ANGSTROM for _, pos in cube.atoms
    ]
    ase_atoms = AseAtoms(symbols=symbols, positions=positions_angstrom)

    G = build_graph_with_covalent_pairs(
        ase_atoms,
        radii=RADII,
        tolerance=tolerance,
    )

    # geometry works in Angstrom → convert to Bohr, enrich rings.
    prepare_graph_for_nci(G, cube.atoms)

    click.echo(f"  atoms:          {sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'atom')}")
    click.echo(f"  aromatic rings: {_count_aromatic_rings(G)}")
    click.echo(f"  covalent bonds: {_count_edges(G, 'covalent')}")
    click.echo()

    # ------------------------------------------------------------------ #
    # NCI detectors
    # ------------------------------------------------------------------ #
    skipped = set(skip)

    if not {"stacking", "H-pi", "n-pi"}.issubset(skipped):
        G = find_nci_with_aromatic(bcps_nci, G, cube.atoms)

    if not {"HB", "XB", "ChB", "PnB"}.issubset(skipped):
        G = find_sigma_hole_interactions(bcps_nci, G, cube.atoms)

    if "carbonyl" not in skipped:
        G = find_carbonyl_interactions(bcps_nci, G, cube.atoms)

    if "dihydrogen" not in skipped:
        G = find_dihydrogen_interactions(bcps_nci, G, cube.atoms)

    if "unclassified" not in skipped:
        G = find_unclassified_interactions(bcps_nci, G, cube.atoms)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    _print_summary(G)

    if output is None:
        output = os.path.splitext(input_file)[0]
    input_ext = os.path.splitext(input_file)[1]

    click.echo(f"Writing results to: {output}.nci")
    topology_output(G, filename=output, ext=input_ext)

    click.echo("\nAnalysis completed.")


# ---------------------------------------------------------------------- #
#  Entry point
# ---------------------------------------------------------------------- #

@click.group()
@click.version_option(version="0.2.0", prog_name="ncitools")
def cli():
    """NCITools analysis (test CLI)."""
    pass


cli.add_command(topology)


if __name__ == "__main__":
    cli()