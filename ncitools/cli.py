#!/usr/bin/env python3
"""
Command-line interface for NCITools – Non-Covalent Interaction analysis.

Subcommands:
    geom       Detect hydrogen bonds from molecular geometry.
    top        Perform topological (AIM/NCI) analysis from a cube file.
    cube       Generate electron density cube file from a PDB using DFT (PySCF).
    fix        Fix PDB structure, add hydrogens, optionally optimize them.
    convert    Convert between file formats (e.g., PDB → XYZ).
"""

import os
import sys
import click
from pathlib import Path

# Import local modules (adjust if package structure changes)
from ncitools.geometry.geom import (
    build_graph_with_covalent_pairs, find_nci_with_aromatic, find_carbonyl_interactions,
    find_nci_bonds, output as geom_output
)
from ncitools.topology.topology import (
    read_cube, BSplineAIM, find_bcps_parallel, find_nci,
)
from ncitools.topology.graph_builder import (
    build_graph_with_carbonyl,
    build_graph_with_unclassified,
    build_graph_with_nci
)
from ncitools.topology.cli import  output as top_output
from ncitools.qc.dft import write_cube_pyscf
from ncitools.io.fixers import fix_structure, optimize_hydrogens
from ncitools.io.writers import pdb2xyz


# ----------------------------------------------------------------------
# geometry subcommand
# ----------------------------------------------------------------------
@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default=None,
              help='Output file name (without extension). Default: input file base name.')
@click.option('--angle-tol', default=120.0, show_default=True,
              help='Minimum D‑H···A angle (degrees) to accept a hydrogen bond.')
@click.option('--tolerance', default=0.1, show_default=True,
              help='Tolerance (Å) added to covalent radii for bond detection.')
@click.option('--radii', default='bondi', type=click.Choice(['bondi', 'covalent']),
              help='Radii set for van der Waals contact: bondi (default) or covalent.')
def geom(input_file, output, angle_tol, tolerance, radii):
    """
    Detect hydrogen bonds using only atomic coordinates (geometry‑based).
    INPUT_FILE can be PDB, XYZ, etc. (ASE readable).
    """
    from ase.io import read
    from ncitools.constants import RADII, BONDI

    # Read structure
    atoms = read(input_file)

    if output is None:
        output = Path(input_file).stem

    # Choose radii set
    vdw_radii = BONDI if radii == 'bondi' else RADII

    # Find covalent pairs (using covalent radii + tolerance)
    G = build_graph_with_covalent_pairs(atoms, radii=RADII, tolerance=tolerance)

    # Count covalent edges
    n_cov = sum(1 for _, _, data in G.edges(data=True) if data.get('bond_type') == 'covalent')
    n_ring = sum(1 for _, data in G.nodes(data=True) if data.get('node_type') == 'aromatic_center' and data.get('aromatic', False))
    click.echo(f"Atoms: {len(atoms)}")
    click.echo(f"Aromatic rings: {n_ring}")
    click.echo(f"Covalent BCP edges: {n_cov}")
    click.echo()

    # Build graph with NCIs
    G = find_nci_bonds(atoms, G, ['H'],                      120, 'HB',
                       desc='Searching for hydrogen bonds')
    G = find_nci_bonds(atoms, G, ['Cl', 'Br', 'I'],         150, 'XB',
                       desc='Searching for halogen bonds')
    G = find_nci_bonds(atoms, G, ['S', 'Se', 'Te'],         150, 'ChB',
                       desc='Searching for chalcogen bonds')
    G = find_nci_bonds(atoms, G, ['P', 'As', 'Sb', 'Bi'],   150, 'PnB',
                       desc='Searching for pnictogen bonds')
    G = find_nci_with_aromatic(atoms, G)
    G = find_carbonyl_interactions(atoms, G)


    # Print detected interaction counts
    click.echo("\nDetected interactions:")
    interaction_types = ["HB", "XB", "ChB", "PnB", "stacking", "H-pi", "n-pi", "carbonyl"]
    for btype in interaction_types:
        count = sum(1 for _, _, data in G.edges(data=True) if data.get('bond_type') == btype)
        if count:
            click.echo(f"  {btype:12s}: {count}")
    click.echo()

    # Write output
    geom_output(G, filename=output, ext=os.path.splitext(input_file)[1])
    click.echo(f"Geometry analysis written to {output}.nci")
    click.echo(f"Done.")


# ----------------------------------------------------------------------
# topology subcommand
# ----------------------------------------------------------------------
@click.command()
@click.argument('cube_file', type=click.Path(exists=True))
@click.option('--output', '-o', default=None,
              help='Output file name (without extension). Default: cube file base name.')
@click.option('--rho-min', default=0.001, show_default=True,
              help='Minimum electron density (e/bohr³) to consider a BCP as NCI.')
@click.option('--rho-max', default=0.1, show_default=True,
              help='Maximum electron density (e/bohr³) for NCI.')
@click.option('--cutoff', default=5.0, show_default=True,
              help='Distance cutoff (Å) for considering atom pairs as candidates for BCP search.')
@click.option('--nproc', default=4, show_default=True,
              help='Number of CPU cores for parallel BCP search.')
@click.option('--rho-covalent', default=0.01, show_default=True,
              help='Density threshold for covalent-like BCPs (e/bohr³).')
@click.option('--covalent-scale', default=1.25, show_default=True,
              help='Scale factor for covalent radii in covalent classification.')
@click.option('--aromatic-mode', default='heuristic',
              type=click.Choice(['heuristic', 'none']),
              help='Mode for aromatic ring detection.')
@click.option('--require-negative-laplacian', is_flag=True,
              help='Require negative Laplacian for covalent BCPs.')
def top(cube_file, output, rho_min, rho_max, cutoff, nproc,
        rho_covalent, covalent_scale, aromatic_mode,
        require_negative_laplacian):
    """
    Perform AIM topological analysis from a Gaussian cube file.
    Finds bond critical points (BCPs) and retains those with density in [rho_min, rho_max]
    as non‑covalent interactions (NCI).
    """
    from ncitools.topology.topology import ANGSTROM_TO_BOHR, bohr_to_angstrom

    if output is None:
        output = Path(cube_file).stem

    click.echo("Reading cube file...")
    cube = read_cube(cube_file, cube_units="auto")
    atoms = cube.atoms  # list of (Z, position in Bohr)
    origin = cube.origin
    axes = cube.axes
    density = cube.density

    click.echo("Building spline interpolator...")
    interp = BSplineAIM.from_density(density, origin, axes)

    click.echo(f"Searching for BCPs (cutoff={cutoff} Å, nproc={nproc})...")
    bcps = find_bcps_parallel(
        interp, atoms,
        bcp_search_radius=cutoff,
        nproc=nproc,
        grad_tol=1.0e-6,
        max_iter=60,
        eig_tol=1.0e-8,
        min_bcp_angle=140.0,
        projection_margin=0.20,
        max_line_fraction=0.35,
        max_step_angstrom=0.75,
        dedup_tol_angstrom=0.03,
    )
    click.echo(f"Found {len(bcps)} BCPs.")

    click.echo("Classifying covalent vs NCI...")
    bcps_nci, bcps_covalent = find_nci(
        bcps, atoms,
        rho_threshold=rho_covalent,
        covalent_scale=covalent_scale,
        require_negative_laplacian=require_negative_laplacian,
        filter_artifacts=True,
    )
    click.echo(f"Covalent-like BCPs: {len(bcps_covalent)}")
    click.echo(f"NCI BCPs (after filtering): {len(bcps_nci)}")

    click.echo("Building molecular graph...")
    from ncitools.topology.graph_builder import build_covalent_graph
    G = build_covalent_graph(bcps_covalent, atoms, aromatic_mode=aromatic_mode)

    # Count covalent edges
    n_cov = sum(1 for _, _, data in G.edges(data=True) if data.get('bond_type') == 'covalent')
    n_ring = sum(1 for _, data in G.nodes(data=True) if data.get('node_type') == 'aromatic_center' and data.get('aromatic', False))
    click.echo(f"Atoms: {len(atoms)}")
    click.echo(f"Aromatic rings: {n_ring}")
    click.echo(f"Covalent BCP edges: {n_cov}")

    click.echo("Searching for non-covalent interactions...")
    G = build_graph_with_nci(bcps_nci, G, atoms)
    G = build_graph_with_carbonyl(bcps_nci, G, atoms)
    G = build_graph_with_unclassified(bcps_nci, G, atoms)

    # Print detected interaction counts
    click.echo("\nDetected interactions:")
    interaction_types = ["HB", "XB", "ChB", "PnB", "stacking", "H-pi", "n-pi", "carbonyl"]
    for btype in interaction_types:
        count = sum(1 for _, _, data in G.edges(data=True) if data.get('bond_type') == btype)
        if count:
            click.echo(f"  {btype:12s}: {count}")

    click.echo()
    click.echo(f"Writing results to {output}.nci")
    top_output(G, filename=output, ext='.cube')
    click.echo("Done.")


# ----------------------------------------------------------------------
# cube subcommand (DFT)
# ----------------------------------------------------------------------
@click.command()
@click.argument('pdb_file', type=click.Path(exists=True))
@click.option('--functional', default='b3lyp', show_default=True,
              help='Exchange‑correlation functional (PySCF syntax).')
@click.option('--basis', default='def2-svp', show_default=True,
              help='Orbital basis set.')
@click.option('--aux-basis', default='def2-universal-jfit', show_default=True,
              help='Auxiliary basis for density fitting (RI‑J).')
@click.option('--charge', default=0, show_default=True, help='Molecular charge.')
@click.option('--mult', default=1, show_default=True, help='Spin multiplicity (2S+1).')
@click.option('--ncores', default=12, show_default=True, help='Number of CPU cores.')
@click.option('--output', '-o', default=None,
              help='Output cube file name. Default: PDB base name + .cube.')
def cube(pdb_file, functional, basis, aux_basis, charge, mult, ncores, output):
    """
    Run a DFT calculation with PySCF and write the electron density cube file.
    Requires PySCF and a valid PDB file.
    """
    if output is None:
        output = Path(pdb_file).stem + '.cube'
    write_cube_pyscf(
        pdb_file, functional=functional, basis=basis,
        aux_basis=aux_basis, charge=charge, multiplicity=mult,
        ncores=ncores, cube_name=output
    )
    click.echo(f"Cube file written: {output}")


# ----------------------------------------------------------------------
# fix subcommand
# ----------------------------------------------------------------------
@click.command()
@click.argument('pdb_file', type=click.Path(exists=True))
@click.option('--add-h', is_flag=True, help='Add missing hydrogens (using PDBFixer).')
@click.option('--ph', default=7.4, show_default=True, help='pH for hydrogen addition.')
@click.option('--optimize-h', is_flag=True, help='Optimise hydrogen positions with GFN2‑xTB.')
@click.option('--charge', default=0, help='Molecular charge for xTB optimisation.')
@click.option('--output', '-o', default=None,
              help='Output file name (PDB if only fixing, XYZ if optimised).')
def fix(pdb_file, add_h, ph, optimize_h, charge, output):
    """
    Fix a PDB file: add missing residues/atoms, optionally add hydrogens
    and optimise their positions with GFN2‑xTB.
    """
    if output is None:
        output = Path(pdb_file).stem + ('_fixed.pdb' if not optimize_h else '_opt.xyz')
    # Fix PDB (add hydrogens if requested)
    fixed_pdb = fix_structure(pdb_file, add_hydrogens=add_h, pH=ph)
    if optimize_h:
        # Optimise hydrogens using xTB (requires ASE and xTB)
        optimize_hydrogens(fixed_pdb, charge=charge)
        # The optimised structure is written as XYZ by optimize_hydrogens
        # We rename/move if needed
        if output != fixed_pdb.replace('_fixed.pdb', '_opt.xyz'):
            os.rename(fixed_pdb.replace('_fixed.pdb', '_opt.xyz'), output)
        click.echo(f"Optimised structure written to {output}")
    else:
        if output != fixed_pdb:
            os.rename(fixed_pdb, output)
        click.echo(f"Fixed PDB written to {output}")


# ----------------------------------------------------------------------
# convert subcommand
# ----------------------------------------------------------------------
@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--to', 'fmt', required=True,
              type=click.Choice(['xyz', 'pdb', 'cif']),
              help='Target format (extension).')
def convert(input_file, fmt):
    """
    Convert a molecular structure file to another format.
    Supported: PDB ↔ XYZ (others can be added).
    """
    if fmt == 'xyz' and input_file.lower().endswith('.pdb'):
        pdb2xyz(input_file)
        click.echo(f"Converted {input_file} to {Path(input_file).stem}.xyz")
    else:
        click.echo(f"Conversion from {Path(input_file).suffix} to .{fmt} not implemented yet.")


# ----------------------------------------------------------------------
# Main CLI group
# ----------------------------------------------------------------------
@click.group()
def cli():
    """NCITools – Analyse non‑covalent interactions (hydrogen bonds) from geometry or electron density."""
    pass

cli.add_command(geom)
cli.add_command(top)
cli.add_command(cube)
cli.add_command(fix)
cli.add_command(convert)

if __name__ == '__main__':
    cli()