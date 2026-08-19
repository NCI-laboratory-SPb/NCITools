"""
Command-line interface: output and main pipeline.
"""

from __future__ import annotations

import os
import random
import textwrap
from datetime import datetime
from typing import Optional, List, Dict

import networkx as nx
from tabulate import tabulate

from ncitools.quotes import quotes
from ncitools.correlations import correlation_1
from ncitools.constants import bohr_to_angstrom
from ncitools.topology.topology import (
    read_cube,
    BSplineAIM,
    find_bcps_parallel,
    find_nci,
    ANGSTROM_TO_BOHR,
)
from ncitools.topology.graph_builder import (
    build_covalent_graph,
    build_graph_with_nci,
    build_graph_with_carbonyl,
    build_graph_with_unclassified,
)


def output(
    G: nx.Graph,
    filename: str = "default",
    ext: str = ".cube",
    correlation=correlation_1,
) -> str:
    """
    Write .nci report.

    All distances are written in Angstrom.
    """
    now = datetime.now()

    edges_by_type = {}
    for _, _, data in G.edges(data=True):
        bond_type = data.get("bond_type")
        if bond_type is None:
            continue
        edges_by_type.setdefault(bond_type, []).append(data)

    def atom_number(data: Dict, key: str) -> str:
        value = data.get(key)
        return "—" if value is None else str(int(value) + 1)

    def fmt_float(data: Dict, key: str, digits: int = 3) -> str:
        value = data.get(key)
        return "—" if value is None else str(round(float(value), digits))

    def fmt_distance(data: Dict, key: str, digits: int = 3) -> str:
        value = data.get(key)
        return "—" if value is None else str(round(float(value), digits))

    def fmt_rho(data: Dict) -> str:
        return fmt_float(data, "rho", 5)

    def fmt_laplacian(data: Dict) -> str:
        return fmt_float(data, "laplacian", 5)

    def fmt_ellipticity(data: Dict) -> str:
        return fmt_float(data, "ellipticity", 3)

    sections = [
        {
            "key": "HB",
            "title": "HYDROGEN BONDS",
            "header": [
                "№", "Type", "D", "H", "A",
                "D-H (Å)", "H···A (Å)", "D···A (Å)", "Angle (°)",
                "ρ", "∇²ρ", "ε", "Energy"
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['donor_symbol']}-H···{d['acceptor_symbol']}",
                atom_number(d, "donor_num"),
                atom_number(d, "hydrogen_num"),
                atom_number(d, "acceptor_num"),
                fmt_distance(d, "h_donor_distance"),
                fmt_distance(d, "distance"),
                fmt_distance(d, "heavy_atom_distance"),
                fmt_float(d, "angle", 1),
                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
                round(float(correlation(float(d["rho"]), f"{d['donor_symbol']}-H...{d['acceptor_symbol']}")), 1),
            ],
        },
        {"key": "XB", "title": "HALOGEN BONDS"},
        {"key": "ChB", "title": "CHALCOGEN BONDS"},
        {"key": "PnB", "title": "PNICTOGEN BONDS"},
        {
            "key": "stacking",
            "title": "π-π STACKING",
            "header": ["№", "Type", "Ring 1", "Ring 2", "Centroid (Å)", "Offset (Å)", "Angle (°)"],
            "row_func": lambda d, i: [
                i + 1, "parallel",
                atom_number(d, "atom_num_cycle_1"),
                atom_number(d, "atom_num_cycle_2"),
                fmt_distance(d, "centroid_distance"),
                fmt_distance(d, "offset"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "T-stacking",
            "title": "T-SHAPED STACKING",
            "header": ["№", "Type", "Ring 1", "Ring 2", "Centroid (Å)", "Angle (°)"],
            "row_func": lambda d, i: [
                i + 1, "T-shaped",
                atom_number(d, "atom_num_cycle_1"),
                atom_number(d, "atom_num_cycle_2"),
                fmt_distance(d, "centroid_distance"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "H-pi",
            "title": "X-H···π INTERACTIONS",
            "header": ["№", "Type", "X", "H", "Ring", "H···π (Å)", "Angle (°)", "Projection (Å)"],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_x']}-H···π",
                atom_number(d, "x_num"),
                atom_number(d, "h_num"),
                atom_number(d, "atom_num_cycle"),
                fmt_distance(d, "h_centroid_distance"),
                fmt_float(d, "angle", 1),
                fmt_distance(d, "projection_distance"),
            ],
        },
        {
            "key": "n-pi",
            "title": "LONE PAIR···π",
            "header": ["№", "Type", "Atom", "Ring", "Distance (Å)", "Angle (°)"],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_lp']}···π",
                atom_number(d, "lp_num"),
                atom_number(d, "atom_num_cycle"),
                fmt_distance(d, "lp_centroid_distance"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "carbonyl",
            "title": "CARBONYL n→π*",
            "header": ["№", "Type", "Donor", "Acceptor C", "C=O", "Distance (Å)", "BD angle (°)", "Plane angle (°)", "ρ", "∇²ρ", "ε"],
            "row_func": lambda d, i: [
                i + 1, "n→π*",
                atom_number(d, "donor_oxygen"),
                atom_number(d, "acceptor_carbon"),
                atom_number(d, "carbonyl_oxygen"),
                fmt_distance(d, "distance"),
                fmt_float(d, "burgi_dunitz_angle", 1),
                fmt_float(d, "plane_angle", 1),
                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },
        {
            "key": "unclassified",
            "title": "UNCLASSIFIED NON-COVALENT CONTACTS",
            "header": ["№", "Type", "Atom 1", "Atom 2", "Element 1", "Element 2", "Distance (Å)", "ρ", "∇²ρ", "ε"],
            "row_func": lambda d, i: [
                i + 1, "unclassified",
                atom_number(d, "atom1_num"),
                atom_number(d, "atom2_num"),
                d.get("atom1_symbol", "—"),
                d.get("atom2_symbol", "—"),
                fmt_distance(d, "distance"),
                fmt_rho(d),
                fmt_laplacian(d),
                fmt_ellipticity(d),
            ],
        },
    ]

    # Add heavy-atom sections headers
    heavy_headers = ["№", "Type", "X", "E", "Nu", "E···Nu (Å)", "Angle (°)", "ρ", "∇²ρ", "ε"]
    for key, title in [("XB", "HALOGEN BONDS"), ("ChB", "CHALCOGEN BONDS"), ("PnB", "PNICTOGEN BONDS")]:
        for section in sections:
            if section["key"] == key:
                section["header"] = heavy_headers
                section["row_func"] = lambda d, i: [
                    i + 1,
                    f"{d.get('neighbor_symbol', '—')}-{d.get('electrophile_symbol', '—')}···{d.get('nucleophile_symbol', '—')}",
                    atom_number(d, "neighbor_num"),
                    atom_number(d, "electrophile_num"),
                    atom_number(d, "nucleophile_num"),
                    fmt_distance(d, "distance"),
                    fmt_float(d, "angle", 1),
                    fmt_rho(d),
                    fmt_laplacian(d),
                    fmt_ellipticity(d),
                ]

    output_filename = f"{filename}.nci"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("NCITools: Topological analysis of electron density\n")
        f.write(f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {filename}{ext}\n")

        for section in sections:
            interaction_edges = edges_by_type.get(section["key"], [])
            if not interaction_edges:
                continue
            f.write("\n" + "=" * 120 + "\n")
            f.write(section["title"])
            f.write("\n" + "=" * 120 + "\n\n")

            rows = [section["row_func"](edge, i) for i, edge in enumerate(interaction_edges)]
            table = tabulate(rows, headers=section["header"], tablefmt="github", numalign="right", stralign="center")
            f.write(table)
            f.write("\n")

        philosopher, quote = random.choice(quotes)
        f.write("\n\n")
        f.write(textwrap.fill(quote.upper(), width=40, initial_indent=" ", subsequent_indent=" "))
        f.write(f"\n   -- {philosopher.upper()}\n")

    return output_filename


def main(
    cube_file: str,
    output_file: Optional[str] = None,
    cube_units: str = "auto",
    bcp_search_radius: float = 5.0,
    nproc: int = 4,
    grad_tol: float = 1.0e-6,
    max_iter: int = 60,
    eig_tol: float = 1.0e-8,
    min_bcp_angle: float = 140.0,
    projection_margin: float = 0.20,
    max_line_fraction: float = 0.35,
    rho_covalent_threshold: float = 0.089,
    covalent_scale: float = 1.25,
    require_negative_laplacian: bool = False,
    aromatic_mode: str = "heuristic",
    correlation=correlation_1,
) -> nx.Graph:
    """
    Run complete analysis.

    All distance-related public parameters are in Angstrom.

    Internally:
        coordinates -> Bohr
        grid vectors -> Bohr
        distances -> Bohr
        density derivatives -> atomic units
    """
    print("=" * 70)
    print("NCI TOPOLOGICAL ANALYSIS")
    print("=" * 70)

    print(f"\nReading electron density: {cube_file}")
    cube = read_cube(cube_file, cube_units=cube_units)
    print(f"Detected coordinate units: {cube.coordinate_units}")
    print(f"Grid shape: {cube.shape}")
    print(f"Number of atoms: {len(cube.atoms)}")
    print("Internal coordinate system: Bohr")
    print("Grid spacing (Å): " + ", ".join(f"{x * bohr_to_angstrom:.4f}" for x in cube.spacing))

    print("\nBuilding B-spline interpolator...")
    interp = BSplineAIM.from_density(cube.density, cube.origin, cube.axes)

    print("\nSearching for (3,-1) critical points...")
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
    )
    print(f"Unique (3,-1) BCPs found: {len(bcps)}")

    bcps_nci, bcps_covalent = find_nci(
        bcps, cube.atoms,
        rho_threshold=rho_covalent_threshold,
        covalent_scale=covalent_scale,
        require_negative_laplacian=require_negative_laplacian,
        filter_artifacts=True,
    )
    print(f"Covalent-like BCPs: {len(bcps_covalent)}")
    print(f"NCI BCPs (after filtering): {len(bcps_nci)}")

    print("\nBuilding covalent molecular graph...")
    G = build_covalent_graph(bcps_covalent, cube.atoms, aromatic_mode=aromatic_mode)

    n_atoms = sum(1 for _, data in G.nodes(data=True) if data.get("node_type") == "atom")
    n_rings = sum(1 for _, data in G.nodes(data=True) if data.get("node_type") == "aromatic_center" and data.get("aromatic", False))
    n_covalent = sum(1 for _, _, data in G.edges(data=True) if data.get("bond_type") == "covalent")
    print(f"Atoms: {n_atoms}")
    print(f"Aromatic rings: {n_rings}")
    print(f"Covalent BCP edges: {n_covalent}")

    print("\nSearching for non-covalent interactions...")
    G = build_graph_with_nci(bcps_nci, G, cube.atoms)
    G = build_graph_with_carbonyl(bcps_nci, G, cube.atoms)
    G = build_graph_with_unclassified(bcps_nci, G, cube.atoms)

    print("\nDetected interactions:")
    interaction_types = ["HB", "XB", "ChB", "PnB", "stacking", "T-stacking", "H-pi", "n-pi", "carbonyl", "unclassified"]
    for bond_type in interaction_types:
        count = sum(1 for _, _, data in G.edges(data=True) if data.get("bond_type") == bond_type)
        if count:
            print(f"  {bond_type:12s}: {count}")

    if output_file is None:
        output_file = os.path.splitext(cube_file)[0]
    input_ext = os.path.splitext(cube_file)[1]
    print(f"\nWriting results to: {output_file}.nci")
    output(G, filename=output_file, ext=input_ext, correlation=correlation)

    print("\nAnalysis completed.")
    return G


if __name__ == "__main__":
    G = main(
        r"C:\Users\User\PycharmProjects\NCITools\tests\data\carbonyl\carbonyl_1.cub",
        nproc=8,
        bcp_search_radius=3.7,
        cube_units="auto",
        rho_covalent_threshold=0.089,
        covalent_scale=1.25,
        aromatic_mode="heuristic",
    )