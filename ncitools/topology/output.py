"""Formatted .nci report writer for topology analysis."""

from __future__ import annotations

import random
import textwrap
from datetime import datetime
from typing import Dict

import networkx as nx
import numpy as np
from tabulate import tabulate

from ncitools.quotes import quotes
from ncitools.correlations import correlation_1


def output(
    G: nx.Graph,
    filename: str = "default",
    ext: str = ".cube",
    correlation=correlation_1,
) -> str:
    """Write a .nci report. All distances are written in Angstrom."""
    now = datetime.now()

    edges_by_type: Dict[str, list] = {}
    for _, _, data in G.edges(data=True):
        bond_type = data.get("bond_type")
        if bond_type is None:
            continue
        edges_by_type.setdefault(bond_type, []).append(data)

    def atom_number(data, key):
        value = data.get(key)
        return "—" if value is None else str(int(value) + 1)

    def fmt_float(data, key, digits=3):
        value = data.get(key)
        return "—" if value is None else str(round(float(value), digits))

    def fmt_distance(data, key, digits=3):
        return fmt_float(data, key, digits)

    def fmt_rho(d):
        return fmt_float(d, "rho", 5)

    def fmt_laplacian(d):
        return fmt_float(d, "laplacian", 5)

    def fmt_ellipticity(d):
        return fmt_float(d, "ellipticity", 3)

    def cycle_numbers_to_string(arr):
        return ",".join(map(str, np.array(arr) + 1))

    sections = [
        {
            "key": "HB",
            "title": "HYDROGEN BONDS",
            "header": [
                "№", "Type", "D", "H", "A",
                "D-H (Å)", "H···A (Å)", "D···A (Å)", "Angle (°)",
                "ρ", "∇²ρ", "ε", "Energy",
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
                fmt_rho(d), fmt_laplacian(d), fmt_ellipticity(d),
                round(
                    float(correlation(
                        float(d["rho"]),
                        f"{d['donor_symbol']}-H...{d['acceptor_symbol']}",
                    )),
                    1,
                ),
            ],
        },
        {"key": "XB", "title": "HALOGEN BONDS"},
        {"key": "ChB", "title": "CHALCOGEN BONDS"},
        {"key": "PnB", "title": "PNICTOGEN BONDS"},
        {
            "key": "stacking",
            "title": "PARALLEL STACKING",
            "header": [
                "№", "Ring 1", "Ring 2",
                "Centroid (Å)", "Offset (Å)", "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                cycle_numbers_to_string(d["ring_1"]),
                cycle_numbers_to_string(d["ring_2"]),
                fmt_distance(d, "centroid_distance"),
                fmt_distance(d, "offset"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "H-pi",
            "title": "X-H···π INTERACTIONS",
            "header": [
                "№", "Type", "X", "H", "Ring",
                "H···π (Å)", "Angle (°)", "Projection (Å)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_x']}-H···π",
                atom_number(d, "x_num"),
                atom_number(d, "h_num"),
                cycle_numbers_to_string(d["ring"]),
                fmt_distance(d, "h_centroid_distance"),
                fmt_float(d, "angle", 1),
                fmt_distance(d, "projection_distance"),
            ],
        },
        {
            "key": "n-pi",
            "title": "LONE PAIR···π",
            "header": [
                "№", "Type", "Atom", "Ring",
                "Distance (Å)", "Angle (°)",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d['symbol_lp']}···π",
                atom_number(d, "lp_num"),
                cycle_numbers_to_string(d["ring"]),
                fmt_distance(d, "lp_centroid_distance"),
                fmt_float(d, "angle", 1),
            ],
        },
        {
            "key": "carbonyl",
            "title": "CARBONYL n→π*",
            "header": [
                "№", "Type", "Donor", "Acceptor C", "C=O",
                "Distance (Å)", "BD angle (°)", "Plane angle (°)",
                "ρ", "∇²ρ", "ε",
            ],
            "row_func": lambda d, i: [
                i + 1, "n→π*",
                atom_number(d, "donor_oxygen"),
                atom_number(d, "acceptor_carbon"),
                atom_number(d, "carbonyl_oxygen"),
                fmt_distance(d, "distance"),
                fmt_float(d, "burgi_dunitz_angle", 1),
                fmt_float(d, "plane_angle", 1),
                fmt_rho(d), fmt_laplacian(d), fmt_ellipticity(d),
            ],
        },
        {
            "key": "dihydrogen",
            "title": "DIHYDROGEN CONTACTS (H···H)",
            "header": [
                "№", "Type", "H 1", "H 2", "D 1", "D 2", "H···H (Å)",
                "ρ", "∇²ρ", "ε",
            ],
            "row_func": lambda d, i: [
                i + 1,
                f"{d.get('donor1_symbol') or '—'}-H···H-"
                f"{d.get('donor2_symbol') or '—'}",
                atom_number(d, "h1_num"),
                atom_number(d, "h2_num"),
                atom_number(d, "donor1_num"),
                atom_number(d, "donor2_num"),
                fmt_distance(d, "distance"),
                fmt_rho(d), fmt_laplacian(d), fmt_ellipticity(d),
            ],
        },
        {
            "key": "unclassified",
            "title": "UNCLASSIFIED NON-COVALENT CONTACTS",
            "header": [
                "№", "Type", "Atom 1", "Atom 2",
                "Element 1", "Element 2", "Distance (Å)",
                "ρ", "∇²ρ", "ε",
            ],
            "row_func": lambda d, i: [
                i + 1, "unclassified",
                atom_number(d, "atom1_num"),
                atom_number(d, "atom2_num"),
                d.get("atom1_symbol", "—"),
                d.get("atom2_symbol", "—"),
                fmt_distance(d, "distance"),
                fmt_rho(d), fmt_laplacian(d), fmt_ellipticity(d),
            ],
        },
    ]

    # Heavy-atom σ-hole sections
    heavy_headers = [
        "№", "Type", "X", "E", "Nu",
        "E···Nu (Å)", "Angle (°)", "ρ", "∇²ρ", "ε",
    ]
    for section in sections:
        if section["key"] in {"XB", "ChB", "PnB"}:
            section["header"] = heavy_headers
            section["row_func"] = lambda d, i: [
                i + 1,
                f"{d.get('neighbor_symbol', '—')}-"
                f"{d.get('electrophile_symbol', '—')}···"
                f"{d.get('nucleophile_symbol', '—')}",
                atom_number(d, "neighbor_num"),
                atom_number(d, "electrophile_num"),
                atom_number(d, "nucleophile_num"),
                fmt_distance(d, "distance"),
                fmt_float(d, "angle", 1),
                fmt_rho(d), fmt_laplacian(d), fmt_ellipticity(d),
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

            rows = [
                section["row_func"](edge, i)
                for i, edge in enumerate(interaction_edges)
            ]
            table = tabulate(
                rows, headers=section["header"],
                tablefmt="github", numalign="right", stralign="center",
            )
            f.write(table)
            f.write("\n")

        philosopher, quote = random.choice(quotes)
        f.write("\n\n")
        f.write(textwrap.fill(
            quote.upper(), width=40,
            initial_indent=" ", subsequent_indent=" ",
        ))
        f.write(f"\n   -- {philosopher.upper()}\n")

    return output_filename