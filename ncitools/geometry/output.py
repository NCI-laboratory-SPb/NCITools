"""
Formatted report writer for detected non-covalent interactions.

The public entry point :func:`output` writes a human-readable ``.nci``
file listing every interaction grouped by family.  Sections with no
matches are omitted.  A random philosopher's quote is appended at the
end as a small signature of NCItools.
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import networkx as nx
import numpy as np
from tabulate import tabulate

from ncitools.correlations import universal_1_hb, universal_1_sb
from ncitools.quotes import quotes


# --------------------------------------------------------------------------- #
#  Section description
# --------------------------------------------------------------------------- #

@dataclass
class Section:
    """One block of the report: a table of interactions of a single family."""
    key: str
    title: str
    header: list
    row_func: Callable


# Column headers shared by the σ-hole families.
_SIGMA_HOLE_HEADER = [
    "№", "Type", "Confidence score",
    "D", "X", "A",
    "D-X (Å)", "X···A (Å)", "D···A (Å)",
    "Angle DXA (°)", "Angle RAX (°)",
    "Energy, kcal/mol",
]


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def output(
    G: nx.Graph,
    filename: str = "default",
    ext: str = ".default",
    correlation_hb=universal_1_hb,
    correlation_sb=universal_1_sb,
) -> None:
    """
    Write a formatted ``.nci`` report summarising all detected interactions.

    Parameters
    ----------
    G : nx.Graph
        Molecular graph produced by the detectors.
    filename : str
        Base filename *without* extension.  The report is written to
        ``f"{filename}.nci"``.
    ext : str
        Original input extension (e.g. ``".xyz"``), only used in the
        report header.
    correlation_hb, correlation_sb
        Empirical correlations used to convert geometries into energies.
    """
    edges_by_type = _group_edges_by_type(G)
    sections = _build_sections(correlation_hb, correlation_sb)

    with open(f"{filename}.nci", "w", encoding="utf-8") as f:
        _write_header(f, filename, ext)

        for section in sections:
            rows = [
                section.row_func(edge, i)
                for i, edge in enumerate(
                    edges_by_type.get(section.key, [])
                )
            ]
            if rows:
                _write_section(f, section, rows)

        _write_quote(f)


# --------------------------------------------------------------------------- #
#  Edge grouping and section construction
# --------------------------------------------------------------------------- #

def _group_edges_by_type(G: nx.Graph) -> dict:
    """Bucket edges of ``G`` by their ``bond_type`` attribute."""
    grouped = {}
    for _, _, data in G.edges(data=True):
        btype = data.get("bond_type")
        if btype is None:
            continue
        grouped.setdefault(btype, []).append(data)
    return grouped


def _build_sections(correlation_hb, correlation_sb) -> list:
    """Construct the ordered list of report sections."""
    return [
        _sigma_hole_section("HB", "HYDROGEN BONDS",
                            lambda d: correlation_hb(
                                d["length_xb"],
                                f"{d['symbol_d']}-H···{d['symbol_a']}",
                            )),
        _sigma_hole_section("XB", "HALOGEN BONDS",
                            lambda d: correlation_sb(
                                d["length_xb"],
                                f"{d['symbol_x']}{d['symbol_a']}",
                            )),
        _sigma_hole_section("ChB", "CHALCOGEN BONDS",
                            lambda d: correlation_sb(
                                d["length_xb"],
                                f"{d['symbol_x']}{d['symbol_a']}",
                            )),
        _sigma_hole_section("PnB", "PNICTOGEN BONDS",
                            lambda d: correlation_sb(
                                d["length_xb"],
                                f"{d['symbol_x']}{d['symbol_a']}",
                            )),
        _sigma_hole_section("TetB", "TETREL BONDS",
                            lambda d: correlation_sb(
                                d["length_xb"],
                                f"{d['symbol_x']}{d['symbol_a']}",
                            )),
        _stacking_section(),
        _h_pi_section(),
        _n_pi_section(),
        _ion_pi_section(),
        _metallophilic_section(),
        _carbonyl_section(),
    ]


# --------------------------------------------------------------------------- #
#  Row builders for each family
# --------------------------------------------------------------------------- #

def _sigma_hole_section(
    key: str,
    title: str,
    correlation: Callable,
) -> Section:
    """
    Build a σ-hole section (HB, XB, ChB, PnB, TetB).

    ``correlation`` receives the edge data and returns the estimated
    interaction energy in kcal/mol.
    """
    def row_func(d, i):
        return [
            i + 1,
            f"{d['symbol_d']}-{d['symbol_x']}···{d['symbol_a']}",
            round(d["confidence_score"], 2),
            int(d["d_num"]) + 1,
            int(d["x_num"]) + 1,
            int(d["a_num"]) + 1,
            round(d["length_cov"], 3),
            round(d["length_xb"], 3),
            round(d["length_da"], 3),
            round(d["angle"], 1),
            "—" if d["angle_xay"] is None else round(d["angle_xay"], 1),
            round(float(correlation(d)), 1),
        ]

    return Section(key, title, _SIGMA_HOLE_HEADER, row_func)


def _stacking_section() -> Section:
    header = [
        "№", "Confidence score", "Ring 1", "Ring 2",
        "Centroid (Å)", "Offset (Å)", "Angle (°)",
    ]

    def row_func(d, i):
        return [
            i + 1,
            round(d["confidence_score"], 2),
            _cycle_to_string(d["atom_num_cycle_1"]),
            _cycle_to_string(d["atom_num_cycle_2"]),
            round(d["centroid_distance"], 3),
            round(d["offset"], 3),
            round(d["angle"], 1),
        ]

    return Section("stacking", "PARALLEL STACKING", header, row_func)


def _h_pi_section() -> Section:
    header = [
        "№", "Type", "Confidence score",
        "X", "H", "Ring",
        "H···π (Å)", "Angle (°)", "Projection (Å)",
    ]

    def row_func(d, i):
        return [
            i + 1,
            f"{d['symbol_x']}-H···π",
            round(d["confidence_score"], 2),
            int(d["x_num"]) + 1,
            int(d["h_num"]) + 1,
            _cycle_to_string(d["cycle"]),
            round(d["h_centroid_distance"], 3),
            round(d["angle"], 1),
            round(d["projection_distance"], 3),
        ]

    return Section("H-pi", "X-H···π INTERACTIONS", header, row_func)


def _n_pi_section() -> Section:
    header = [
        "№", "Type", "Confidence score",
        "Atom", "Ring", "Distance (Å)", "Angle (°)",
    ]

    def row_func(d, i):
        return [
            i + 1,
            f"{d['symbol_lp']}···π",
            round(d["confidence_score"], 2),
            int(d["lp_num"]) + 1,
            _cycle_to_string(d["cycle"]),
            round(d["lp_centroid_distance"], 3),
            "—" if d["angle"] is None else round(d["angle"], 1),
        ]

    return Section("n-pi", "LONE PAIR···π", header, row_func)


def _ion_pi_section() -> Section:
    header = [
        "№", "Type", "Confidence score",
        "Atom", "Ring", "Distance (Å)", "Angle (°)",
    ]

    def row_func(d, i):
        return [
            i + 1,
            f"{d['symbol_ion']}···π",
            round(d["confidence_score"], 2),
            int(d["ion_num"]) + 1,
            _cycle_to_string(d["cycle"]),
            round(d["ion_centroid_distance"], 3),
            "—" if d["angle_to_normal"] is None
            else round(d["angle_to_normal"], 1),
        ]

    return Section("ion-pi", "ION···π", header, row_func)


def _metallophilic_section() -> Section:
    header = [
        "№", "Type", "Confidence score",
        "Atom 1", "Atom 2", "Distance (Å)",
    ]

    def row_func(d, i):
        return [
            i + 1,
            f"{d['metal_sym_1']}···{d['metal_sym_2']}",
            round(d["confidence_score"], 2),
            int(d["metal_1"]) + 1,
            int(d["metal_2"]) + 1,
            round(d["distance"], 3),
        ]

    return Section("metallophilic", "METALLOPHILIC-LIKE INTERACTIONS",
                   header, row_func)


def _carbonyl_section() -> Section:
    header = [
        "№", "Type", "Confidence score",
        "Donor O", "Acceptor C",
        "Distance (Å)", "BD angle (°)", "Plane angle (°)",
    ]

    def row_func(d, i):
        return [
            i + 1,
            "n→π*",
            round(d["confidence_score"], 2),
            int(d["donor_oxygen"]) + 1,
            int(d["acceptor_carbon"]) + 1,
            round(d["distance"], 3),
            round(d["burgi_dunitz_angle"], 1),
            "—" if d["plane_angle"] is None
            else round(d["plane_angle"], 1),
        ]

    return Section("carbonyl", "CARBONYL n→π*", header, row_func)


# --------------------------------------------------------------------------- #
#  Low-level writing
# --------------------------------------------------------------------------- #

def _cycle_to_string(cycle) -> str:
    """Format a ring as a comma-separated list of 1-based atom indices."""
    return ",".join(str(int(i) + 1) for i in np.asarray(cycle))


def _write_header(f, filename: str, ext: str) -> None:
    now = datetime.now()
    f.write(
        "NCITools: Geometry based analysis\n"
        "Authors: Kaplanskiy M.V., Tupikina E.Yu., Sutkin V.S., Rudenko V.A.\n"
        f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Input file: {filename}{ext}\n"
    )


def _write_section(f, section: Section, rows: list) -> None:
    f.write("\n")
    f.write("=" * 155 + "\n")
    f.write(section.title + "\n")
    f.write("=" * 155 + "\n\n")

    table = tabulate(
        rows,
        headers=section.header,
        tablefmt="github",
    #    floatfmt=".3f",
        numalign="right",
        stralign="center",
    )
    f.write(table)
    f.write("\n")


def _write_quote(f) -> None:
    philosopher, quote = random.choice(quotes)
    f.write("\n\n")
    f.write(
        textwrap.fill(
            quote.upper(),
            width=40,
            initial_indent=" ",
            subsequent_indent=" ",
        )
    )
    f.write(f"\n   -- {philosopher.upper()}\n")