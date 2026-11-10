"""Topological analysis of electron density: BCP-based NCI classification.

Public API
----------
The module re-exports the pieces that are used by the CLI so that client
code can import everything from a single place::

    from ncitools.topology import (
        read_cube, BSplineAIM,
        find_bcps_parallel, find_nci,
        find_nci_with_aromatic,
        find_sigma_hole_interactions,
        find_carbonyl_interactions,
        find_unclassified_interactions,
        prepare_graph_for_nci,
        output as topology_output,
    )

The covalent graph itself is *not* built here — use
``ncitools.geometry.graph_builder.build_graph_with_covalent_pairs`` and then
run ``prepare_graph_for_nci`` to convert it into the units / metadata
expected by the detectors.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Density model, BCP search, topological classification
# ---------------------------------------------------------------------------
from .topology import (
    # CUBE reader
    CubeData,
    read_cube,
    # B-spline density model
    BSplineAIM,
    # BCP search / classification
    classify_hessian,
    classify_covalent_like,
    deduplicate_bcps,
    filter_nci_artifacts,
    find_bcps_parallel,
    find_bcp_parallel,        # backward-compatible alias
    find_nci,
    generate_candidate_pairs,
    newton_stationary_point,
    pair_alignment_cosine,
    pair_geometry,
    positive_hessian_eigenvector,
    valid_bcp_pair_geometry,
)

# ---------------------------------------------------------------------------
# NCI detectors
# ---------------------------------------------------------------------------
from .detectors.aromatic import find_nci_with_aromatic
from .detectors.sigma_hole import find_sigma_hole_interactions
from .detectors.carbonyl import find_carbonyl_interactions
from .detectors.unclassified import find_unclassified_interactions

# Bridge between geometry.graph_builder output and the topology detectors.
from .detectors._common import _prepare_graph_for_nci as prepare_graph_for_nci

# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
from .output import output


__all__ = [
    # --- CUBE reader ---
    "CubeData",
    "read_cube",
    # --- Density model / BCP search ---
    "BSplineAIM",
    "classify_hessian",
    "classify_covalent_like",
    "deduplicate_bcps",
    "filter_nci_artifacts",
    "find_bcps_parallel",
    "find_bcp_parallel",
    "find_nci",
    "generate_candidate_pairs",
    "newton_stationary_point",
    "pair_alignment_cosine",
    "pair_geometry",
    "positive_hessian_eigenvector",
    "valid_bcp_pair_geometry",
    # --- Detectors ---
    "find_nci_with_aromatic",
    "find_sigma_hole_interactions",
    "find_carbonyl_interactions",
    "find_unclassified_interactions",
    "prepare_graph_for_nci",
    # --- Report ---
    "output",
]