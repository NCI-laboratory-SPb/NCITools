"""
Geometry-based detection of non-covalent interactions.

Pipeline
--------
    atoms -> build_graph_with_covalent_pairs -> detectors -> output

Modules
-------
* :mod:`~ncitools.geometry.soft`          — soft (fuzzy) threshold functions.
* :mod:`~ncitools.geometry.geom`          — pure-geometry helpers.
* :mod:`~ncitools.geometry.graph_builder` — molecular graph construction.
* :mod:`~ncitools.geometry.detectors`     — one module per interaction family.
* :mod:`~ncitools.geometry.output`        — formatted ``.nci`` report writer.
"""

from ncitools.geometry.detectors import (
    find_carbonyl_interactions,
    find_intermetalls_contact,
    find_nci_bonds,
    find_nci_with_aromatic,
)
from ncitools.geometry.graph_builder import build_graph_with_covalent_pairs
from ncitools.geometry.output import output

__all__ = [
    "build_graph_with_covalent_pairs",
    "find_nci_bonds",
    "find_nci_with_aromatic",
    "find_carbonyl_interactions",
    "find_intermetalls_contact",
    "output",
]