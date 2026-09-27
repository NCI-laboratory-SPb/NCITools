"""
Interaction detectors.

Every detector takes an :class:`ase.Atoms` object and the molecular graph
produced by
:func:`ncitools.geometry.graph_builder.build_graph_with_covalent_pairs`,
and adds edges with a specific ``bond_type`` to the graph.
"""

from ncitools.geometry.detectors.aromatic import find_nci_with_aromatic
from ncitools.geometry.detectors.carbonyl import find_carbonyl_interactions
from ncitools.geometry.detectors.metallophilic import find_intermetalls_contact
from ncitools.geometry.detectors.sigma_hole import find_nci_bonds

__all__ = [
    "find_nci_bonds",
    "find_nci_with_aromatic",
    "find_carbonyl_interactions",
    "find_intermetalls_contact",
]