import pytest
import numpy as np
from ncitools.geometry.detectors import check_spatial_clashes
from ncitools.geometry import build_graph_with_covalent_pairs, find_nci_bonds

def test_water_dimer_has_no_clashes(water_dimer):
    """Valid geometry must produce zero clash warnings."""
    G = build_graph_with_covalent_pairs(water_dimer)
    G = find_nci_bonds(water_dimer, G, ["H"], 120, "HB")
    warnings = check_spatial_clashes(water_dimer, G)
    assert len(warnings) == 0

def test_clashing_h2_dimer_detected(clashing_h2_dimer):
    """Close covalent bonds from separate molecules must trigger a warning."""
    G = build_graph_with_covalent_pairs(clashing_h2_dimer)
    warnings = check_spatial_clashes(clashing_h2_dimer, G)
    assert len(warnings) == 1

def test_covalent_neighborhood_screens_adjacent_bonds(water_dimer_with_2_HB):
    """
    Two HBs originating from covalently bonded atoms must be screened and
    not reported as a clash (even if the distance between the bonds is close enough).
    """
    G = build_graph_with_covalent_pairs(water_dimer_with_2_HB)
    G = find_nci_bonds(water_dimer_with_2_HB, G, ["H"], 120, "HB")

    warnings = check_spatial_clashes(water_dimer_with_2_HB, G)
    assert len(warnings) == 0






