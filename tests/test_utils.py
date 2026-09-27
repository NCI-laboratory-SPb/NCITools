from ncitools.utils import get_symbol


def test_get_symbol_known():
    assert get_symbol(1) == "H"
    assert get_symbol(6) == "C"
    assert get_symbol(8) == "O"


def test_get_symbol_unknown():
    assert get_symbol(999) == "X"
    assert get_symbol(0) == "X"