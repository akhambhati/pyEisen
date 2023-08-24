"""Test cases for the family.py module."""
import pytest

from pyeisen import family
import numpy as np


@pytest.fixture
def define_family() -> family.Family:
    """Define a test Family typed dict."""
    fam: family.Family = {
        "kernel": np.array([[3 + 5j]]),
        "params": {"frequency": [5]},
        "sample": {"timestamp": np.array([0.1])},
        "axis_ord": ("sample", "n_kernel"),
    }
    return fam


def test_family_definition(define_family: family.Family) -> None:
    """Confirm the defined_family key, value pairs match."""
    assert define_family["kernel"] == np.array([[3 + 5j]])
    assert define_family["params"] == {"frequency": [5]}
    assert define_family["sample"] == {"timestamp": np.array([0.1])}
    assert define_family["axis_ord"] == ("sample", "n_kernel")
    assert True


def test_morlet() -> None:
    wv_fam = family.morlet(
        freqs=np.array([0.5]), cycles=np.array([6]), fs=1.0, n_win=7, complete=True
    )
    assert type(wv_fam) == dict
    assert wv_fam["kernel"].shape == (1, 14)
    assert np.isclose(wv_fam["params"]["scales"], 1 / 0.583333333333333)
    assert wv_fam["params"]["freqs"] == np.array([0.5])
    assert wv_fam["params"]["cycles"] == np.array([6])
    assert wv_fam["sample"]["time"].shape == (14,)
    assert "axis_ord" in wv_fam
