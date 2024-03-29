"""Test cases for the convolve.py module."""
from typing import Any

import numpy as np
import pyfftw
import pytest
import scipy.fftpack

from pyeisen import convolve


def test_centered() -> None:
    """Define a test for centered function."""
    c = convolve.centered(17, 7)
    assert c == slice(5, 12)


@pytest.mark.parametrize("boundary", [None, "mirror"])
@pytest.mark.parametrize("interp_nan", [True, False])
@pytest.mark.parametrize("fftpack", [scipy.fftpack, pyfftw.builders])
def test_fconv_fftw(fftpack: Any, boundary: None | str, interp_nan: bool) -> None:
    """Ensure that fconv can run with different parameter combinations."""
    k = np.arange(5)[:, None].astype(np.complex_)
    s = np.random.randn(100, 3)

    csig = convolve.fconv(
        k, s, fftpack=fftpack, boundary=boundary, interp_nan=interp_nan
    )
    assert csig.shape[1] == 1
    assert csig.shape[2] == 3
    assert csig.shape[0] == s.shape[0]
    assert True
