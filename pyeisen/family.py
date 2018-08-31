"""
Functions and Wrappers to define wavelet families.

Author: Ankit N. Khambhati
Adapted from: https://github.com/pennmem/ptsa_new/blob/master/ptsa/wavelet.py
Last Updated: 2018/08/31
"""

import numpy as np
from scipy.signal import morlet as scipy_morlet


def morlet(freqs, cycles, Fs, n_win=7, complete=True):
    """
    Calculate Morlet wavelets with the total energy normalized to 1.

    Calls the scipy.signal.wavelet.morlet() function to generate
    Morlet wavelets with the specified frequencies, samplerates, and
    widths (in cycles); see the docstring for the scipy morlet function
    for details. These wavelets are normalized before they are returned.

    Parameters
    ----------
    freqs: np.ndarray, shape: (n_wavelet, dtype=float)
        The center frequency (e.g. 10 Hz) for each wavelet.
    cycles: np.ndarray, shape (n_wavelet, dtype=float)
        The number of cycles for each wavelet.
        Needs to be same size as freqs.
    Fs: float
        The sampling frequency (e.g. 200 Hz) of the signal to which wavelet
        will be applied.
    n_win: float, (default=7)
        Length of the wavelet that will be sampled (usually >= 7).
        Provides a multiplicative factor for time sampling based on
        the requested cycles at the center frequency.
    complete : bool, (default=True)
        Whether to generate a complete or standard approximation to
        the complete version of a Morlet wavelet. Complete should be True,
        especially for low (<=5) values of width. See
        scipy.signal.wavelet.morlet() for details.

    Returns
    -------
    family: np.ndarray, shape: (n_wavelet, n_samples)
        A family of Morlet wavelets equal in number to the frequency/cycle
        pairs provided. Each wavelet entry spans same length but diff. decay.
    """

    # Check Inputs:
    if len(freqs) < 1:
        raise ValueError('At least one frequency must be specified.')
    if len(cycles) != len(freqs):
        raise ValueError(
            'Each frequency must have an associated number of cycles.')
    if Fs <= 0:
        raise ValueError('Sampling frequency must be non-negative.')

    # Temporal standard deviation of the wavelet (ratio of cycles to frequency)
    st = cycles / (2 * np.pi * freqs)

    # Support length for the wavelet (length in samples)
    max_len = np.max(np.ceil(st * Fs * n_win))

    # Scaling factor for the wavelet.
    # Depends on frequency, cycles, support length, and sampling frequency.
    scales = (freqs * max_len) / (2 * cycles * Fs)

    # generate list of unnormalized wavelets:
    family = [
        scipy_morlet(max_len, w=cycles[i], s=scales[i], complete=complete)
        for i in range(len(scales))
    ]

    # generate list of energies for the wavelets:
    energies = [np.sqrt(np.sum(np.abs(wavelet)**2)) for wavelet in family]

    # normalize the wavelets by dividing each one by its energy:
    family = [family[i] / energies[i] for i in range(len(family))]

    # Convert to an array bank
    family = np.array(family)

    # Sort all vals to the scale
    scales = 1 / scales
    scale_ord = np.argsort(scales)

    wavelet_family = {
        'wavelets': family[scale_ord],
        'scales': scales[scale_ord],
        'freqs': freqs[scale_ord],
        'cycles': cycles[scale_ord]
    }

    return wavelet_family
