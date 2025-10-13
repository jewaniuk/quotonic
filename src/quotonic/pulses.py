"""
The `quotonic.pulses` module includes ...
"""


import numpy as np
import numpy.typing as npt


def gaussian_t(t: np.ndarray, t0: float, sigt: float) -> np.ndarray:
    """Compute the single-photon Gaussian wavefunction in the temporal domain.

    Args:
        t: array that specifies the time domain, contains all times, $t$, to evaluate the wavefunction at
        t0: center time of the Gaussian, $t_0$
        sigt: temporal width of the Gaussian, $\\sigma_t$

    Returns:
        Gaussian wavefunction in the temporal domain, taking the same shape as the domain itself
    """
    alph: npt.NDArray[np.complex128] = ((2 / (np.pi * (sigt**2))) ** 0.25) * np.exp(
        -((t - t0) ** 2) / (sigt**2), dtype=complex
    )
    return alph


def gaussian_w(w: np.ndarray, w0: float, sigw: float) -> np.ndarray:
    """Compute the single-photon Gaussian wavefunction in the frequency domain.

    Args:
        w: array that specifies the frequency domain, contains all angular frequencies, $\\omega$, to evaluate the wavefunction at
        w0: center frequency of the Gaussian, $\\omega_0$
        sigw: spectral width of the Gaussian, $\\sigma_\\omega$

    Returns:
        Gaussian wavefunction in the frequency domain, taking the same shape as the domain itself
    """
    alph: npt.NDArray[np.complex128] = ((2 * np.pi * (sigw**2)) ** -0.25) * np.exp(
        -0.25 * ((w - w0) ** 2) / (sigw**2), dtype=complex
    )
    return alph
