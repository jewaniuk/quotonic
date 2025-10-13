"""
The `quotonic.nl` module includes ...
"""

from typing import Optional, Union

import numpy as np

from quotonic.fock import build_asymm_basis, build_symm_basis


def build_kerr(
    n: int,
    m: int,
    varphi: Union[float, np.ndarray],
    basis_type: str = "symmetric",
    burnout_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct the diagonal nonlinear Kerr unitary in the relevant Fock basis.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$
        varphi: effective nonlinear phase shifts for each single-site nonlinear element in $\\text{rad}$, $\\varphi$, a float when all $m$ are the same, otherwise an $m$-length array
        basis_type: specifies whether the unitary should be constructed in the symmetric or asymmetric Fock basis
        burnout_map: array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are on/off for specific modes

    Returns:
        $N\\times N$ array, the matrix representation of the set of single-site Kerr nonlinearities resolved in the relevant Fock basis
    """

    # check that basis_type is valid
    assert (basis_type == "symmetric") or (basis_type == "asymmetric"), "Basis type must be 'symmetric' or 'asymmetric'"

    # if varphi has been provided as a float, then all elements have the same nonlinear phase shift
    if isinstance(varphi, float):
        varphi = varphi * np.ones(m, dtype=float)

    # check if burnoutMap has been provided, otherwise, choose default (all nonlinearities applied)
    if burnout_map is None:
        burnout_map = np.ones(m)

    # build Fock basis for the given numbers of photons and optical modes
    basis = build_symm_basis(n, m) if basis_type == "symmetric" else build_asymm_basis(n, m)
    N = basis.shape[0]

    # initialize the diagonal of the Kerr unitary \Sigma(\phi)
    Sigma = np.ones(N, dtype=complex)

    # if the number of photons is 0 or 1, then the Kerr unitary is an identity matrix
    if n < 2:
        return np.diag(Sigma)

    for i, state in enumerate(basis):
        # calculate the number of photons in each optical mode
        photons_per_mode = state if basis_type == "symmetric" else np.bincount(state, minlength=m)

        phase = 0
        # for each basis state, sum the phase shifts from each optical mode
        for mode in range(m):
            if photons_per_mode[mode] > 1 and burnout_map[mode] == 1:
                phase += photons_per_mode[mode] * (photons_per_mode[mode] - 1) * varphi[mode] * 0.5

        Sigma[i] = np.exp(1j * phase)

    # return N x N diagonal Kerr unitary matrix
    return np.diag(Sigma)


def build_photon_mp(
    n: int,
    m: int,
    varphi1: float,
    varphi2: float,
    basis_type: str = "symmetric",
    burnout_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct the diagonal nonlinear $\\Lambda$-type 3LS photon $\\mp$ unitary in the relevant Fock basis.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$
        varphi1: phase shift applied to the subtracted photon in $\\text{rad}$, $\\varphi_1$
        varphi2: phase shift applied to the remaining photons in $\\text{rad}$, $\\varphi_2$
        basis_type: specifies whether the unitary should be constructed in the symmetric or asymmetric Fock basis
        burnout_map: array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are on/off for specific modes

    Returns:
        $N\\times N$ array, the matrix representation of the set of single-site photon $\\mp$ nonlinearities resolved in the relevant Fock basis
    """

    # check that basis_type is valid
    assert (basis_type == "symmetric") or (basis_type == "asymmetric"), "Basis type must be 'symmetric' or 'asymmetric'"

    # check if burnoutMap has been provided, otherwise, choose default (all nonlinearities applied)
    if burnout_map is None:
        burnout_map = np.ones(m)

    # build Fock basis for the given numbers of photons and optical modes
    basis = build_symm_basis(n, m) if basis_type == "symmetric" else build_asymm_basis(n, m)
    N = basis.shape[0]

    # initialize the diagonal of the Kerr unitary \Sigma(\phi)
    Sigma = np.ones(N, dtype=complex)

    # if the number of photons is 0 or 1, then the Kerr unitary is an identity matrix
    if n < 2:
        return np.diag(Sigma * np.exp(1j * varphi1))

    for i, state in enumerate(basis):
        # calculate the number of photons in each optical mode
        photons_per_mode = state if basis_type == "symmetric" else np.bincount(state, minlength=m)

        phase = 0
        # for each basis state, sum the phase shifts from each optical mode
        for mode in range(m):
            if photons_per_mode[mode] > 0 and burnout_map[mode] == 1:
                phase += varphi1 + (photons_per_mode[mode] - 1) * varphi2

        Sigma[i] = np.exp(1j * phase)

    # return N x N diagonal Kerr unitary matrix
    return np.diag(Sigma)
