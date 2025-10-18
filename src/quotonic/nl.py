"""
The `quotonic.nl` module includes functions required to construct the transfer matrix of a specified set of
single-site optical nonlinearities across $m$ optical modes.

All code in this module is inspired by [Bosonic: A Quantum Optics Library](https://github.com/steinbrecher/bosonic),
as originally designed for use in [G. R. Steinbrecher *et al*., “Quantum optical neural networks”,
*npj Quantum Inf* **5**, 60 (2019)](https://doi.org/10.1038/s41534-019-0174-7), and by [Cascaded Optical Systems
Approach to Neural Networks (CasOptAx)](https://github.com/JasvithBasani/CasOptAx) as originally designed for use in
[J. R. Basani *et al*., "Universal logical quantum photonic neural network processor via cavity-assisted
interactions", *npj Quantum Inf* **11**, 142 (2025)](https://doi.org/10.1038/s41534-025-01096-9).
"""

import numpy as np

from quotonic.fock import build_firq_basis, build_secq_basis
from quotonic.types import np_ndarray


def build_kerr(
    n: int,
    m: int,
    varphi: float | np_ndarray,
    basis_type: str = "secq",
    burnout_map: np_ndarray | None = None,
) -> np_ndarray:
    """Construct the diagonal nonlinear Kerr unitary in the relevant basis.

    This function constructs the diagonal unitary matrix, $\\boldsymbol{\\Sigma}(\\varphi)$, corresponding to
    single-site Kerr nonlinearities of strength $\\varphi$ applied across a set of $m$ optical modes. If all
    single-site nonlinearities are applied, then this matrix is expressed mathematically as,

    $$ \\boldsymbol{\\Sigma}(\\varphi) = \\sum_{n=0}^\\infty e^{in(n-1)\\frac{\\varphi}{
    2}}\\left|n\\right\\rangle\\left\\langle n\\right|. $$

    From this form, it is evident that $\\boldsymbol{\\Sigma}(\\varphi)$ is the $N\\times N$ identity matrix (
    $\\mathbf{I}_N$), where $N$ is the Fock basis dimension, in cases of $n < 2$, where $n$ is the number of photons.
    The $n$ is the previous expression is better explained as the number of photons in the optical mode for which a
    single-site optical Kerr nonlinearity is applied. This is best displayed by example. In the case where $n=3$, $m=2$,
    the Fock basis has a dimension of $N = 4$ with basis states $\\left\\{\\left|30\\right\\rangle,
    \\left|21\\right\\rangle, \\left|12\\right\\rangle, \\left|03\\right\\rangle\\right\\}$. As evident from the
    expression above, given the orthonormality of the Fock basis, the only nonzero elements lie on the diagonal of
    the unitary $\\boldsymbol{\\Sigma}(\\varphi)$. Consider the calculation of the element located at the row and
    column both corresponding to state $\\left|21\\right\\rangle$,

    $$ \\left\\langle 21 \\right|(\\boldsymbol{\\Sigma}(\\varphi)\\otimes\\mathbf{I}_N) (\\mathbf{
    I}_N\\otimes\\boldsymbol{\\Sigma}(\\varphi))\\left| 21 \\right\\rangle = \\left[\\sum_{n=0}^\\infty e^{in(
    n-1)\\frac{\\varphi}{2}}\\left\\langle 2|n \\right\\rangle\\left\\langle n|2 \\right\\rangle\\right] \\left[
    \\sum_{n=0}^\\infty e^{in(n-1)\\frac{\\varphi}{2}}\\left\\langle 1|n \\right\\rangle\\left\\langle n|1
    \\right\\rangle\\right] = \\left[e^{i\\varphi}\\right]\\left[e^{i(0)}\\right] = e^{i\\varphi}. $$

    From the tensor products, it is evident that the Kerr nonlinearities are single-site and thus applied by mode.
    After completing the calculation of the other elements along the diagonal, the resulting unitary is given as
    follows,

    $$ \\boldsymbol{\\Sigma}(\\varphi) = \\begin{pmatrix} e^{i3\\varphi} & 0 & 0 & 0 \\\\ 0 & e^{i\\varphi} & 0 & 0
    \\\\ 0 & 0 & e^{i\\varphi} & 0 \\\\ 0 & 0 & 0 & e^{i3\\varphi} \\end{pmatrix}, $$

    where the ordering of the rows and columns follows that in which the basis states were listed previously. The
    previous process can be modified by *burning out* some of the single-site nonlinearities. This is conducted by
    passing an array of binary/boolean values to the function in `burnoutMap`. This 1D array has $m$ elements,
    each respectively telling the function whether the single-site nonlinearity at a specific mode is on or off.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$
        varphi: effective nonlinear phase shifts for each single-site nonlinear element in $\\text{rad}$, $\\varphi$,
            a float when all $m$ are the same, otherwise an $m$-length array
        basis_type: specifies whether the unitary should be constructed in the first or second-quantized basis
        burnout_map: array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are
            on/off for specific modes

    Returns:
        Sigma: $N\\times N$ array, the matrix representation of the set of single-site Kerr nonlinearities resolved in
            the relevant basis
    """

    # check that basis_type is valid
    assert (basis_type == "secq") or (basis_type == "firq"), "Basis type must be 'secq' or 'firq'"

    # if varphi has been provided as a float, then all elements have the same nonlinear phase shift
    if isinstance(varphi, float):
        varphi = varphi * np.ones(m, dtype=float)

    # check if burnoutMap has been provided, otherwise, choose default (all nonlinearities applied)
    if burnout_map is None:
        burnout_map = np.ones(m)

    # build basis for the given numbers of photons and optical modes
    basis = build_secq_basis(n, m) if basis_type == "secq" else build_firq_basis(n, m)
    N = basis.shape[0]

    # initialize the diagonal of the Kerr unitary \Sigma(\phi)
    Sigma = np.ones(N, dtype=complex)

    # if the number of photons is 0 or 1, then the Kerr unitary is an identity matrix
    if n < 2:
        return np.diag(Sigma)

    for i, state in enumerate(basis):
        # calculate the number of photons in each optical mode
        photons_per_mode = state if basis_type == "secq" else np.bincount(state, minlength=m)

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
    basis_type: str = "secq",
    burnout_map: np_ndarray | None = None,
) -> np_ndarray:
    """Construct the diagonal nonlinear $\\Lambda$-type 3LS photon $\\mp$ unitary in the relevant basis.

    This function constructs the diagonal unitary matrix, $\\boldsymbol{\\Sigma}(\\varphi)$, corresponding to
    single-site optical nonlinearities applied across a set of $m$ optical modes via a cavity-assisted interaction with
    a $\\Lambda$-type 3LS. It is designed in analogy with `build_kerr`, so refer to that documentation for further
    details. The only difference is that there are two controllable phase shift parameters, $\\varphi_1$ and
    $\\varphi_2$. When $n$ photons enter each nonlinear component, one of them is deterministically subtracted from
    the rest, picking up a phase shift of $\\varphi_1$, while the remaining $n - 1$ each pick up a phase shift of
    $\\varphi_2$. This is outlined in further detail in the publication that proposed these nonlinear components,
    [J. R. Basani *et al*., "Universal logical quantum photonic neural network processor via cavity-assisted
    interactions", *npj Quantum Inf* **11**, 142 (2025)](https://doi.org/10.1038/s41534-025-01096-9).

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$
        varphi1: phase shift applied to the subtracted photon in $\\text{rad}$, $\\varphi_1$
        varphi2: phase shift applied to the remaining photons in $\\text{rad}$, $\\varphi_2$
        basis_type: specifies whether the unitary should be constructed in the first or second-quantized basis
        burnout_map: array of length $m$, with either boolean or binary elements, specifying whether nonlinearities are
            on/off for specific modes

    Returns:
        Sigma: $N\\times N$ array, the matrix representation of the set of single-site photon $\\mp$ nonlinearities
            resolved in the relevant basis
    """

    # check that basis_type is valid
    assert (basis_type == "secq") or (basis_type == "firq"), "Basis type must be 'secq' or 'firq'"

    # check if burnoutMap has been provided, otherwise, choose default (all nonlinearities applied)
    if burnout_map is None:
        burnout_map = np.ones(m)

    # build basis for the given numbers of photons and optical modes
    basis = build_secq_basis(n, m) if basis_type == "secq" else build_firq_basis(n, m)
    N = basis.shape[0]

    # initialize the diagonal of the nonlinear unitary \Sigma(\phi)
    Sigma = np.ones(N, dtype=complex)

    # if the number of photons is 0, then the nonlinear unitary is an identity matrix; if 1, then apply varphi1 shift
    if n == 0:
        return np.diag(Sigma)
    elif n < 2:
        return np.diag(Sigma * np.exp(1j * varphi1))

    for i, state in enumerate(basis):
        # calculate the number of photons in each optical mode
        photons_per_mode = state if basis_type == "secq" else np.bincount(state, minlength=m)

        phase = 0
        # for each basis state, sum the phase shifts from each optical mode
        for mode in range(m):
            if photons_per_mode[mode] > 0 and burnout_map[mode] == 1:
                phase += varphi1 + (photons_per_mode[mode] - 1) * varphi2

        Sigma[i] = np.exp(1j * phase)

    # return N x N diagonal nonlinear unitary matrix
    return np.diag(Sigma)
