"""
The `quotonic.fock` module includes functions required to generate a list of states that together define either the
first or second-quantized basis for a provided number of photons $n$ and number of optical modes $m$.

Much of this code is inpsired by and adapted from [Bosonic: A Quantum Optics Library](
https://github.com/steinbrecher/bosonic), as originally designed for use in [G. R. Steinbrecher *et al*.,
“Quantum optical neural networks”, *npj Quantum Inf* **5**, 60 (2019)](https://doi.org/10.1038/s41534-019-0174-7).
"""

from functools import cache
from itertools import combinations_with_replacement, product

import numpy as np

from quotonic.types import np_ndarray


@cache
def calc_firq_dim(n: int, m: int) -> int:
    """Calculate the dimension of the first quantization basis.

    Given a number of photons $n$ and a number of optical modes $m$, this function efficiently computes the
    dimension of the corresponding first-quantized basis. The result is cached to ensure that this function is not
    evaluated redundantly for a constant number of photons and optical modes.

    The dimension of the basis is given by $N = m^n$.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        N: dimenstion of the first quantization basis, $N$
    """
    N: int = m**n
    return N


@cache
def calc_secq_dim(n: int, m: int) -> int:
    """Calculate the dimension of the second quantization Fock basis.

    Given a number of photons $n$ and a number of optical modes $m$, this function efficiently computes the
    dimension of the corresponding Fock basis. The result is cached to ensure that this function is not evaluated
    redundantly for a constant number of photons and optical modes.

    The dimension of the Fock basis is given by $N = {n+m-1 \\choose n}$. This operation can be expressed
    alternatively as,

    $$ N = {n+m-1 \\choose n} = \\frac{(n+m-1)!}{n!(m-1)!} = \\frac{(n+m-1)(n+m-2)\\dots m}{n(n-1)\\dots 1}, $$

    which simplifies the algorithm.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        N: dimenstion of the second quantization Fock basis, $N$
    """

    # store the top of {n + m - 1 \\choose n}
    top = n + m - 1

    # evaluate the simplified version of {n + m - 1 \\choose n}
    i = 0
    dim = 0
    numerator = 1
    denominator = 1
    while top - i >= m:
        numerator *= top - i
        i += 1
        denominator *= i
    dim += numerator // denominator

    return dim


@cache
def build_firq_basis(n: int, m: int) -> np_ndarray:
    """Generate a catalog of all states in the first quantization basis.

    Given a number of photons $n$ and a number of optical modes $m$, this function creates each state in the basis
    as an array of length $n$ where each element is an integer corresponding to the optical mode of the
    corresponding photon. All the states are then placed in a 2D array that contains the entire basis. This function
    caches results to ensure that it is not evaluated redundantly for a constant number of photons and optical modes.

    The states are computed from all combinations of the available optical modes, then converted to an array. An
    example result is displayed below for $n = 2$ photons and $m = 4$ optical modes:

    ```
    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1],
    [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]]
    ```

    $$ n = 2, m = 4 \\implies \\{\\left|00\\right\\rangle, \\left|01\\right\\rangle, \\left|02\\right\\rangle,
    \\left|03\\right\\rangle, \\left|10\\right\\rangle, \\left|11\\right\\rangle, \\left|12\\right\\rangle,
    \\left|13\\right\\rangle, \\left|20\\right\\rangle, \\left|21\\right\\rangle, \\left|22\\right\\rangle,
    \\left|23\\right\\rangle, \\left|30\\right\\rangle, \\left|31\\right\\rangle, \\left|32\\right\\rangle,
    \\left|33\\right\\rangle\\} $$

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        $N\\times n$ array that catalogs all states in the $N$-dimensional first quantization basis
    """

    return np.array(list(product(range(m), repeat=n)))


@cache
def build_firq_basis_wo_dups(n: int, m: int) -> np_ndarray:
    """Generate a catalog of all states in the first quantization basis, avoiding indistinguishable duplicates,
    where each is denoted with $n$ slots where each slot specifies which of the $m$ modes the photon resides in.

    Given a number of photons $n$ and a number of optical modes $m$, this function creates each state in the basis
    as an array of length $n$ where each element is an integer corresponding to the optical mode of the
    corresponding photon. All the states are then placed in a 2D array that contains the entire basis. This function
    caches results to ensure that it is not evaluated redundantly for a constant number of photons and optical modes.

    The states are computed from all combinations of the available optical modes, removing duplicate permutations,
    then converts them to an array. With the duplicates removed, the dimension of this basis is the same as the
    second-quantized Fock basis. An example result is displayed below for $n = 2$ photons and $m = 4$ optical modes:

    ```
    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]
    ```

    $$ n = 2, m = 4 \\implies \\{\\left|00\\right\\rangle, \\left|01\\right\\rangle, \\left|02\\right\\rangle,
    \\left|03\\right\\rangle, \\left|11\\right\\rangle, \\left|12\\right\\rangle, \\left|13\\right\\rangle,
    \\left|22\\right\\rangle, \\left|23\\right\\rangle, \\left|33\\right\\rangle\\} $$

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        $N\\times n$ array that catalogs all states in the $N$-dimensional first quantization basis,
            when indistinguishable duplicates are removed
    """
    return np.array(list(combinations_with_replacement(range(m), n)))


@cache
def build_secq_basis(n: int, m: int) -> np_ndarray:
    """Generate a catalog of all states in the second quantization Fock basis, denoted with $m$ slots where each slot
    specifies the number of photons residing in the corresponding mode.

    Given a number of photons $n$ and a number of optical modes $m$, this function creates each state in the Fock basis
    as an array of length $m$ where each element is an integer corresponding to the number of photons occupying a
    given optical mode. All the states are then placed in a 2D array that contains the entire basis. This function
    caches results to ensure that it is not evaluated redundantly for a constant number of photons and optical modes.

    All possible combinations of the modes for the given number of photons are first computed and stored in
    `firq_basis` as a list of tuples of the form `[(photon 1 mode, photon 2 mode, ..., photon n mode), ...]`. For
    each element of the `firq_basis`, a Fock basis state is generated by counting the number of photons in each mode
    and creating a list of those counts. Each of these lists corresponding to Fock basis states are appended to the
    full `fock_basis` 2D array. Example results are displayed below for $n = 2$ photons and $m = 4$ optical modes:

    ```
    >>> firq_basis
    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]

    >>> fock_basis
    [[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 2, 0, 0],
    [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 2, 0], [0, 0, 1, 1], [0, 0, 0, 2]]
    ```

    $$ n = 2, m = 4 \\implies \\{\\left|2000\\right\\rangle, \\left|1100\\right\\rangle, \\left|1010\\right\\rangle,
    \\left|1001\\right\\rangle, \\left|0200\\right\\rangle, \\left|0110\\right\\rangle, \\left|0101\\right\\rangle,
    \\left|0020\\right\\rangle, \\left|0011\\right\\rangle, \\left|0002\\right\\rangle\\} $$

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        $N\\times m$ array that catalogs all states in the $N$-dimensional second quantization Fock basis
    """

    # initialize array to store the catalog of basis states
    N = calc_secq_dim(n, m)
    fock_basis = np.zeros((N, m), dtype=int)

    # generate a list of tuples of all combinations of the modes for a given number of photons
    firq_basis = build_firq_basis_wo_dups(n, m)

    # for each combination of modes, compute the number of photons in each mode
    # and insert an array that is representative of the Fock basis state
    for i in range(N):
        fock_basis[i, :] = np.bincount(firq_basis[i], minlength=m)

    return fock_basis
