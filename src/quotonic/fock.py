"""
The `quotonic.fock` module includes ...
"""

from functools import cache
from itertools import combinations_with_replacement, product

import numpy as np


@cache
def calc_symm_dim(n: int, m: int) -> int:
    """Calculate the dimension of the symmetric Fock basis.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        Dimenstion of the symmetric Fock basis, $N$
    """

    # store the top of {n + m - 1 \choose n}
    top = n + m - 1

    # evaluate the simplified version of {n + m - 1 \choose n}
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
def build_symm_mode_basis(n: int, m: int) -> np.ndarray:
    """Generate a catalog of all states in the symmetric Fock basis, denoted with $n$ slots where each slot specifies which mode $m$ the photon resides in.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        $N\\times n$ array that catalogs all states in the $N$-dimensional symmetric Fock basis, expressed in mode-specifying form
    """
    return np.array(list(combinations_with_replacement(range(m), n)))


@cache
def build_symm_basis(n: int, m: int) -> np.ndarray:
    """Generate a catalog of all states in the symmetric Fock basis.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        $N\\times m$ array that catalogs all states in the $N$-dimensional symmetric Fock basis
    """

    # initialize array to store the catalog of basis states
    N = calc_symm_dim(n, m)
    fockBasis = np.zeros((N, m), dtype=int)

    # generate a list of tuples of all combinations of the modes for a given number of photons
    modeBasis = build_symm_mode_basis(n, m)

    # for each combination of modes, compute the number of photons in each mode
    # and insert an array that is representative of the Fock basis state
    for i in range(N):
        fockBasis[i, :] = np.bincount(modeBasis[i], minlength=m)

    return fockBasis


@cache
def calc_asymm_dim(n: int, m: int) -> int:
    """Calculate the dimension of the asymmetric Fock basis.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        Dimenstion of the asymmetric Fock basis, $N$
    """
    N: int = m**n
    return N


@cache
def build_asymm_basis(n: int, m: int) -> np.ndarray:
    """Generate a catalog of all states in the asymmetric Fock basis.

    Args:
        n: number of photons, $n$
        m: number of optical modes, $m$

    Returns:
        $N\\times n$ array that catalogs all states in the $N$-dimensional asymmetric Fock basis
    """

    return np.array(list(product(range(m), repeat=n)))
