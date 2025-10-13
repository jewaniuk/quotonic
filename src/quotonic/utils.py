"""
The `quotonic.utils` module includes ...
"""

from itertools import permutations
from typing import Optional

import numpy as np
from jax import vmap
from jax.scipy.special import factorial


def genHaarUnitary(m: int) -> np.ndarray:
    """Generate an $m\\times m$ unitary sampled randomly from the Haar measure.

    This function follows the procedure outlined in [F. Mezzadri, “How to
    generate random matrices from classical compact groups”, arXiv:math-ph/0609050v2
    (2007)](https://arxiv.org/abs/math-ph/0609050).

    Args:
        m: dimension of the square $m \\times m$ unitary

    Returns:
        A 2D array storing the Haar random $m\\times m$ unitary
    """

    z = np.random.randn(m, m) + 1j * np.random.randn(m, m) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    Lambda = d / np.abs(d)
    U: np.ndarray = np.multiply(q, Lambda)
    return U


def comp_to_symm_fock(comp_state: np.ndarray) -> np.ndarray:
    """Convert a computational basis state to its corresponding symmetric Fock basis state in dual-rail encoding.

    Args:
        comp_state: $n$-length array, where $n$ is the number of qubits, where each element specifies whether a qubit is 0 or 1

    Returns:
        Symmetric Fock basis state that corresponds to the given computational basis state by dual-rail encoding, a $2n$-length array
    """

    # check the validity of the provided computational basis state
    assert max(comp_state) < 2, "The provided computational basis state is invalid"
    assert len(comp_state) > 0, "The provided computational basis state must have elements"
    assert min(comp_state) >= 0, "Computational basis states do not have negative labels"

    # for each slot of the computational basis state, insert the corresponding slots to the Fock state
    n = len(comp_state)
    fock_state = np.zeros(2 * n, dtype=int)
    for i, j in zip(range(n), range(0, 2 * n, 2)):
        fock_state[j : j + 2] = np.array([1, 0]) if comp_state[i] == 0 else np.array([0, 1])
    return fock_state


def symm_fock_to_comp(fock_state: np.ndarray) -> np.ndarray:
    """Convert a symmetric Fock basis state, that is dual-rail encoded, to its corresponding computational basis state.

    Args:
        fock_state: $2n$-length array, where $n$ is the number of qubits, where each consecutive pair of elements signifies whether a qubit is 0 or 1

    Returns:
        Computational basis state that corresponds to the given symmetric Fock basis state by dual-rail encoding, an $n$-length array
    """

    # check the validity of the provided symmetric Fock basis state
    assert len(fock_state) > 0, "The provided symmetric Fock basis state must have elements"
    assert min(fock_state) >= 0, "Symmetric Fock basis states do not have negative labels"
    assert len(fock_state) % 2 == 0, "The provided symmetric Fock basis state is not dual-rail encoded"

    # for each pair of consecutive slots in the symmetric Fock basis state, insert the corresponding slot to the computational basis state
    n = len(fock_state) // 2
    comp_state = np.zeros(n, dtype=int)
    for i, j in zip(range(n), range(0, 2 * n, 2)):
        assert sum(fock_state[j : j + 2]) == 1, "The provided symmetric Fock basis state is not dual-rail encoded"
        comp_state[i] = 0 if (fock_state[j : j + 2] == np.array([1, 0])).all() else 1
    return comp_state


def comp_indices_from_symm_fock(fock_basis: np.ndarray, ancillary_modes: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract the indices of symmetric Fock basis states that correspond to computational basis states by dual-rail encoding.

    Args:
        fock_basis: $N\\times 2n$ array, where $n$ is the number of qubits, that catalogs all states in the $N$-dimensional symmetric Fock basis
        ancillary_modes: array that specifies which optical modes are ancillary and thus should not contribute to logical encoding

    Returns:
        $2^n$-length array whose elements are the indices of the symmetric Fock basis where dual-rail encoded computational basis states lie
    """

    # check the validity of the provided Fock basis
    n = int(np.amax(fock_basis))
    state_slots = (fock_basis.shape[1] - len(ancillary_modes)) if ancillary_modes is not None else fock_basis.shape[1]
    assert n * 2 == state_slots, "The provided symmetric Fock basis cannot be dual-rail encoded"

    # for each symmetric Fock basis state, remove ancillary modes, then check that each consecutive pair of slots sums to 1
    indices = np.zeros(2**n, dtype=int)
    i = 0
    for s, state in enumerate(fock_basis):
        reduced_state = np.delete(state, ancillary_modes) if ancillary_modes is not None else state
        if (np.sum(reduced_state.reshape((n, 2)), axis=1) == np.ones(n)).all():
            indices[i] = s
            i += 1
    return indices


def comp_indices_from_asymm_fock(
    asymm_basis: np.ndarray,
    symm_basis: np.ndarray,
    symm_mode_basis: np.ndarray,
    ancillary_modes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract the indices of asymmetric Fock basis states that correspond to computational basis states by dual-rail encoding.

    Args:
        asymm_basis: $N_a\\times n$ array, where $n$ is the number of qubits, that catalogs all states in the $N_a$-dimensional asymmetric Fock basis
        symm_basis: $N_s\\times 2n$ array, where $n$ is the number of qubits, that catalogs all states in the $N_s$-dimensional symmetric Fock basis
        symm_mode_basis: $N_s\\times n$ array, where $n$ is the number of qubits, that catalogs all states in the $N_s$-dimensional symmetric Fock basis, expressed in mode-specifying form
        ancillary_modes: array that specifies which optical modes are ancillary and thus should not contribute to logical encoding

    Returns:
        $2^n\\times n!$ array where the $n!$ permutations in the asymmetric Fock basis that correspond to each of the $2^n$ computational basis states are catalogued
    """

    # check the validity of the provided asymmetric Fock basis
    n = asymm_basis.shape[1]
    m = np.amax(asymm_basis) + 1
    K = 2**n
    state_slots = m - len(ancillary_modes) if ancillary_modes is not None else m
    assert state_slots == 2 * n, "The provided asymmetric Fock basis cannot be dual-rail encoded"

    # extract the computational basis state indices from the symmetric Fock basis
    comp_indices_symm = comp_indices_from_symm_fock(symm_basis, ancillary_modes=ancillary_modes)

    # for each computational basis state, find all relevant permutations in the asymmetric Fock basis
    comp_indices_asymm = np.zeros((K, int(factorial(n))), dtype=int)
    for i, sym_ind in enumerate(comp_indices_symm):
        symm_mode_state = symm_mode_basis[sym_ind]
        asymm_states = np.array(list(permutations(symm_mode_state)))
        for j, asymm_state in enumerate(asymm_states):
            comp_indices_asymm[i, j] = int(np.where((asymm_basis == asymm_state).all(axis=1))[0].item())
    return comp_indices_asymm


@vmap
def vectorial_factorial(x: int | float) -> int | float:
    """Compute the factorial on the input vectorially.

    Simply put, this function wraps `jax.scipy.special.factorial` with the `jax.vmap` decorator. It doesn't really
    need its own documented function, but I thought the name `vectorial_factorial` sounded cool.

    Args:
        x: integer to compute the factorial of

    Returns:
        Factorial of the input
    """
    return factorial(x)  # type: ignore
