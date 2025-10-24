"""
The `quotonic.utils` module includes miscellaneous helper functions for a variety of other modules within `quotonic`.
"""

import numpy as np
from jax import vmap
from jax.scipy.special import factorial

from quotonic.types import np_ndarray


def genHaarUnitary(m: int) -> np_ndarray:
    """Generate an $m\\times m$ unitary sampled randomly from the Haar measure.

    This function follows the procedure outlined in [F. Mezzadri, “How to generate random matrices from classical
    compact groups”, arXiv:math-ph/0609050v2 (2007)](https://arxiv.org/abs/math-ph/0609050).

    Args:
        m: dimension of the square $m \\times m$ unitary

    Returns:
        A 2D array storing the Haar random $m\\times m$ unitary
    """

    z = np.random.randn(m, m) + 1j * np.random.randn(m, m) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    Lambda = d / np.abs(d)
    U: np_ndarray = np.multiply(q, Lambda)
    return U


def comp_to_secq(comp_state: np_ndarray) -> np_ndarray:
    """Convert a computational basis state to its corresponding second-quantized Fock basis state in dual-rail encoding.

    In dual-rail encoding, a qubit is defined by the location of a single photon in two consecutive optical modes.
    Mathematically, this can be written as $\\left|0\\right\\rangle_\\mathrm{log} \\equiv \\left|10\\right\\rangle$ and
    $\\left|1\\right\\rangle_\\mathrm{log} \\equiv \\left|01\\right\\rangle$. This function takes a computational basis
    state, iterates through each qubit, and inserts the corresponding representation in the second-quantized Fock basis.

    Args:
        comp_state: $n$-length array, where $n$ is the number of qubits, where each element specifies whether a qubit
            is 0 or 1

    Returns:
        Fock basis state that corresponds to the given computational basis state by dual-rail encoding, a $2n$-length
        array

    Examples:
        ```python
        >>> comp_to_secq(np.array([0]))
        array([1, 0])
        >>> comp_to_secq(np.array([0, 1]))
        array([1, 0, 0, 1])
        ```
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


def secq_to_comp(fock_state: np_ndarray) -> np_ndarray:
    """Convert a Fock basis state, that is dual-rail encoded, to its corresponding computational basis state.

    In dual-rail encoding, a qubit is defined by the location of a single photon in two consecutive optical modes.
    Mathematically, this can be written as $\\left|0\\right\\rangle_\\mathrm{log} \\equiv \\left|10\\right\\rangle$ and
    $\\left|1\\right\\rangle_\\mathrm{log} \\equiv \\left|01\\right\\rangle$. This function takes a second-quantized
    Fock basis state, iterates through each qubit, and inserts the corresponding 0 or 1 from the computational basis.

    Args:
        fock_state: $2n$-length array, where $n$ is the number of qubits, where each consecutive pair of elements
            signifies whether a qubit is 0 or 1

    Returns:
        Computational basis state that corresponds to the given Fock basis state by dual-rail encoding, an
            $n$-length array

    Examples:
        ```python
        >>> secq_to_comp(np.array([1, 0]))
        array([0])
        >>> secq_to_comp(np.array([1, 0, 0, 1]))
        array([0, 1])
        ```
    """

    # check the validity of the provided symmetric Fock basis state
    assert len(fock_state) > 0, "The provided symmetric Fock basis state must have elements"
    assert min(fock_state) >= 0, "Symmetric Fock basis states do not have negative labels"
    assert len(fock_state) % 2 == 0, "The provided symmetric Fock basis state is not dual-rail encoded"

    # for each pair of consecutive slots in the Fock state, insert the corresponding slot to the computational state
    n = len(fock_state) // 2
    comp_state = np.zeros(n, dtype=int)
    for i, j in zip(range(n), range(0, 2 * n, 2)):
        assert sum(fock_state[j : j + 2]) == 1, "The provided symmetric Fock basis state is not dual-rail encoded"
        comp_state[i] = 0 if (fock_state[j : j + 2] == np.array([1, 0])).all() else 1
    return comp_state


def comp_indices_from_secq(fock_basis: np_ndarray, ancillary_modes: np_ndarray | None = None) -> np_ndarray:
    """Extract the indices of Fock basis states that correspond to computational basis states by dual-rail encoding.

    This function iterates through each second-quantized Fock basis states, ignores the specified ancillary modes if
    provided (these are not involved with logical encoding), checks if the state corresponds to the dual-rail encoding,
    and stores the index of this state within the Fock basis if it is deemed logical. The list of indices is returned
    as an array.

    Args:
        fock_basis: $N\\times 2n$ array, where $n$ is the number of qubits, that catalogs all states in the
            $N$-dimensional second quantization Fock basis
        ancillary_modes: array that specifies which optical modes are ancillary and thus should not contribute to
            logical encoding

    Returns:
        $2^n$-length array whose elements are the indices of the second quantization Fock basis where dual-rail encoded
            computational basis states lie

    Examples:
        ```python
        >>> from quotonic.fock import build_secq_basis
        >>> n = 2
        >>> m = 4
        >>> fock_basis = build_secq_basis(n, m)
        >>> fock_basis
        array([[2, 0, 0, 0],
               [1, 1, 0, 0],
               [1, 0, 1, 0],
               [1, 0, 0, 1],
               [0, 2, 0, 0],
               [0, 1, 1, 0],
               [0, 1, 0, 1],
               [0, 0, 2, 0],
               [0, 0, 1, 1],
               [0, 0, 0, 2]])
        >>> comp_indices_from_secq(fock_basis)
        array([2, 3, 5, 6])
        ```
    """

    # check the validity of the provided Fock basis
    n = int(np.amax(fock_basis))
    state_slots = (fock_basis.shape[1] - len(ancillary_modes)) if ancillary_modes is not None else fock_basis.shape[1]
    assert n * 2 == state_slots, "The provided symmetric Fock basis cannot be dual-rail encoded"

    # for each Fock basis state, remove ancillary modes, then check that each consecutive pair of slots sums to 1
    indices = np.zeros(2**n, dtype=int)
    i = 0
    for s, state in enumerate(fock_basis):
        reduced_state = np.delete(state, ancillary_modes) if ancillary_modes is not None else state
        if (np.sum(reduced_state.reshape((n, 2)), axis=1) == np.ones(n)).all():
            indices[i] = s
            i += 1
    return indices


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
