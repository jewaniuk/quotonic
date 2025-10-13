"""
The `quotonic.logic` module includes ...
"""

from functools import reduce

import numpy as np
import numpy.typing as npt


def build_comp_basis(n: int) -> np.ndarray:
    """Generate the computational basis for a given number of qubits.

    Args:
        n: number of qubits, $n$

    Returns:
        $N\\times n$ array that catalogs all states in the $N$-dimensional computational basis
    """

    # compute the dimension of the computational basis
    N = 2**n

    # leverage the relation between computational basis states and binary to create the states
    basis = np.zeros((N, n), dtype=int)
    for i in range(N):
        basis[i] = np.array([int(j) for j in format(i, "0" + str(n) + "b")])

    return basis


def H(n: int = 1) -> np.ndarray:
    """Generate the matrix representation of a $n$ Hadamard gates applied to $n$ qubits individually.

    Args:
        n: number of qubits, $n$

    Returns:
        Matrix representation of a Hadamard gate, as a $2\\times 2 array
    """
    mat_1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    mat: npt.NDArray[np.complex128] = reduce(np.kron, [mat_1] * n)
    return mat


def CNOT(control: int = 0, target: int = 1, n: int = 2) -> np.ndarray:
    """Generate the matrix representation of a CNOT gate between the specified control and target qubits.

    Args:
        control: index of the control qubit
        target: index of the target qubit
        n: total number of qubits, $n$

    Returns:
        Matrix representation of a CNOT gate between the control and target qubits, as a $2^n\\times 2^n$ array
    """
    assert (n > control) and (
        n > target
    ), "Check that you are indexing correctly and that you are passing the correct number of qubits"
    assert control != target, "Control and target qubits should be different"

    # define relevant single-qubit gates
    Id = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    # define single-qubit states and their projectors
    ket0 = Id[0].reshape(2, 1)
    ket1 = Id[1].reshape(2, 1)
    proj0 = np.kron(ket0, ket0.T)
    proj1 = np.kron(ket1, ket1.T)

    # apply the appropriate Kronecker product based on each qubit
    term1 = np.array([1], dtype=complex)
    term2 = np.array([1], dtype=complex)
    for i in reversed(range(n)):
        if i == control:
            term1 = np.kron(proj0, term1)
            term2 = np.kron(proj1, term2)
        elif i == target:
            term1 = np.kron(Id, term1)
            term2 = np.kron(X, term2)
        else:
            term1 = np.kron(Id, term1)
            term2 = np.kron(Id, term2)

    mat: npt.NDArray[np.complex128] = term1 + term2
    return mat


def CZ(control: int = 0, target: int = 1, n: int = 2) -> np.ndarray:
    """Generate the matrix representation of a CZ gate between the specified control and target qubits.

    Args:
        control: index of the control qubit
        target: index of the target qubit
        n: total number of qubits, $n$

    Returns:
        Matrix representation of a CZ gate between the control and target qubits, as a $2^n\\times 2^n$ array
    """
    assert (n > control) and (
        n > target
    ), "Check that you are indexing correctly and that you are passing the correct number of qubits"
    assert control != target, "Control and target qubits should be different"

    # define relevant single-qubit gates
    Id = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # define single-qubit states and their projectors
    ket0 = Id[0].reshape(2, 1)
    ket1 = Id[1].reshape(2, 1)
    proj0 = np.kron(ket0, ket0.T)
    proj1 = np.kron(ket1, ket1.T)

    # apply the appropriate Kronecker product based on each qubit
    term1 = np.array([1], dtype=complex)
    term2 = np.array([1], dtype=complex)
    for i in reversed(range(n)):
        if i == control:
            term1 = np.kron(proj0, term1)
            term2 = np.kron(proj1, term2)
        elif i == target:
            term1 = np.kron(Id, term1)
            term2 = np.kron(Z, term2)
        else:
            term1 = np.kron(Id, term1)
            term2 = np.kron(Id, term2)

    mat: npt.NDArray[np.complex128] = term1 + term2
    return mat


def BSA() -> np.ndarray:
    """Generate the matrix representation of a Bell State Analyzer in the computational basis.

    Returns:
        Matrix representation of a Bell State Analyzer in the computational basis, as a $4\\times 4$ array
    """
    mat: npt.NDArray[np.complex128] = np.kron(H(), np.eye(2, dtype=complex)) @ CNOT()
    return mat
