"""
The `quotonic.logic` module includes functions required to conduct basic modeling of qubits and logic gate
operations when resolved in the computational basis. It is by no means comprehensive, only including functionalities
that are helpful to the other modules within `quotonic` ([training_sets](training_sets.md) in particular).
"""

from functools import reduce

import numpy as np
import numpy.typing as npt

from quotonic.types import np_ndarray


def build_comp_basis(n: int) -> np_ndarray:
    """Generate the computational basis for a given number of qubits.

    The array formed lists the possible combinations of qubit states (i.e. 0s and 1s) for a given number of qubits,
    $n$. To understand the ordering, please see the examples below.

    Args:
        n: number of qubits, $n$

    Returns:
        $N\\times n$ array that catalogs all states in the $N$-dimensional computational basis

    Examples:
        ```
        >>> build_comp_basis(1)
        array([[0],
               [1]])

        >>> build_comp_basis(2)
        array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])

        >>> build_comp_basis(3)
        array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
        ```
    """

    # compute the dimension of the computational basis
    N = 2**n

    # leverage the relation between computational basis states and binary to create the states
    basis = np.zeros((N, n), dtype=int)
    for i in range(N):
        basis[i] = np.array([int(j) for j in format(i, "0" + str(n) + "b")])

    return basis


def H(n: int = 1) -> np_ndarray:
    """Generate the matrix representation of $n$ Hadamard gates applied to $n$ qubits individually.

    Constructs and returns the matrix $H^{\\otimes n}$, where

    $$ H = \\frac{1}{\\sqrt{2}} \\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \\end{pmatrix}. $$

    Args:
        n: number of qubits, $n$

    Returns:
        Matrix representation of a Hadamard gate, as a $2\\times 2 array

    Examples:
        ```
        >>> H()
        array([[ 0.70710678+0.j,  0.70710678+0.j],
               [ 0.70710678+0.j, -0.70710678+0.j]])
        >>> H(n=2)
        array([[ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j],
               [ 0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j],
               [ 0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j],
               [ 0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5-0.j]])
        ```
    """
    mat_1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    mat = reduce(np.kron, [mat_1] * n)
    return mat  # type: ignore


def CNOT(control: int = 0, target: int = 1, n: int = 2) -> np_ndarray:
    """Generate the matrix representation of a CNOT gate between the specified control and target qubits.

    For any number of qubits, this function will form the matrix representation of a single CNOT gate between a
    specified control qubit and a specified target qubit. By default, it forms the familiar

    $$ \\mathrm{CNOT} = \\begin{pmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \\\\ 0 & 0 & 1 & 0
    \\end{pmatrix}. $$

    Args:
        control: index of the control qubit
        target: index of the target qubit
        n: total number of qubits, $n$

    Returns:
        Matrix representation of a CNOT gate between the control and target qubits, as a $2^n\\times 2^n$ array

    Examples:
        ```
        >>> CNOT()
        array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
        >>> CNOT(control=0, target=1, n=3)
        array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
        ```
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


def CZ(control: int = 0, target: int = 1, n: int = 2) -> np_ndarray:
    """Generate the matrix representation of a CZ gate between the specified control and target qubits.

    For any number of qubits, this function will form the matrix representation of a single CZ gate between a
    specified control qubit and a specified target qubit. By default, it forms the familiar

    $$ \\mathrm{CZ} = \\begin{pmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & -1
    \\end{pmatrix}. $$

    Args:
        control: index of the control qubit
        target: index of the target qubit
        n: total number of qubits, $n$

    Returns:
        Matrix representation of a CZ gate between the control and target qubits, as a $2^n\\times 2^n$ array

    Examples:
        ```
        >>> CZ()
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])
        >>> CZ(control=0, target=1, n=3)
        array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, 0.+0.j],
               [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, 0.+0.j],
               [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, 0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, 0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j, 0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j, 0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j, 0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])
        ```
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


def BSA() -> np_ndarray:
    """Generate the matrix representation of a Bell State Analyzer (BSA) in the computational basis.

    The BSA operates on two qubits and is defined as

    $$ \\mathrm{BSA} = (H \\otimes I)\\mathrm{CNOT}. $$

    Returns:
        Matrix representation of a BSA in the computational basis, as a $4\\times 4$ array
    """
    mat = np.kron(H(), np.eye(2, dtype=complex)) @ CNOT()
    return mat  # type: ignore
