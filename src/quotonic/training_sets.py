"""
The `quotonic.training_sets` module includes ...
"""

from functools import reduce
from itertools import combinations
from typing import Tuple

import numpy as np

import quotonic.logic as logic
from quotonic.fock import build_secq_basis
from quotonic.utils import comp_indices_from_secq
from quotonic.types import np_ndarray


def CNOT() -> Tuple[np_ndarray, np_ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based CNOT gate, resolved in the second-quantized Fock basis.

    ADD DOCUMENTATION HERE

    Returns:
        A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the
            $N$-dimensional second quantization Fock basis, the second of which contains the corresponding target states
    """

    # define number of photons, optical modes, and input-target state pairs,
    # which is equivalent to the dimension of the computational basis in this case
    n = 2
    m = 4
    K = 2**n

    # build the N-dim Fock basis and a list of indices in it where the corresponding computational basis states lie
    fock_basis = build_secq_basis(n, m)
    N = fock_basis.shape[0]
    comp_indices = comp_indices_from_secq(fock_basis)

    # all the input states in the computational basis can be extracted from the identity matrix of the same dimension
    psi_in_comps = np.eye(K, dtype=complex)

    # build the gate in the computational basis to easily compute the target states in the computational basis
    gate = logic.CNOT()

    # for each input-target pair, compute the target in the computational basis, then convert both to the Fock basis
    psi_in = np.zeros((K, N), dtype=complex)
    psi_targ = np.zeros((K, N), dtype=complex)
    for k, psi_in_comp in enumerate(psi_in_comps):
        psi_targ_comp = gate @ psi_in_comp

        psi_in[k, comp_indices] = psi_in_comp
        psi_targ[k, comp_indices] = psi_targ_comp

    return psi_in, psi_targ


def CZ() -> Tuple[np_ndarray, np_ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based CZ gate, resolved in the second-quantized Fock basis.

    ADD DOCUMENTATION HERE

    Returns:
        A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the
            $N$-dimensional second quantization Fock basis, the second of which contains the corresponding target states
    """

    # define number of photons, optical modes, and input-target state pairs,
    # which is equivalent to the dimension of the computational basis in this case
    n = 2
    m = 4
    K = 2**n

    # build the N-dim Fock basis and a list of indices in it where the corresponding computational basis states lie
    fock_basis = build_secq_basis(n, m)
    N = fock_basis.shape[0]
    comp_indices = comp_indices_from_secq(fock_basis)

    # all the input states in the computational basis can be extracted from the identity matrix of the same dimension
    psi_in_comps = np.eye(K, dtype=complex)

    # build the gate in the computational basis to easily compute the target states in the computational basis
    gate = logic.CZ()

    # for each input-target pairs, compute the target in the computational basis, then convert both to the Fock basis
    psi_in = np.zeros((K, N), dtype=complex)
    psi_targ = np.zeros((K, N), dtype=complex)
    for k, psi_in_comp in enumerate(psi_in_comps):
        psi_targ_comp = gate @ psi_in_comp

        psi_in[k, comp_indices] = psi_in_comp
        psi_targ[k, comp_indices] = psi_targ_comp

    return psi_in, psi_targ


def BSA() -> Tuple[np_ndarray, np_ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based Bell State Analyzer, resolved in the
        second-quantized Fock basis.

    ADD DOCUMENTATION HERE

    Returns:
        A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the
            $N$-dimensional second quantization Fock basis, the second of which contains the corresponding target states
    """

    # define number of photons, optical modes, and input-target state pairs,
    # which is equivalent to the dimension of the computational basis in this case
    n = 2
    m = 4
    K = 2**n

    # build the N-dim Fock basis and a list of indices in it where the corresponding computational basis states lie
    fock_basis = build_secq_basis(n, m)
    N = fock_basis.shape[0]
    comp_indices = comp_indices_from_secq(fock_basis)

    # generate the input states in the computational basis
    c00, c01, c10, c11 = np.eye(K, dtype=complex)
    psi_in_comps = np.zeros((K, K), dtype=complex)
    psi_in_comps[0] = (c00 + c11) / np.sqrt(2)
    psi_in_comps[1] = (c00 - c11) / np.sqrt(2)
    psi_in_comps[2] = (c01 + c10) / np.sqrt(2)
    psi_in_comps[3] = (c01 - c10) / np.sqrt(2)

    # build the gate in the computational basis to easily compute the target states in the computational basis
    gate = logic.BSA()

    # for each input-target pair, compute the target in the computational basis, then convert both to the Fock basis
    psi_in = np.zeros((K, N), dtype=complex)
    psi_targ = np.zeros((K, N), dtype=complex)
    for k, psi_in_comp in enumerate(psi_in_comps):
        psi_targ_comp = gate @ psi_in_comp

        psi_in[k, comp_indices] = psi_in_comp
        psi_targ[k, comp_indices] = psi_targ_comp

    return psi_in, psi_targ


def Tree(b: int) -> tuple:  # noqa: C901
    """Construct the training set for a QPNN that powers a tree-type photonic cluster state generation protocol.

    ADD DOCUMENTATION HERE

    Args:
        b: number of branches in the tree, $b$

    Returns:
        Tuple of two tuples of $K\\times N$ arrays, the first of which contains the $K$ input states resolved in the
            $N$-dimensional second quantization Fock basis, the second of which contains the corresponding target
            states, for each $1 \\leq n \\leq b + 1$
    """

    # define the number of photons and number of optical modes
    n = b + 1
    m = 2 * n

    # define the relevant single-qubit states that will be used to construct each input in the computational basis
    Id = np.eye(2, dtype=complex)
    ket0 = Id[0].reshape(2, 1)
    ket1 = Id[1].reshape(2, 1)
    ketp = logic.H() @ ket0
    ketm = logic.H() @ ket1

    # construct the training set for each number of photons 1 <= n <= b + 1
    psi_in = []
    psi_targ = []
    for _n in range(1, n + 1):
        # construct the target circuit in the computational basis
        if _n == 1:
            # build the training set, keeping in kind that only |+> must be routed through the top modes of the network
            K = 1
            psi_in_n = np.zeros((K, m), dtype=complex)
            psi_targ_n = np.zeros((K, m), dtype=complex)

            fock_basis = build_secq_basis(_n, m)
            for i in range(2):
                state = np.zeros(m)
                state[i] = 1
                s = int(np.where((fock_basis == state).all(axis=1))[0].item())
                psi_in_n[0, s] = 1 / np.sqrt(2)
                psi_targ_n[0, s] = 1 / np.sqrt(2)

            psi_in.append(psi_in_n)
            psi_targ.append(psi_targ_n)

        else:
            CZs = [logic.CZ(control=0, target=i, n=_n) for i in range(1, _n)]
            circuit = reduce(np.dot, CZs[::-1])

            # build the training set, keeping in mind that the top qubit is |+> for all input states,
            # and the other two qubits are swept over the X and Z computational bases
            comp_basis = logic.build_comp_basis(_n - 1)
            all_missing_slots = list(combinations(list(range(m - 2, 0, -2)), n - _n)) if n != _n else [(0,)]
            K = 2 * (comp_basis.shape[0]) * len(all_missing_slots)
            fock_basis = build_secq_basis(_n, m)
            N = fock_basis.shape[0]
            psi_in_n = np.zeros((K, N), dtype=complex)
            psi_targ_n = np.zeros((K, N), dtype=complex)
            for i, basis in enumerate(["X", "Z"]):
                for j, comp_state in enumerate(comp_basis):
                    # construct input state in the computational basis, then get output by applying the circuit
                    psi_in_comp = np.copy(ketp)
                    for qubit in comp_state:
                        if basis == "X":
                            psi_in_comp = np.kron(psi_in_comp, ketp) if qubit == 0 else np.kron(psi_in_comp, ketm)
                        elif basis == "Z":
                            psi_in_comp = np.kron(psi_in_comp, ket0) if qubit == 0 else np.kron(psi_in_comp, ket1)
                    psi_targ_comp = circuit @ psi_in_comp

                    if _n == n:
                        # convert input and output states to their corresponding represenation in the Fock basis
                        comp_inds = comp_indices_from_secq(fock_basis)
                        psi_in_n[i * comp_basis.shape[0] + j, comp_inds] = psi_in_comp.reshape((2**n,))
                        psi_targ_n[i * comp_basis.shape[0] + j, comp_inds] = psi_targ_comp.reshape((2**n,))
                    else:
                        for k, missing_slots in enumerate(all_missing_slots):
                            # convert input and output states to their corresponding represenation in the Fock basis
                            missing_modes = []
                            for slot in missing_slots:
                                missing_modes.extend([slot, slot + 1])
                            comp_inds = comp_indices_from_secq(fock_basis, ancillary_modes=np.array(missing_modes))
                            psi_in_n[
                                i * comp_basis.shape[0] * len(all_missing_slots) + j * len(all_missing_slots) + k,
                                comp_inds,
                            ] = psi_in_comp.reshape((2**_n,))
                            psi_targ_n[
                                i * comp_basis.shape[0] * len(all_missing_slots) + j * len(all_missing_slots) + k,
                                comp_inds,
                            ] = psi_targ_comp.reshape((2**_n,))

            psi_in.append(psi_in_n)
            psi_targ.append(psi_targ_n)

    return tuple(psi_in), tuple(psi_targ)
