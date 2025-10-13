"""
The `quotonic.training_sets` module includes ...
"""

from functools import reduce
from itertools import combinations
from typing import Tuple

import numpy as np

import quotonic.logic as logic
from quotonic.fock import build_asymm_basis, build_symm_basis, build_symm_mode_basis
from quotonic.pulses import gaussian_t
from quotonic.utils import comp_indices_from_asymm_fock, comp_indices_from_symm_fock


def CNOT() -> Tuple[np.ndarray, np.ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based CNOT gate, resolved in the symmetric Fock basis.

    Returns:
        A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states
    """

    # define number of photons, optical modes, and input-target state pairs, which is equivalent to the dimension of the computational basis in this case
    n = 2
    m = 4
    K = 2**n

    # build the N-dimensional symmetric Fock basis and a list of indices within this basis where the corresponding computational basis states lie
    fock_basis = build_symm_basis(n, m)
    N = fock_basis.shape[0]
    comp_indices = comp_indices_from_symm_fock(fock_basis)

    # all the input states in the computational basis can simply be extracted from the identity matrix of the same dimension
    psi_in_comps = np.eye(K, dtype=complex)

    # build the gate in the computational basis to easily compute the target states in the computational basis
    gate = logic.CNOT()

    # for each of the input-target state pairs, compute the target state in the computational basis, then convert both to the symmetric Fock basis
    psi_in = np.zeros((K, N), dtype=complex)
    psi_targ = np.zeros((K, N), dtype=complex)
    for k, psi_in_comp in enumerate(psi_in_comps):
        psi_targ_comp = gate @ psi_in_comp

        psi_in[k, comp_indices] = psi_in_comp
        psi_targ[k, comp_indices] = psi_targ_comp

    return (psi_in, psi_targ)


def CZ() -> Tuple[np.ndarray, np.ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based CZ gate, resolved in the symmetric Fock basis.

    Returns:
        A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states
    """

    # define number of photons, optical modes, and input-target state pairs, which is equivalent to the dimension of the computational basis in this case
    n = 2
    m = 4
    K = 2**n

    # build the N-dimensional symmetric Fock basis and a list of indices within this basis where the corresponding computational basis states lie
    fock_basis = build_symm_basis(n, m)
    N = fock_basis.shape[0]
    comp_indices = comp_indices_from_symm_fock(fock_basis)

    # all the input states in the computational basis can simply be extracted from the identity matrix of the same dimension
    psi_in_comps = np.eye(K, dtype=complex)

    # build the gate in the computational basis to easily compute the target states in the computational basis
    gate = logic.CZ()

    # for each of the input-target state pairs, compute the target state in the computational basis, then convert both to the symmetric Fock basis
    psi_in = np.zeros((K, N), dtype=complex)
    psi_targ = np.zeros((K, N), dtype=complex)
    for k, psi_in_comp in enumerate(psi_in_comps):
        psi_targ_comp = gate @ psi_in_comp

        psi_in[k, comp_indices] = psi_in_comp
        psi_targ[k, comp_indices] = psi_targ_comp

    return (psi_in, psi_targ)


def BSA() -> Tuple[np.ndarray, np.ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based Bell State Analyzer, resolved in the symmetric Fock basis.

    Returns:
        A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states
    """

    # define number of photons, optical modes, and input-target state pairs, which is equivalent to the dimension of the computational basis in this case
    n = 2
    m = 4
    K = 2**n

    # build the N-dimensional symmetric Fock basis and a list of indices within this basis where the corresponding computational basis states lie
    fock_basis = build_symm_basis(n, m)
    N = fock_basis.shape[0]
    comp_indices = comp_indices_from_symm_fock(fock_basis)

    # generate the input states in the computational basis
    c00, c01, c10, c11 = np.eye(K, dtype=complex)
    psi_in_comps = np.zeros((K, K), dtype=complex)
    psi_in_comps[0] = (c00 + c11) / np.sqrt(2)
    psi_in_comps[1] = (c00 - c11) / np.sqrt(2)
    psi_in_comps[2] = (c01 + c10) / np.sqrt(2)
    psi_in_comps[3] = (c01 - c10) / np.sqrt(2)

    # build the gate in the computational basis to easily compute the target states in the computational basis
    gate = logic.BSA()

    # for each of the input-target state pairs, compute the target state in the computational basis, then convert both to the symmetric Fock basis
    psi_in = np.zeros((K, N), dtype=complex)
    psi_targ = np.zeros((K, N), dtype=complex)
    for k, psi_in_comp in enumerate(psi_in_comps):
        psi_targ_comp = gate @ psi_in_comp

        psi_in[k, comp_indices] = psi_in_comp
        psi_targ[k, comp_indices] = psi_targ_comp

    return (psi_in, psi_targ)


def JitterBSA(
    tj: float,
    sigt: float = 1.0,
    Nt: int = 200,
    tlim: float = 10.0,
    num_ancillary_modes: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based Bell State Analyzer, resolved in the asymmetric Fock basis, operating on photons with finite wavepackets and some time jitter between them.

    Args:
        tj: time jitter between the two photons, in dimensionless units normalized to the temporal width of the photon wavepackets
        sigt: temporal width of the photon wavepackets
        Nt: number of points in the time domain to keep track of numerically
        tlim: boundary of the time domain such that it spans from -tlim to +tlim
        num_ancillary_modes: number of ancillary modes for the QPNN-based Bell State Analyzer

    Returns:
        First, a tuple including an $N_t$-length array that provides the time domain for the photon wavefunctions; two $K\\times N\\times N_t\\times N_t$ arrays, the first of which contains $K$ input states resolved in the asymmetric Fock basis,
            with attached wavefunctions, the second of which contains the corresponding target states; and one $K\\times n!$ array containing the $n!$ indices of the asymmetric Fock basis, for each of the $K$ target states in the QPNN training
            set, where the photons are situated in the targeted time bins; second, a $2^n\\times n!$ array where the $n!$ permutations in the asymmetric Fock basis that correspond to each of the $2^n$ computational basis states are catalogued
    """
    assert num_ancillary_modes % 2 == 0, "This function currently only supports an even number of ancillary modes"

    # calculate wavefunctions for photons with time jitter
    t = np.linspace(-tlim, tlim, Nt)
    t1, t2 = np.meshgrid(t, t)
    wf12 = gaussian_t(t1, -0.5 * tj, sigt) * gaussian_t(t2, 0.5 * tj, sigt) / np.sqrt(2)
    wf21 = gaussian_t(t1, 0.5 * tj, sigt) * gaussian_t(t2, -0.5 * tj, sigt) / np.sqrt(2)

    # prepare asymmetric basis and extract the indices where states correspond to the logical dual-rail encoding
    n = 2
    m = 4 + num_ancillary_modes
    K = 4
    basis = build_asymm_basis(n, m)
    N = basis.shape[0]
    ancillary_modes = (
        np.hstack((np.arange(0, num_ancillary_modes // 2), np.arange(num_ancillary_modes + 4, m)), dtype=int) if num_ancillary_modes > 0 else None
    )
    comp_inds = comp_indices_from_asymm_fock(basis, build_symm_basis(n, m), build_symm_mode_basis(n, m), ancillary_modes=ancillary_modes)

    # construct the input states
    psi_in = np.zeros((K, N, Nt, Nt), dtype=complex)

    psi_in[0, comp_inds[0]] = (wf12, wf21)
    psi_in[0, comp_inds[3]] = (wf12, wf21)

    psi_in[1, comp_inds[0]] = (wf12, wf21)
    psi_in[1, comp_inds[3]] = (-wf12, -wf21)

    psi_in[2, comp_inds[1]] = (wf12, wf21)
    psi_in[2, comp_inds[2]] = (wf12, wf21)

    psi_in[3, comp_inds[1]] = (wf12, wf21)
    psi_in[3, comp_inds[2]] = (-wf12, -wf21)

    psi_in /= np.sqrt(2.0)

    # construct the target states
    psi_targ = np.zeros((K, N, Nt, Nt), dtype=complex)
    psi_targ[0, comp_inds[0]] = (wf12, wf21)
    psi_targ[1, comp_inds[2]] = (wf12, wf21)
    psi_targ[2, comp_inds[1]] = (wf12, wf21)
    psi_targ[3, comp_inds[3]] = (wf12, wf21)

    # construct the target bins
    bins_targ = np.copy(comp_inds)
    bins_targ[[1, 2]] = bins_targ[[2, 1]]

    return (t, psi_in, psi_targ, bins_targ), comp_inds


def TreeUnitCell(b: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct training set for the unit cell functionality of a QPNN that powers a tree-type photonic cluster state generation protocol.

    Args:
        b: number of branches in the tree, $b$

    Returns:
        A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states
    """

    # define the number of photons and number of optical modes
    n = b + 1
    m = 2 * n

    # construct the target circuit in the computational basis
    CZs = [logic.CZ(control=0, target=i, n=n) for i in range(1, n)]
    circuit = reduce(np.dot, CZs[::-1])

    # define the relevant single-qubit states that will be used to construct each input in the computational basis
    Id = np.eye(2, dtype=complex)
    ket0 = Id[0].reshape(2, 1)
    ket1 = Id[1].reshape(2, 1)
    ketp = logic.H() @ ket0
    ketm = logic.H() @ ket1

    # build the training set, keeping in mind that the top qubit is |+> for all input states, and the other two qubits are swept over the X and Z computational bases
    comp_basis = logic.build_comp_basis(n - 1)
    K = 2 * (comp_basis.shape[0])
    fock_basis = build_symm_basis(n, m)
    N = fock_basis.shape[0]
    comp_inds = comp_indices_from_symm_fock(fock_basis)
    psi_in = np.zeros((K, N), dtype=complex)
    psi_targ = np.zeros((K, N), dtype=complex)
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

            # convert input and output states to their corresponding represenation in the symmetric Fock basis
            psi_in[i * comp_basis.shape[0] + j, comp_inds] = psi_in_comp.reshape((2**n,))
            psi_targ[i * comp_basis.shape[0] + j, comp_inds] = psi_targ_comp.reshape((2**n,))

    return (psi_in, psi_targ)


def TreeRouting(b: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct training set for the routing functionality of a QPNN that powers a tree-type photonic cluster state generation protocol.

    Args:
        b: number of branches in the tree, $b$

    Returns:
        A tuple including two $K\\times m$ arrays, the first of which contains $K$ input states resolved in the $m$-dimensional symmetric Fock basis, the second of which contains the corresponding target states
    """

    # define the number of photons and number of optical modes
    n = 1
    m = 2 * (b + 1)

    # retrieve the corresponding symmetric Fock basis
    basis = build_symm_basis(n, m)

    # build the training set, keeping in kind that only |+> must be routed straight through the top two modes of the network
    K = 1
    psi_in = np.zeros((K, m), dtype=complex)
    psi_targ = np.zeros((K, m), dtype=complex)

    for i in range(2):
        state = np.zeros(m)
        state[i] = 1
        s = int(np.where((basis == state).all(axis=1))[0].item())
        psi_in[0, s] = 1 / np.sqrt(2)
        psi_targ[0, s] = 1 / np.sqrt(2)

    return (psi_in, psi_targ)


def Tree(b: int) -> tuple:  # noqa: C901
    """Construct the training set for a QPNN that powers a tree-type photonic cluster state generation protocol.

    Args:
        b: number of branches in the tree, $b$

    Returns:
        Tuple of two tuples of $K\\times N$ arrays, the first of which contains the $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states, for each $1 \\leq n \\leq b + 1$
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
            # build the training set, keeping in kind that only |+> must be routed straight through the top two modes of the network
            K = 1
            psi_in_n = np.zeros((K, m), dtype=complex)
            psi_targ_n = np.zeros((K, m), dtype=complex)

            fock_basis = build_symm_basis(_n, m)
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

            # build the training set, keeping in mind that the top qubit is |+> for all input states, and the other two qubits are swept over the X and Z computational bases
            comp_basis = logic.build_comp_basis(_n - 1)
            all_missing_slots = list(combinations(list(range(m - 2, 0, -2)), n - _n)) if n != _n else [(0,)]
            K = 2 * (comp_basis.shape[0]) * len(all_missing_slots)
            fock_basis = build_symm_basis(_n, m)
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
                        # convert input and output states to their corresponding represenation in the symmetric Fock basis
                        comp_inds = comp_indices_from_symm_fock(fock_basis)
                        psi_in_n[i * comp_basis.shape[0] + j, comp_inds] = psi_in_comp.reshape((2**n,))
                        psi_targ_n[i * comp_basis.shape[0] + j, comp_inds] = psi_targ_comp.reshape((2**n,))
                    else:
                        for k, missing_slots in enumerate(all_missing_slots):
                            # convert input and output states to their corresponding represenation in the symmetric Fock basis
                            missing_modes = []
                            for slot in missing_slots:
                                missing_modes.extend([slot, slot + 1])
                            comp_inds = comp_indices_from_symm_fock(fock_basis, ancillary_modes=np.array(missing_modes))
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

    return (tuple(psi_in), tuple(psi_targ))
