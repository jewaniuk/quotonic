"""
The `quotonic.training_sets` module includes functions that are used to prepare training sets for quantum photonic
neural network (QPNN) training simulations. Training sets can vary depending on the specific QPNN model used and the
considered application. That being said, these functions typically return a set of input-target state pairs, where the
QPNN should be trained to map each input state to its corresponding target state (see [qpnn](qpnn.md) and
[trainer](trainer.md) for more details on network models and training, respectively). Depending on the specific task,
these states may be resolved in a first-quantized basis, the second-quantized Fock basis, or the computational basis of
the photonic qubits. As a result, the [fock](fock.md) and [logic](logic.md) modules are particularly useful in these
functions.

If you decide to use `quotonic` to perform research on QPNNs, feel free to develop a training set function to go with
your QPNN model. Also, we'd be happy to add it if it fits the format appropriately, so please reach out!
"""

from functools import reduce
from itertools import combinations

import numpy as np

import quotonic.logic as logic
from quotonic.fock import build_secq_basis
from quotonic.types import np_ndarray
from quotonic.utils import comp_indices_from_secq


def CNOT() -> tuple[np_ndarray, np_ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based CNOT gate, resolved in the second-quantized Fock basis.

    See [logic](logic.md) for more details on CNOT gates from a logical standpoint. The truth table is as follows,
    where a logical 0 (1) state is defined as $\\left|0\\right\\rangle_\\mathrm{log} \\equiv \\left|10\\right\\rangle$
    ($\\left|1\\right\\rangle_\\mathrm{log} \\equiv \\left|01\\right\\rangle$) for the dual-rail encoding considered
    here.

    <table>
      <thead>
        <tr><th>$\\left|\\mathrm{in}\\right\\rangle$</th><th>$\\left|\\mathrm{targ}\\right\\rangle$</th></tr>
      </thead>
      <tbody>
        <tr><td>$\\left|1010\\right\\rangle$</td><td>$\\left|1010\\right\\rangle$</td></tr>
        <tr><td>$\\left|1001\\right\\rangle$</td><td>$\\left|1001\\right\\rangle$</td></tr>
        <tr><td>$\\left|0110\\right\\rangle$</td><td>$\\left|0101\\right\\rangle$</td></tr>
        <tr><td>$\\left|0101\\right\\rangle$</td><td>$\\left|0110\\right\\rangle$</td></tr>
      </tbody>
    </table>

    Returns:
        psi_in: $K\\times N$ array containing the $K$ input states resolved in the $N$-dimensional second quantization
            Fock basis
        psi_targ: $K\\times N$ array containing the $K$ target states resolved in the $N$-dimensional second
            quantization Fock basis
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


def CZ() -> tuple[np_ndarray, np_ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based CZ gate, resolved in the second-quantized Fock basis.

    See [logic](logic.md) for more details on CZ gates from a logical standpoint. The truth table is as follows,
    where a logical 0 (1) state is defined as $\\left|0\\right\\rangle_\\mathrm{log} \\equiv \\left|10\\right\\rangle$
    ($\\left|1\\right\\rangle_\\mathrm{log} \\equiv \\left|01\\right\\rangle$) for the dual-rail encoding considered
    here.

    <table>
      <thead>
        <tr><th>$\\left|\\mathrm{in}\\right\\rangle$</th><th>$\\left|\\mathrm{targ}\\right\\rangle$</th></tr>
      </thead>
      <tbody>
        <tr><td>$\\left|1010\\right\\rangle$</td><td>$+\\left|1010\\right\\rangle$</td></tr>
        <tr><td>$\\left|1001\\right\\rangle$</td><td>$+\\left|1001\\right\\rangle$</td></tr>
        <tr><td>$\\left|0110\\right\\rangle$</td><td>$+\\left|0110\\right\\rangle$</td></tr>
        <tr><td>$\\left|0101\\right\\rangle$</td><td>$-\\left|0101\\right\\rangle$</td></tr>
      </tbody>
    </table>

    Returns:
        psi_in: $K\\times N$ array containing the $K$ input states resolved in the $N$-dimensional second quantization
            Fock basis
        psi_targ: $K\\times N$ array containing the $K$ target states resolved in the $N$-dimensional second
            quantization Fock basis
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


def BSA() -> tuple[np_ndarray, np_ndarray]:
    """Construct training set for a dual-rail encoded QPNN-based Bell State Analyzer, resolved in the
        second-quantized Fock basis.

    See [logic](logic.md) for more details on BSA gates from a logical standpoint. The truth table is as follows,
    where a logical 0 (1) state is defined as $\\left|0\\right\\rangle_\\mathrm{log} \\equiv \\left|10\\right\\rangle$
    ($\\left|1\\right\\rangle_\\mathrm{log} \\equiv \\left|01\\right\\rangle$) for the dual-rail encoding considered
    here.

    <table>
      <thead>
        <tr><th>$\\left|\\mathrm{in}\\right\\rangle$</th><th>$\\left|\\mathrm{targ}\\right\\rangle$</th></tr>
      </thead>
      <tbody>
        <tr><td>$\\left|\\Phi^+\\right\\rangle \\equiv \\frac{1}{\\sqrt{2}}\\left(\\left|1010\\right\\rangle +
        \\left|0101\\right\\rangle\\right)$</td><td>$\\left|1010\\right\\rangle$</td></tr>
        <tr><td>$\\left|\\Phi^-\\right\\rangle \\equiv \\frac{1}{\\sqrt{2}}\\left(\\left|1010\\right\\rangle -
        \\left|0101\\right\\rangle\\right)$</td><td>$\\left|0110\\right\\rangle$</td></tr>
        <tr><td>$\\left|\\Psi^+\\right\\rangle \\equiv \\frac{1}{\\sqrt{2}}\\left(\\left|1001\\right\\rangle +
        \\left|0110\\right\\rangle\\right)$</td><td>$\\left|1001\\right\\rangle$</td></tr>
        <tr><td>$\\left|\\Psi^-\\right\\rangle \\equiv \\frac{1}{\\sqrt{2}}\\left(\\left|1001\\right\\rangle -
        \\left|0110\\right\\rangle\\right)$</td><td>$\\left|0101\\right\\rangle$</td></tr>
      </tbody>
    </table>

    Returns:
        psi_in: $K\\times N$ array containing the $K$ input states resolved in the $N$-dimensional second quantization
            Fock basis
        psi_targ: $K\\times N$ array containing the $K$ target states resolved in the $N$-dimensional second
            quantization Fock basis
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


def Tree(b: int) -> tuple[tuple, tuple, tuple]:  # noqa: C901
    """Construct the training set for a QPNN that powers a tree-type photonic cluster state generation protocol.

    ADD DOCUMENTATION HERE

    Args:
        b: maximum number of branches in the tree, $\\max\\left\\{\\vec{b}\\}$

    Returns:
        Tuple of three tuples, the first two of which are the input and target states resolved in the computational
            basis for each $1 \\leq n \\leq b + 1$, the last of which contains the computational basis indices for each
            unit cell operation that exists for each $n$
    """

    # define the number of photons and number of optical modes
    assert b >= 2, "the smallest useful tree has at least one section with 2 or more branches"
    n = b + 1
    m = 2 * n

    def build_tset_for_n(_n: int) -> tuple:
        # define the relevant single-qubit states that will be used to construct each input in the computational basis
        Id = np.eye(2, dtype=complex)
        ket0 = Id[0].reshape(2, 1)
        ket1 = Id[1].reshape(2, 1)
        ketp = logic.H() @ ket0
        ketm = logic.H() @ ket1

        if _n == 1:
            # if just one photon, it is |+> and should be routed through unchanged
            psi_in = np.copy(ketp)
            psi_targ = np.copy(ketp)
            return psi_in.reshape(1, 2), psi_targ.reshape(1, 2)

        else:
            # if more than one photon, the circuit should apply CZ gates between the control (0) and all targets
            CZs = [logic.CZ(control=0, target=i, n=_n) for i in range(1, _n)]
            circuit = reduce(np.dot, CZs[::-1])

            # build the training set, keeping in mind that the top qubit is |+> for all input states,
            # and the other two qubits are swept over the X and Z computational bases
            comp_basis = logic.build_comp_basis(_n - 1)
            K = 2 * (comp_basis.shape[0])
            psi_in = np.zeros((K, 2**_n), dtype=complex)
            psi_targ = np.zeros((K, 2**_n), dtype=complex)
            for i, basis in enumerate(["X", "Z"]):
                for j, comp_state in enumerate(comp_basis):
                    # construct input state in the computational basis, then get output by applying the circuit
                    _psi_in = np.copy(ketp)
                    for qubit in comp_state:
                        if basis == "X":
                            _psi_in = np.kron(_psi_in, ketp) if qubit == 0 else np.kron(_psi_in, ketm)
                        elif basis == "Z":
                            _psi_in = np.kron(_psi_in, ket0) if qubit == 0 else np.kron(_psi_in, ket1)
                    psi_in[i * comp_basis.shape[0] + j] = _psi_in.reshape((2**_n,))
                    psi_targ[i * comp_basis.shape[0] + j] = (circuit @ _psi_in).reshape((2**_n,))
            return psi_in, psi_targ

    def build_comp_indices_for_n(_n: int) -> np.ndarray:
        # compute all combinations of empty qubit slots (2 modes each) for the given number of qubits/photons
        empty_slot_combos = list(combinations(range(1, n), n - _n))

        # for each combination, prepare a corresponding list of empty modes and use this to prepare the indices of the
        # computational basis within the larger N-dimensional second quantization Fock basis
        basis = build_secq_basis(_n, m)
        comp_indices = np.zeros((len(empty_slot_combos), 2**_n), dtype=int)
        for i, empty_slots in enumerate(empty_slot_combos):
            if len(empty_slots) == 0:
                empty_modes = None
            else:
                empty_modes = np.zeros(2 * len(empty_slots), dtype=int)
                for j, slot in enumerate(empty_slots):
                    empty_modes[2 * j] = 2 * slot
                    empty_modes[2 * j + 1] = 2 * slot + 1
            comp_indices[i] = comp_indices_from_secq(basis, ancillary_modes=empty_modes)
        return comp_indices

    # construct the full training set for each number of photons 1 <= n <= b + 1
    psi_in = []
    psi_targ = []
    comp_indices = []
    for _n in range(1, n + 1):
        psi_in_n, psi_targ_n = build_tset_for_n(_n)
        psi_in.append(psi_in_n)
        psi_targ.append(psi_targ_n)
        comp_indices.append(build_comp_indices_for_n(_n))
    return tuple(psi_in), tuple(psi_targ), tuple(comp_indices)
