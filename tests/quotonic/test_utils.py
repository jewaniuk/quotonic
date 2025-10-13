import numpy as np

import quotonic.fock as fock
import quotonic.utils as utils


def test_comp_to_symm_fock():
    result = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    assert np.allclose(utils.comp_to_symm_fock(np.array([1, 0, 0, 1])), result)


def test_symm_fock_to_comp():
    result = np.array([1, 0, 0, 1])
    assert np.allclose(utils.symm_fock_to_comp(np.array([0, 1, 1, 0, 1, 0, 0, 1])), result)


def test_comp_indices_from_symm_fock():
    n = 4
    m = 8
    basis = fock.build_symm_basis(n, m)
    result = np.array([77, 78, 80, 81, 92, 93, 95, 96, 161, 162, 164, 165, 176, 177, 179, 180])
    assert np.allclose(utils.comp_indices_from_symm_fock(basis), result)


def test_comp_indices_from_asymm_fock():
    n = 2
    m = 4
    asymm_basis = fock.build_asymm_basis(n, m)
    symm_basis = fock.build_symm_basis(n, m)
    symm_mode_basis = fock.build_symm_mode_basis(n, m)
    result = np.array([[2, 8], [3, 12], [6, 9], [7, 13]])
    assert np.allclose(utils.comp_indices_from_asymm_fock(asymm_basis, symm_basis, symm_mode_basis), result)

    n = 3
    m = 6
    asymm_basis = fock.build_asymm_basis(n, m)
    symm_basis = fock.build_symm_basis(n, m)
    symm_mode_basis = fock.build_symm_mode_basis(n, m)
    result = np.array(
        [
            [16, 26, 76, 96, 146, 156],
            [17, 32, 77, 102, 182, 192],
            [22, 27, 112, 132, 147, 162],
            [23, 33, 113, 138, 183, 198],
            [52, 62, 82, 97, 152, 157],
            [53, 68, 83, 103, 188, 193],
            [58, 63, 118, 133, 153, 163],
            [59, 69, 119, 139, 189, 199],
        ]
    )
    assert np.allclose(utils.comp_indices_from_asymm_fock(asymm_basis, symm_basis, symm_mode_basis), result)

    n = 2
    m = 6
    asymm_basis = fock.build_asymm_basis(n, m)
    symm_basis = fock.build_symm_basis(n, m)
    symm_mode_basis = fock.build_symm_mode_basis(n, m)
    ancillary_modes = np.array([0, 5])
    result = np.array([[9, 19], [10, 25], [15, 20], [16, 26]])
    assert np.allclose(utils.comp_indices_from_asymm_fock(asymm_basis, symm_basis, symm_mode_basis, ancillary_modes=ancillary_modes), result)
