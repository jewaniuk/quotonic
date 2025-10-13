import numpy as np

import quotonic.fock as fock
import quotonic.utils as utils


def test_comp_to_secq():
    result = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    assert np.allclose(utils.comp_to_secq(np.array([1, 0, 0, 1])), result)


def test_symm_fock_to_comp():
    result = np.array([1, 0, 0, 1])
    assert np.allclose(utils.secq_to_comp(np.array([0, 1, 1, 0, 1, 0, 0, 1])), result)


def test_comp_indices_from_symm_fock():
    n = 4
    m = 8
    basis = fock.build_secq_basis(n, m)
    result = np.array([77, 78, 80, 81, 92, 93, 95, 96, 161, 162, 164, 165, 176, 177, 179, 180])
    assert np.allclose(utils.comp_indices_from_secq(basis), result)
