import numpy as np

import quotonic.fock as fock


def test_calc_symm_dim():
    n = 2
    m = 4
    N = 10
    assert fock.calc_symm_dim(n, m) == N

    n = 4
    m = 8
    N = 330
    assert fock.calc_symm_dim(n, m) == N

    n = 1
    m = 5
    N = 5
    assert fock.calc_symm_dim(n, m) == N

    n = 3
    m = 5
    N = 35
    assert fock.calc_symm_dim(n, m) == N


def test_build_symm_basis():
    n = 2
    m = 4
    result = np.array(
        [[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 2, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 2, 0], [0, 0, 1, 1], [0, 0, 0, 2]]
    )
    assert np.allclose(fock.build_symm_basis(n, m), result)

    n = 3
    m = 3
    result = np.array([[3, 0, 0], [2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2], [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3]])
    assert np.allclose(fock.build_symm_basis(n, m), result)


def test_calc_asymm_dim():
    n = 2
    m = 4
    N = 16
    assert fock.calc_asymm_dim(n, m) == N

    n = 4
    m = 8
    N = 4096
    assert fock.calc_asymm_dim(n, m) == N

    n = 1
    m = 5
    N = 5
    assert fock.calc_asymm_dim(n, m) == N

    n = 3
    m = 5
    N = 125
    assert fock.calc_asymm_dim(n, m) == N


def test_build_asymm_basis():
    n = 2
    m = 4
    result = np.array(
        [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]]
    )
    assert np.allclose(fock.build_asymm_basis(n, m), result)

    n = 3
    m = 3
    result = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
        ]
    )
    assert np.allclose(fock.build_asymm_basis(n, m), result)
