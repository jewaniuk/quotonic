import numpy as np

import quotonic.logic as logic


def test_build_comp_basis():
    n = 1
    result = np.array([[0], [1]])
    assert np.allclose(logic.build_comp_basis(n), result)

    n = 2
    result = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.allclose(logic.build_comp_basis(n), result)

    n = 3
    result = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    assert np.allclose(logic.build_comp_basis(n), result)


def test_H():
    assert np.allclose(logic.H() @ np.array([1, 0], dtype=complex), np.array([0.70710678 + 0.0j, 0.70710678 + 0.0j], dtype=complex))


def test_CNOT():
    assert np.allclose(
        logic.CNOT() @ np.array([0, 0, 0, 1], dtype=complex), np.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    )
    assert np.allclose(
        logic.CNOT(control=1, target=2, n=3) @ np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=complex), np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=complex)
    )


def test_CZ():
    assert np.allclose(logic.CZ() @ np.array([0, 0, 0, 1], dtype=complex), np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j], dtype=complex))
    assert np.allclose(
        logic.CZ(control=1, target=2, n=3) @ np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=complex), np.array([0, 0, 0, -1, 0, 0, 0, 0], dtype=complex)
    )


def test_BSA():
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0)
    assert np.allclose(logic.BSA() @ psi, np.array([1, 0, 0, 0], dtype=complex))
