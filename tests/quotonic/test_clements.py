import numpy as np
from jax import config

from quotonic.clements import Mesh
from quotonic.utils import genHaarUnitary

config.update("jax_enable_x64", True)


def test_Mesh():
    mesh = Mesh(
        4,
        ell_mzi=np.zeros((4, 4), dtype=float),
        ell_ps=np.zeros(4, dtype=float),
        t_dc=0.5 * np.ones((2, 6), dtype=float),
    )
    assert mesh.m == 4


def test_mzi():
    mesh = Mesh(2)

    result = np.array([[0, 1j], [1j, 0]], dtype=complex)
    assert np.allclose(mesh.mzi(0.0, 0.0), result)

    result = np.conjugate(result).T
    assert np.allclose(mesh.mzi_inv(0.0, 0.0), result)


def test_haar_to_decode_to_encode():
    for m in range(3, 7):
        mesh = Mesh(m)
        for _ in range(100):
            U = genHaarUnitary(m)
            phases = mesh.decode(U)
            assert np.allclose(mesh.encode(*phases), U, atol=1e-4)


def test_encode_with_imperfections():
    m = 4
    ell_mzi = 1.0 - np.array(
        [[0.98, 0.99, 0.95, 0.92], [0.96, 0.94, 0.99, 0.99], [0.97, 0.97, 0.93, 0.98], [0.99, 0.95, 0.99, 0.93]],
        dtype=float,
    )
    ell_ps = 1.0 - np.array([0.995, 0.992, 0.998, 0.997], dtype=float)
    t_dc = np.array([[0.51, 0.52, 0.49, 0.50, 0.48, 0.50], [0.50, 0.47, 0.51, 0.50, 0.49, 0.49]], dtype=float)
    mesh = Mesh(4, ell_mzi=ell_mzi, ell_ps=ell_ps, t_dc=t_dc)
    phases = mesh.decode(np.eye(m, dtype=complex))
    U = np.array(
        [
            [
                9.18493683e-01 + 7.04239114e-17j,
                -3.84103882e-18 + 9.18585551e-03j,
                -2.98357088e-33 - 1.08914148e-18j,
                5.45280136e-20 - 7.59577280e-35j,
            ],
            [
                -9.80081095e-19 + 9.36306739e-03j,
                9.36213099e-01 - 1.16477412e-18j,
                1.27419229e-17 - 2.80584551e-02j,
                1.49759458e-03 + 2.59930090e-18j,
            ],
            [
                2.77521453e-04 - 1.22760399e-18j,
                -8.78209819e-18 - 2.77493699e-02j,
                9.23021230e-01 + 1.90642708e-16j,
                -1.86207590e-17 + 5.54755662e-02j,
            ],
            [
                -4.26699993e-21 + 1.85001559e-06j,
                1.84983057e-04 + 1.58200575e-18j,
                7.56486737e-18 + 5.57417050e-02j,
                9.27476932e-01 - 5.65871102e-18j,
            ],
        ],
        dtype=complex,
    )
    assert np.allclose(mesh.encode(*phases), U, atol=1e-4)
