import numpy as np
from jax import config

import quotonic.aa as aa
from quotonic.perm import Permanent, calc_perm

config.update("jax_enable_x64", True)


def test_SymmetricTransformer():
    # 50:50 beamsplitter
    U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    m = 2
    n = 2
    result = np.array([[0.5, 2**-0.5, 0.5], [2**-0.5, 0, -(2**-0.5)], [0.5, -(2**-0.5), 0.5]], dtype=complex)
    assert np.allclose(aa.SecqTransformer(n, m).transform(U), result, atol=1e-4)
    n = 3
    result = np.array(
        [
            [0.35355339 + 0.0j, 0.61237244 + 0.0j, 0.61237244 + 0.0j, 0.35355339 + 0.0j],
            [0.61237244 + 0.0j, 0.35355339 + 0.0j, -0.35355339 + 0.0j, -0.61237244 + 0.0j],
            [0.61237244 + 0.0j, -0.35355339 + 0.0j, -0.35355339 + 0.0j, 0.61237244 + 0.0j],
            [0.35355339 + 0.0j, -0.61237244 + 0.0j, 0.61237244 + 0.0j, -0.35355339 + 0.0j],
        ],
        dtype=complex,
    )
    assert np.allclose(aa.SecqTransformer(n, m).transform(U), result, atol=1e-4)

    # random n = 4, m = 2
    n = 4
    m = 2
    U = np.array(
        [
            [1.11895135e-01 + 0.77749458j, -1.15952494e-05 - 0.61885512j],
            [-1.14623614e-01 + 0.60814726j, -2.53951746e-01 + 0.7433215j],
        ],
        dtype=complex,
    )
    result = np.array(
        [
            [
                0.32016261 - 2.06003033e-01j,
                -0.54556545 + 2.49433725e-01j,
                0.5553335 - 1.63247876e-01j,
                -0.36854525 + 5.30612842e-02j,
                0.14667495 - 1.09927678e-05j,
            ],
            [
                0.58232822 - 1.44057949e-01j,
                -0.32658202 + 3.26343883e-02j,
                -0.27841327 - 1.20686725e-02j,
                0.55252267 + 1.04107329e-01j,
                -0.35235626 - 1.20358484e-01j,
            ],
            [
                0.57665587 + 5.01314794e-02j,
                0.27135943 + 6.34323122e-02j,
                -0.38917709 - 1.52090995e-01j,
                -0.24246418 - 1.37370735e-01j,
                0.45784395 + 3.54152620e-01j,
            ],
            [
                0.34059697 + 1.50448688e-01j,
                0.47670381 + 2.98115936e-01j,
                0.21282188 + 1.79906767e-01j,
                -0.21791598 - 2.45425028e-01j,
                -0.33034354 - 5.00731456e-01j,
            ],
            [
                0.10780127 + 9.94606861e-02j,
                0.2349093 + 2.88892236e-01j,
                0.29749025 + 4.96532689e-01j,
                0.23187245 + 5.53257637e-01j,
                0.09564539 + 3.68501429e-01j,
            ],
        ],
        dtype=complex,
    )
    assert np.allclose(aa.SecqTransformer(n, m, algo="bbfg").transform(U), result, atol=1e-4)
    assert np.allclose(aa.SecqTransformer(n, m, algo="ryser").transform(U), result, atol=1e-4)


def test_extra_perm():
    U = np.array(
        [
            [0.05030219 - 0.05710856j, 0.38623717 - 0.37529863j, 0.03448542 + 0.12082856j, 0.57700862 + 0.59619826j],
            [0.9109055 + 0.26600087j, -0.25189162 - 0.13916038j, -0.08436918 - 0.03166838j, -0.0461551 + 0.08017707j],
            [0.19494714 - 0.11139955j, 0.38728436 - 0.12388739j, 0.81554126 - 0.23149101j, -0.18106198 - 0.18102326j],
            [-0.17608286 + 0.11083131j, -0.65066956 + 0.1960833j, 0.50498278 - 0.04885628j, 0.07132948 + 0.48208846j],
        ],
        dtype=complex,
    )
    result = -0.1762361228466034 + 0.014973999932408333j
    assert np.allclose(complex(calc_perm(U, algo="ryser")), result, atol=1e-4)

    U = np.ones((1, 1), dtype=float)
    result = 1.0
    assert np.allclose(Permanent(1).perm(U), result)
