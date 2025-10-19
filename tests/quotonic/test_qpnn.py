import jax.numpy as jnp
import numpy as np

from quotonic.qpnn import IdealQPNN, ImperfectQPNN, TreeQPNN
from quotonic.training_sets import BSA, CNOT, Tree
from quotonic.utils import genHaarUnitary


def test_IdealQPNN():
    n = 2
    m = 4
    L = 2
    training_set = CNOT()
    qpnn = IdealQPNN(n, m, L, training_set=training_set)

    phi, theta, delta = (
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m), dtype=float),
    )
    for i in range(L):
        U = genHaarUnitary(m)
        phi[i], theta[i], delta[i] = qpnn.mesh.decode(U)
    phases = (jnp.asarray(phi), jnp.asarray(theta), jnp.asarray(delta))

    cost = 1 - float(qpnn.calc_fidelity(*phases))
    assert (cost > 0) and (cost < 1)


def test_ImperfectQPNN():
    n = 2
    m = 4
    L = 3
    ell_mzi = (0.2130, 0.0124)  # (0.2130 +/- 0.0124) dB loss per MZI
    ell_ps = (0.106, 0.006)  # (0.106 +/- 0.006) dB loss per phase shifter
    t_dc = (0.50, 0.05)  # (50 +/- 5) % T:R directional coupler splitting ratio
    training_set = BSA()
    qpnn = ImperfectQPNN(n, m, L, ell_mzi=ell_mzi, ell_ps=ell_ps, t_dc=t_dc, training_set=training_set)

    phi, theta, delta = (
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m), dtype=float),
    )
    for i in range(L):
        U = genHaarUnitary(m)
        phi[i], theta[i], delta[i] = qpnn.meshes[0].decode(U)
    phases = (jnp.asarray(phi), jnp.asarray(theta), jnp.asarray(delta))

    cost = float(qpnn.calc_unc_fidelity(*phases))
    assert (cost > 0) and (cost < 1)

    Func, Fcon, rate = qpnn.calc_performance_measures(*phases)
    assert (Func > 0) and (Func < 1)
    assert (Fcon > 0) and (Fcon < 1)
    assert (rate > 0) and (rate < 1)


def test_TreeQPNN():
    b = 2
    L = 2
    training_set = Tree(b)
    qpnn = TreeQPNN(b, L, training_set=training_set)

    m = 2 * (b + 1)
    phi, theta, delta = (
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m), dtype=float),
    )
    for i in range(L):
        U = genHaarUnitary(m)
        phi[i], theta[i], delta[i] = qpnn.meshes[0].decode(U)
    phases = (jnp.asarray(phi), jnp.asarray(theta), jnp.asarray(delta))

    cost = float(qpnn.calc_cost(*phases))
    assert (cost > 0) and (cost < 1)

    F, succ_rate, logi_rate = qpnn.calc_overall_performance_measures(*phases)
    assert (F > 0) and (F < 1)
    assert (succ_rate > 0) and (succ_rate < 1)
    assert (logi_rate > 0) and (logi_rate < 1)

    F, succ_rate, logi_rate = qpnn.calc_unit_cell_performance_measures(*phases)
    for i in range(qpnn.n):
        assert not np.isnan(F[i]).any()
        assert not np.isnan(succ_rate[i]).any()
        assert not np.isnan(logi_rate[i]).any()
