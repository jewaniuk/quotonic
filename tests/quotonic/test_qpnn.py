import jax.numpy as jnp
import numpy as np

from quotonic.qpnn import JitterQPNN
from quotonic.training_sets import JitterBSA
from quotonic.utils import genHaarUnitary


def test_JitterQPNN():
    n = 2
    m = 4
    L = 2
    training_set, comp_indices = JitterBSA(0.0)
    qpnn = JitterQPNN(n, m, L, training_set=training_set, comp_indices=comp_indices)

    phi, theta, delta = (
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m * (m - 1) // 2), dtype=float),
        np.zeros((L, m), dtype=float),
    )
    for i in range(L):
        U = genHaarUnitary(m)
        phi[i], theta[i], delta[i] = qpnn.mesh.decode(U)
    phases = (jnp.asarray(phi), jnp.asarray(theta), jnp.asarray(delta))

    Fw, Fb = qpnn.calc_unc_fidelities(*phases)
    Fw = float(Fw)
    Fb = float(Fb)
    assert (Fw > 0) and (Fw < 1)
    assert (Fb > 0) and (Fb < 1)

    Fuw, Fub, Fcw, Fcb, rate = qpnn.calc_performance_measures(*phases)
    assert (Fuw > 0) and (Fuw < 1)
    assert (Fub > 0) and (Fub < 1)
    assert (Fcw > 0) and (Fcw < 1)
    assert (Fcb > 0) and (Fcb < 1)
    assert (rate > 0) and (rate < 1)
