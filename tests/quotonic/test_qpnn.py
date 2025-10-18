import jax.numpy as jnp
import numpy as np

from quotonic.qpnn import TreeQPNN
from quotonic.training_sets import Tree
from quotonic.utils import genHaarUnitary


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
