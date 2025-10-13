"""
The `quotonic.aa` module includes ...
"""

from functools import partial, reduce
from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import jit, vmap
from jax.scipy.special import factorial
from jax.typing import ArrayLike

from quotonic.fock import build_symm_basis, build_symm_mode_basis
from quotonic.perm import EmptyPermanent, Permanent, calc_perm


def gen_basis_combos(basis: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate combinations of basis states corresponding to each element of a matrix resolved in the basis.

    Args:
        basis: $N\\times x$ array that catalogs all states in the $N$-dimensional basis, where each state has $x$ labels

    Returns:
        Tuple of $N^2\\times x$ arrays, the first of which repeats each state $N$ times vertically before moving to the next state, the second of which repeats the entire basis in the order given $N$ times
    """
    N = jnp.shape(basis)[0]
    return jnp.repeat(basis, N, axis=0), jnp.vstack([basis] * N)


@vmap
def vectorial_factorial(x: Union[int, float]) -> Union[int, float]:
    """Compute the factorial on the input vectorially.

    Args:
        x: integer to compute the factorial of

    Returns:
        Factorial of the input
    """
    return factorial(x)  # type: ignore


@vmap
def calc_norm(S: jnp.ndarray, T: jnp.ndarray) -> float:
    """Calculate the normalization factor for an element of a symmetric multi-photon unitary.

    Args:
        S: state $\\left| S\\right\\rangle$ corresponding to the row of the multi-photon unitary, length $m$
        T: state $\\left| T\\right\\rangle$ corresponding to the column of the multi-photon unitary, length $m$

    Returns:
        Normalization factor for symmetric multi-photon unitary element $\\left\\langle S\\right|\\boldsymbol{\\Phi}(\\mathbf{U})\\left| T\\right\\rangle$
    """
    return 1.0 / jnp.sqrt(jnp.prod(vectorial_factorial(jnp.concatenate((S, T)))))  # type: ignore


@partial(jit, static_argnums=(1,))
def _symmetric_transform(
    U: jnp.ndarray,
    N: int,
    modeBasis_combos: Tuple[jnp.ndarray, jnp.ndarray],
    norms: jnp.ndarray,
) -> jnp.ndarray:
    """Perform a symmetric multi-photon unitary transformation on a single-photon unitary $\\mathbf{U}$, given prepared combinations of basis states.

    Args:
        U: $m\\times m$ single-photon unitary, $\\mathbf{U}$, to transform
        N: dimension of the multi-photon unitary as resolved in the symmetric Fock basis of $n$ photons and $m$ modes
        modeBasis_combos: tuple of $N^2\\times n$ arrays, the first of which repeats each mode basis state $N$ times vertically before moving to the next state, the second of which repeats the entire mode basis in the order given $N$ times
        norms: normalization factors for each element of the multi-photon unitary, flattened to a $N^2\\times 1$ array

    Returns:
        $N\times N$ multi-photon unitary, $\\boldsymbol{\\Phi(\\mathbf{U})}$, in the $N$-dimensional symmetric Fock basis
    """

    # TODO: Try a similar forcible parallel implementation as what is written below for shard_map rather than pmap
    # To use the implementation below, add n: int and num_cores: int = 0 to the arguments, make each a static arg, and change how this function is called elsewhere
    # also need to import device_count and pmap from jax
    # also need to make sure whatever script runs this code specifies the number of cpu cores in the environment variable -> os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    # @vmap
    # def calc_perm_of_UST(S: jnp.ndarray, T: jnp.ndarray) -> complex:
    #     U_ST = U[:, T][S, :]
    #     return calc_perm(U_ST)

    # # if number of cores is not specified, check how many cores are available
    # if num_cores == 0:
    #     num_cores = device_count()

    # N_sq = N**2
    # if num_cores > 1:
    #     N_adj = N_sq - (N_sq % num_cores)
    #     modeBasis_combos_batched = (
    #         modeBasis_combos[0][0:N_adj].reshape((num_cores, N_sq // num_cores, n)),
    #         modeBasis_combos[1][0:N_adj].reshape((num_cores, N_sq // num_cores, n)),
    #     )
    #     modeBasis_combos_leftover = (modeBasis_combos[0][N_adj::], modeBasis_combos[1][N_adj::])

    #     perms_batched = pmap(calc_perm_of_UST)(modeBasis_combos_batched[0], modeBasis_combos_batched[1]).reshape(
    #         (N_adj,)
    #     )
    #     perms_leftover = calc_perm_of_UST(modeBasis_combos_leftover[0], modeBasis_combos_leftover[1])
    #     perms = jnp.concatenate((perms_batched, perms_leftover))
    # else:
    #     perms = calc_perm_of_UST(modeBasis_combos[0], modeBasis_combos[1])

    # vectorially build all U_{S,T} matrices required to compute the multi-photon transform
    U_STs = vmap(lambda S, T: U[:, T][S, :])(*modeBasis_combos)

    # vectorially compute the permanents of all U_{S,T} matrices
    perms = vmap(calc_perm)(U_STs)

    PhiU: jnp.ndarray = (perms * norms).reshape(N, N)
    return PhiU


def symmetric_transform(U: ArrayLike, n: int) -> np.ndarray:
    """Perform a symmetric multi-photon unitary transformation on a single-photon unitary $\\mathbf{U}$.

    Args:
        U: $m\\times m$ single-photon unitary, $\\mathbf{U}$, to transform
        n: number of photons, $n$

    Returns:
        $N\times N$ multi-photon unitary, $\\boldsymbol{\\Phi(\\mathbf{U})}$, in the $N$-dimensional symmetric Fock basis
    """
    # ensure that U is a jax array
    U = jnp.asarray(U)

    # extract the number of optical modes from the dimension of the square matrix U
    Ushape = jnp.shape(U)
    assert Ushape[0] == Ushape[1], "Matrix must be square"
    assert Ushape[0] > 0, "Matrix must have elements"
    m = Ushape[0]

    # construct the symmetric Fock basis in both representations
    modeBasis = jnp.asarray(build_symm_mode_basis(n, m))
    basis = jnp.asarray(build_symm_basis(n, m))
    N = basis.shape[0]

    # stack and repeat the bases to generate combinations that correspond to each multi-photon unitary matrix element
    modeBasis_combos = gen_basis_combos(modeBasis)
    basis_combos = gen_basis_combos(basis)

    # vectorially compute the normalization factors for each element of the multi-photon unitary
    norms = calc_norm(*basis_combos)

    PhiU: npt.NDArray[np.complex128] = np.asarray(_symmetric_transform(U, N, modeBasis_combos, norms))
    return PhiU


class SymmetricTransformer:
    """Wrapper class for performing symmetric multi-photon unitary transformations while the required overhead is stored in memory.

    Attributes:
        n (int): number of photons, $n$
        N (int): dimension of the symmetric Fock basis for $n$ photons and $m$ optical modes
        modeBasis_combos (Tuple[jnp.ndarray, jnp.ndarray]): tuple of $N^2\\times n$ arrays, the first of which repeats each mode basis state $N$ times vertically before moving to the next state, the second of which repeats the entire
            mode basis in the order given $N$ times; defaults to a tuple of empty arrays if $n = 1$
        norms (jnp.ndarray): normalization factors for each element of the multi-photon unitary, flattened to a $N^2\\times 1$ array, defaults to an empty array if $n = 1$
        calculator (Union[Permanent, EmptyPermanent]): instance of a wrapped permanent calculator that computes overhead for the selected algorithms ahead of time, defaults to EmptyPermanent if $n = 1$
    """

    def __init__(self, n: int, m: int, algo: str = "bbfg") -> None:
        """Initialization of a Symmetric Transformer.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            algo: algorithm to compute permanents with if $n > 3$, either "bbfg" or "ryser"
        """

        # check the validity of the provided arguments
        assert n > 0, "There must be at least one photon for this class to be relevant"
        assert m > 1, "There must be at least two optical modes for this class to be relevant"

        self.n = n
        if n > 1:
            # construct the symmetric Fock basis in both representations
            modeBasis = jnp.asarray(build_symm_mode_basis(n, m))
            basis = jnp.asarray(build_symm_basis(n, m))
            self.N = basis.shape[0]

            # stack and repeat the bases to generate combinations that correspond to each multi-photon unitary matrix element
            self.modeBasis_combos = gen_basis_combos(modeBasis)
            basis_combos = gen_basis_combos(basis)

            # vectorially compute the normalization factors for each element of the multi-photon unitary
            self.norms: jnp.ndarray = calc_norm(*basis_combos)  # type: ignore

            # instantiate a permanent calculator
            self.calculator: Union[Permanent, EmptyPermanent] = Permanent(n, algo=algo)
        else:
            self.N = m
            self.modeBasis_combos = (jnp.array(()), jnp.array(()))
            self.norms = jnp.array(())
            self.calculator = EmptyPermanent()

    @partial(jit, static_argnums=(0,))
    def transform(self, U: jnp.ndarray) -> jnp.ndarray:
        """Perform a symmetric multi-photon unitary transformation on a single-photon unitary $\\mathbf{U}$.

        Args:
            U: $m\\times m$ single-photon unitary, $\\mathbf{U}$, to transform

        Returns:
            $N\times N$ multi-photon unitary, $\\boldsymbol{\\Phi(\\mathbf{U})}$, in the $N$-dimensional symmetric Fock basis, that of $n$ photons and $m$ optical modes
        """

        # no multi-photon unitary transformation is required if n = 1
        if self.n == 1:
            return U

        # vectorially build all U_{S,T} matrices required to compute the multi-photon transform
        U_STs = vmap(lambda S, T: U[:, T][S, :])(*self.modeBasis_combos)

        # vectorially compute the permanents of all U_{S,T} matrices
        perms = vmap(self.calculator.perm)(U_STs)

        PhiU: jnp.ndarray = (perms * self.norms).reshape(self.N, self.N)
        return PhiU


@partial(jit, static_argnums=(1,))
def asymmetric_transform(U: ArrayLike, n: int) -> jnp.ndarray:
    """Perform a asymmetric multi-photon unitary transformation on a single-photon unitary $\\mathbf{U}$.

    Args:
        U: $m\\times m$ single-photon unitary, $\\mathbf{U}$, to transform
        n: number of photons, $n$

    Returns:
        $N\times N$ multi-photon unitary, $\\boldsymbol{\\Phi(\\mathbf{U})}$, in the $N$-dimensional asymmetric Fock basis
    """
    # ensure that U is a square jax array with elements
    U = jnp.asarray(U)
    Ushape = jnp.shape(U)
    assert Ushape[0] == Ushape[1], "Matrix must be square"
    assert Ushape[0] > 0, "Matrix must have elements"

    # stack U n times and kronecker product them all together
    Us = jnp.dstack([U] * n).T
    PhiU: jnp.ndarray = reduce(jnp.kron, Us)
    return PhiU
