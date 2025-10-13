"""
The `quotonic.perm` module includes functions and classes used to compute matrix permanents as efficiently as
possible when using `jax` as a backend. Currently, there is support both for Ryser's algorithm and the
Balasubramanian-Bax-Franklin-Glynn (BBFF) algorithm, each using Gray code ordering.

This code was inspired by [Piquasso](https://github.com/Budapest-Quantum-Computing-Group/piquasso), [The Walrus](
https://github.com/XanaduAI/thewalrus), and [Cascaded Optical Systems Approach to Neural Networks (CasOptAx)](
https://github.com/JasvithBasani/CasOptAx) as originally designed for use in [J. R. Basani *et al*., "Universal
logical quantum photonic neural network processor via cavity-assisted interactions", *npj Quantum Inf* **11**,
142 (2025)](https://doi.org/10.1038/s41534-025-01096-9).
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.typing import DTypeLike

from quotonic.types import jnp_ndarray


@vmap
def prep_gray_code(i: int) -> tuple:
    """Preparation of Gray code for computing matrix permanents.

    This function is wrapped with `jax.vmap` such that it can be used vectorially.

    Args:
        i: index typically used in loops that prepare Gray code

    Returns:
        A tuple including the difference between the old Gray value and the new, followed by the direction of the
            algorithm
    """
    old_gray = i ^ (i // 2)
    new_gray = (i + 1) ^ ((i + 1) // 2)
    gray_diff = old_gray ^ new_gray
    direction = lax.cond(
        old_gray > new_gray,
        lambda: 1,
        lambda: -1,
    )
    return gray_diff, direction


@jit
def calc_perm_ryser(U: jnp_ndarray) -> DTypeLike:
    """Compute the permanent of a square matrix $\\mathbf{U}$ using Ryser's algorithm with Gray code ordering.

    This function is wrapped with `jax.jit` such that it can be compiled at runtime.

    Args:
        U: square matrix whose permanent is to be computed

    Returns:
        Permanent of the given square matrix
    """

    n = U.shape[0]
    two_to_n = 2**n

    # prepare Gray code for the permanent calculation
    gray_diff, direction = prep_gray_code(jnp.arange(two_to_n - 1, dtype=jnp.int16))  # type: ignore
    gray_diff_ind = jnp.array(jnp.log2(gray_diff), dtype=jnp.int16)
    sign = jnp.resize(jnp.array([-1, 1], dtype=jnp.int16), two_to_n - 1)

    # calculate permanent by vectorizing Ryser's algorithm
    perm: DTypeLike = jnp.sum(
        sign * jnp.prod(jnp.cumsum(vmap(lambda ind, direc: U[ind] * direc)(gray_diff_ind, direction), axis=0), axis=1)
    )
    return perm


@jit
def calc_perm_bbfg(U: jnp_ndarray) -> DTypeLike:
    """Compute the permanent of a square matrix $\\mathbf{U}$ using the BBFG algorithm with Gray code ordering.

    This function is wrapped with `jax.jit` such that it can be compiled at runtime.

    Args:
        U: square matrix whose permanent is to be computed

    Returns:
        Permanent of the given square matrix
    """

    n = U.shape[0]
    N = 2 ** (n - 1)

    # prepare Gray code for the permanent calculation
    gray_diff, direction = prep_gray_code(jnp.arange(N - 1, dtype=jnp.int16))  # type: ignore
    gray_diff_ind = jnp.array(jnp.log2(gray_diff), dtype=jnp.int16)
    sign = jnp.resize(jnp.array([-1, 1], dtype=jnp.int16), N - 1)

    # calculate permanent by vectorizing the BBFG algorithm
    perm: DTypeLike = (
        jnp.prod(jnp.sum(U, axis=0))
        + jnp.sum(
            sign
            * (
                jnp.prod(
                    jnp.sum(U, axis=0)
                    + jnp.cumsum(
                        vmap(lambda ind, direc: U[ind] * direc * 2)(gray_diff_ind, direction),
                        axis=0,
                    ),
                    axis=1,
                )
            )
        )
    ) / N
    return perm


@partial(jit, static_argnums=(1,))
def calc_perm(U: jnp_ndarray, algo: str = "bbfg") -> DTypeLike:
    """Compute the permanent of a square matrix $\\mathbf{U}$.

    Args:
        U: square matrix whose permanent is to be computed
        algo: algorithm to compute the permanent with if the matrix dim is greater than 3, either "bbfg" or "ryser"

    Returns:
        Permanent of the given square matrix
    """

    # extract the dimension of the square matrix U
    Ushape = jnp.shape(U)
    assert Ushape[0] == Ushape[1], "Matrix must be square"
    assert Ushape[0] > 0, "Matrix must have elements"
    N = Ushape[0]

    if N == 1:
        return U[0, 0]

    if N == 2:
        return U[0, 0] * U[1, 1] + U[0, 1] * U[1, 0]

    if N == 3:
        return (
            U[0, 2] * U[1, 1] * U[2, 0]
            + U[0, 1] * U[1, 2] * U[2, 0]
            + U[0, 2] * U[1, 0] * U[2, 1]
            + U[0, 0] * U[1, 2] * U[2, 1]
            + U[0, 1] * U[1, 0] * U[2, 2]
            + U[0, 0] * U[1, 1] * U[2, 2]
        )

    perm: DTypeLike = calc_perm_bbfg(U) if algo == "bbfg" else calc_perm_ryser(U)
    return perm


class Permanent:
    """Wrapper class for computing permanents of matrices of constant dimension $n$ while the Gray code overhead is
    stored in memory.

    Attributes:
        n (int): dimension of the square matrices, $n$
        perm (callable): function that computes the permanent of a given $n\\times n$ matrix $\\mathbf{U}$
        gray_diff_ind (jnp_ndarray): array of matrix indices for computing permanents using Gray code ordering,
            defaults to an empty array if $n < 3$
        direction (jnp_ndarray): array of factors to apply in individual steps of the permanent calculation
            algorithms, defaults to an empty array if $n < 3$
        sign (jnp_ndarray): array of $\\pm 1$ factors to apply to the results of individual steps of the permanent
            calculation algorithms, defaults to an empty array if $n < 3$
        N (int): if the BBFG algorithm is selected, $N = 2^{n-1}$, otherwise, it is unused and defaults to 0
    """

    def __init__(self, n: int, algo: str = "bbfg") -> None:
        """Initialization of a Permanent instance.

        Args:
            n: dimension of the square matrices whose permanents are to be computed, $n$
            algo: algorithm to compute the permanent with if the matrix dimension is greater than 3, either "bbfg" or "ryser"
        """

        # check the validity of the provided arguments
        assert n > 0, "Matrices must have elements to compute a permanent"
        assert (algo == "bbfg") or (algo == "ryser"), "The only algorithm options are 'bbfg' or 'ryser'"

        # store the dimension of the square matrices, then compute overhead if necessary
        self.n = n
        if n < 3:
            self.gray_diff_ind = jnp.array(())
            self.direction = jnp.array(())
            self.sign = jnp.array(())
            self.N = 0
            if n == 1:
                self.perm = jit(lambda U: U[0, 0])
            elif n == 2:
                self.perm = jit(lambda U: U[0, 0] * U[1, 1] + U[0, 1] * U[1, 0])
            elif n == 3:
                self.perm = jit(
                    lambda U: U[0, 2] * U[1, 1] * U[2, 0]
                    + U[0, 1] * U[1, 2] * U[2, 0]
                    + U[0, 2] * U[1, 0] * U[2, 1]
                    + U[0, 0] * U[1, 2] * U[2, 1]
                    + U[0, 1] * U[1, 0] * U[2, 2]
                    + U[0, 0] * U[1, 1] * U[2, 2]
                )
        else:
            if algo == "bbfg":
                self.prep_gray_code_bbfg()
                self.perm = self.perm_bbfg
            elif algo == "ryser":
                self.prep_gray_code_ryser()
                self.perm = self.perm_ryser
                self.N = 0

    def prep_gray_code_bbfg(self) -> None:
        """Preparation of Gray code for computing matrix permanents using the BBFG algorithm."""
        self.N = 2 ** (self.n - 1)
        gray_diff, direction = prep_gray_code(jnp.arange(self.N - 1, dtype=jnp.int16))  # type: ignore
        self.direction = 2 * direction
        self.gray_diff_ind = jnp.array(jnp.log2(gray_diff), dtype=jnp.int16)
        self.sign = jnp.resize(jnp.array([-1, 1], dtype=jnp.int16), (self.N - 1,))

    def prep_gray_code_ryser(self) -> None:
        """Preparation of Gray code for computing matrix permanents using Ryser's algorithm."""
        two_to_n = 2**self.n
        gray_diff, self.direction = prep_gray_code(jnp.arange(two_to_n - 1, dtype=jnp.int16))  # type: ignore
        self.gray_diff_ind = jnp.array(jnp.log2(gray_diff), dtype=jnp.int16)
        self.sign = jnp.resize(jnp.array([-1, 1], dtype=jnp.int16), (two_to_n - 1,))

    @partial(jit, static_argnums=(0,))
    def perm_bbfg(self, U: jnp_ndarray) -> DTypeLike:
        """Compute the permanent of a square matrix $\\mathbf{U}$ using the BBFG algorithm with Gray code ordering.

        Args:
            U: square matrix whose permanent is to be computed

        Returns:
            Permanent of the given square matrix
        """
        return (
            jnp.prod(jnp.sum(U, axis=0))
            + jnp.sum(
                self.sign
                * (
                    jnp.prod(
                        jnp.sum(U, axis=0)
                        + jnp.cumsum(
                            vmap(lambda ind, direc: U[ind] * direc)(self.gray_diff_ind, self.direction),
                            axis=0,
                        ),
                        axis=1,
                    )
                )
            )
        ) / self.N

    @partial(jit, static_argnums=(0,))
    def perm_ryser(self, U: jnp_ndarray) -> DTypeLike:
        """Compute the permanent of a square matrix $\\mathbf{U}$ using Ryser's algorithm with Gray code ordering.

        Args:
            U: square matrix whose permanent is to be computed

        Returns:
            Permanent of the given square matrix
        """
        return jnp.sum(
            self.sign
            * jnp.prod(
                jnp.cumsum(vmap(lambda ind, direc: U[ind] * direc)(self.gray_diff_ind, self.direction), axis=0), axis=1
            )
        )


class EmptyPermanent:
    """Placeholder permanent class for mypy typing."""

    def __init__(self) -> None:
        """Initialization of an EmptyPermanent instance."""
        self.fake = True

    def perm(self, U: jnp_ndarray) -> jnp_ndarray:
        """Fake method required for the placeholder."""
        return U[0, 0] if self.fake else U[0, 0]
