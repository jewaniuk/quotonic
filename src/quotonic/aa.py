"""
The `quotonic.aa` module includes functions and classes required to perform transformations from a single-photon
unitary matrix ($m\\times m$ where $m$ is the number of optical modes) to a multi-photon unitary matrix ($N\\times
N$ where $N$ is the dimension of the Fock basis for $n$ photons and $m$ modes). This transformation is required to
describe how unitaries encoded in the Clements configuration act on states resolved in a general basis of multiple
photons since $N \\neq m : n > 1$.

The code in this module has been inspired by the description of multi-photon unitary transformations in [S.
Aaronson & A. Arkhipov, “The Computational Complexity of Linear Optics”, arXiv:1011.3245 [quant-ph] (2010)](
https://arxiv.org/abs/1011.3245), by [Bosonic: A Quantum Optics Library](https://github.com/steinbrecher/bosonic),
as originally designed for use in [G. R. Steinbrecher *et al*., “Quantum optical neural networks”, *npj Quantum Inf*
**5**, 60 (2019)](https://doi.org/10.1038/s41534-019-0174-7), and by [Cascaded Optical Systems Approach to Neural
Networks (CasOptAx)](https://github.com/JasvithBasani/CasOptAx) as originally designed for use in [J. R. Basani *et
al*., "Universal logical quantum photonic neural network processor via cavity-assisted interactions", *npj Quantum
Inf* **11**, 142 (2025)](https://doi.org/10.1038/s41534-025-01096-9)/
"""

from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import factorial

from quotonic.fock import build_secq_basis, build_firq_basis_wo_dups
from quotonic.perm import EmptyPermanent, Permanent
from quotonic.types import jnp_ndarray


def gen_basis_combos(basis: jnp_ndarray) -> Tuple[jnp_ndarray, jnp_ndarray]:
    """Generate combinations of basis states corresponding to each element of a matrix resolved in the basis.

    By placing the elements of the two tuples next to each other, you end up with (row, column) combinations for each
    matrix element in the input basis. This is used to vectorially compute all matrix elements.

    Args:
        basis: $N\\times x$ array that catalogs all states in the $N$-dimensional basis, where each state has $x$ labels

    Returns:
        Tuple of $N^2\\times x$ arrays, the first of which repeats each state $N$ times vertically before moving to
            the next state, the second of which repeats the entire basis in the order given $N$ times
    """
    N = jnp.shape(basis)[0]
    return jnp.repeat(basis, N, axis=0), jnp.vstack([basis] * N)


@vmap
def vectorial_factorial(x: int | float) -> int | float:
    """Compute the factorial on the input vectorially.

    Simply put, this function wraps `jax.scipy.special.factorial` with the `jax.vmap` decorator. It doesn't really
    need its own documented function, but I thought the name `vectorial_factorial` sounded cool.

    Args:
        x: integer to compute the factorial of

    Returns:
        Factorial of the input
    """
    return factorial(x)  # type: ignore


@vmap
def calc_norm(S: jnp_ndarray, T: jnp_ndarray) -> float:
    """Calculate the normalization factor for an element of a multi-photon unitary in the second quantization Fock
    basis.

    Each normalization constant involves the product of factorials for each mode of the two basis states that define
    an element of the multi-photon unitary $\\boldsymbol{\\Phi}(\\mathbf{U})$. The mathematical form of the
    normalization constants is given in the documentation of `SecqTransformer.transform`. This function computes the
    required product of factorials for each basis state, square roots this product, then combines all results in a
    2D array that stores the normalization constant for  each element of $\\boldsymbol{\\Phi}(\\mathbf{U})$ in the
    corresponding position.

    This function is wrapped by the `jax.vmap` decorator such that it can be used vectorially.

    Args:
        S: state $\\left| S\\right\\rangle$ corresponding to the row of the multi-photon unitary, length $m$
        T: state $\\left| T\\right\\rangle$ corresponding to the column of the multi-photon unitary, length $m$

    Returns:
        Normalization factor for symmetric multi-photon unitary element
            $\\left\\langle S\\right|\\boldsymbol{\\Phi}(\\mathbf{U})\\left| T\\right\\rangle$
    """
    return 1.0 / jnp.sqrt(jnp.prod(vectorial_factorial(jnp.concatenate((S, T)))))  # type: ignore


class SecqTransformer:
    """Wrapper class for performing multi-photon unitary transformations in the second quantization basis while the
    required overhead is stored in memory.

    Attributes:
        n (int): number of photons, $n$
        N (int): dimension of the second quantization Fock basis for $n$ photons and $m$ optical modes
        firq_combos (Tuple[jnp_ndarray, jnp_ndarray]): tuple of $N^2\\times n$ arrays, the first of which repeats each
            first-quantized state $N$ times vertically before moving to the next state, the second of which repeats the
            entire first-quantized basis (without indistinguishable duplicates) in the order given $N$ times; defaults
            to a tuple of empty arrays if $n = 1$
        norms (jnp_ndarray): normalization factors for each element of the multi-photon unitary, flattened to a
            $N^2\\times 1$ array, defaults to an empty array if $n = 1$
        calculator (Permanent | EmptyPermanent): instance of a wrapped permanent calculator that computes overhead
            for the selected algorithms ahead of time, defaults to EmptyPermanent if $n = 1$
    """

    def __init__(self, n: int, m: int, algo: str = "bbfg") -> None:
        """Initialization of a Second Quantization Transformer.

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
            # construct both bases representations
            firq_basis = jnp.asarray(build_firq_basis_wo_dups(n, m))
            basis = jnp.asarray(build_secq_basis(n, m))
            self.N = basis.shape[0]

            # stack & repeat the bases to form combinations that correspond to each multi-photon unitary matrix element
            self.firq_combos = gen_basis_combos(firq_basis)
            basis_combos = gen_basis_combos(basis)

            # vectorially compute the normalization factors for each element of the multi-photon unitary
            self.norms: jnp_ndarray = calc_norm(*basis_combos)  # type: ignore

            # instantiate a permanent calculator
            self.calculator: Permanent | EmptyPermanent = Permanent(n, algo=algo)
        else:
            self.N = m
            self.modeBasis_combos = (jnp.array(()), jnp.array(()))
            self.norms = jnp.array(())
            self.calculator = EmptyPermanent()

    @partial(jit, static_argnums=(0,))
    def transform(self, U: jnp_ndarray) -> jnp_ndarray:
        """Perform a multi-photon unitary transformation on a single-photon unitary $\\mathbf{U}$, in the second
        quantization Fock basis.

        This method constructs the corresponding multi-photon unitary, $\\boldsymbol{\\Phi}(\\mathbf{U})$, from input
        single-photon unitary $\\mathbf{U}$. Vectorially, each element of the multi-photon unitary $\\boldsymbol{
        \\Phi}(\\mathbf{U})$ is computed using the transformation of Aaronson & Arkhipov. Each  element can be
        denoted as $\\left\\langle S \\right|\\boldsymbol{\\Phi}(\\mathbf{U})\\left| T \\right\\rangle$ where
        $\\left|S\\right\\rangle = \\left|s_1,s_2,\\dots,s_m\\right\\rangle$, $\\left|T\\right\\rangle = \\left|t_1,
        t_2,\\dots,t_m\\right\\rangle$ represent arbitrary Fock basis states and $m$ denotes the number of optical
        modes. For a given element, an $m\\times n$ matrix, $\\mathbf{U}_T$, is constructed by taking $t_j$ copies of
        column $j$ in the input single-photon unitary $\\mathbf{U}$ for all $j \\in \\{1,\\dots,m\\}$. Next,
        an $n\\times n$ matrix, $\\mathbf{U}_{S,T}$, is constructed by taking $s_j$ copies of row $j$ in the
        previously generated matrix, $\\mathbf{U}_T$, for all $j \\in \\{1,\\dots,m\\}$. The matrix element of
        multi-photon unitary, $\\boldsymbol{\\Phi}(\\mathbf{U})$ is then given by,

        $$ \\left\\langle S\\right|\\boldsymbol{\\Phi}(\\mathbf{U})\\left| T\\right\\rangle =
        \\left\\langle s_1,s_2,\\dots,s_m\\right| \\boldsymbol{\\Phi}(\\mathbf{U})\\left|t_1,t_2,\\dots,
        t_m\\right\\rangle = \\frac{\\text{Per}(\\mathbf{U}_{S,T})}{\\sqrt{s_1!\\dots s_m!t_1!\\dots t_m!}}. $$

        As an example, consider a case where there are 2 photons ($n = 2$) and 3 modes ($m = 3$). The input
        Clements-encoded single-photon unitary is given by,

        $$ \\mathbf{U} = \\begin{pmatrix} u_{00} & u_{01} & u_{02} \\\\ u_{10} & u_{11} & u_{12} \\\\ u_{20} & u_{21}
        & u_{22} \\end{pmatrix}. $$

        To compute the matrix element $\\left\\langle 101\\right|\\boldsymbol{\\Phi}(\\mathbf{U})\\left|011\\right
        \\rangle$, first build $\\mathbf{U}_T$ by taking 0 copies of the first column of $\\mathbf{U}$, 1 copy of the
        second, and 1 copy of the third,

        $$ \\mathbf{U}_T = \\begin{pmatrix} u_{01} & u_{02} \\\\ u_{11} & u_{12} \\\\ u_{21} & u_{22} \\end{pmatrix}. $$

        Next, build $\\mathbf{U}_{S,T}$ by taking 1 copy of the first row of $\\mathbf{U}_T$, 0 copies of the second,
        and 1 copy of the third,

        $$ \\mathbf{U}_{S,T} = \\begin{pmatrix} u_{01} & u_{02} \\\\ u_{21} & u_{22} \\end{pmatrix}. $$

        The permanent of $\\mathbf{U}_{S,T}$ must be calculated to compute the corresponding matrix element of
        $\\boldsymbol{\\Phi}(\\mathbf{U})$. This is managed by the [perm](perm.md) module.

        Args:
            U: $m\\times m$ single-photon unitary, $\\mathbf{U}$, to transform

        Returns:
            $N\\times N$ multi-photon unitary, $\\boldsymbol{\\Phi(\\mathbf{U})}$, in the $N$-dimensional Fock
                basis, that of $n$ photons and $m$ optical modes
        """

        # no multi-photon unitary transformation is required if n = 1
        if self.n == 1:
            return U

        # vectorially build all U_{S,T} matrices required to compute the multi-photon transform
        U_STs = vmap(lambda S, T: U[:, T][S, :])(*self.firq_combos)

        # vectorially compute the permanents of all U_{S,T} matrices
        perms = vmap(self.calculator.perm)(U_STs)

        PhiU: jnp_ndarray = (perms * self.norms).reshape(self.N, self.N)
        return PhiU
