"""
The `quotonic.clements` module includes ...
"""

from functools import partial, reduce
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit


class Mesh:
    """Mesh of Mach-Zehnder interferometers arranged in the Clements configuration.

    Attributes:
        m (int): number of optical modes, $m$
    """

    def __init__(
        self,
        m: int,
        ell_mzi: Optional[np.ndarray] = None,
        ell_ps: Optional[np.ndarray] = None,
        t_dc: Optional[np.ndarray] = None,
    ) -> None:
        """Initialization of a Mach-Zehnder interferometer mesh arranged in the Clements configuration.

        Args:
            m: number of optical modes, $m$
            ell_mzi: $m\\times m$ array containing the percentage loss per arm of the interferometer mesh, for each column of MZIs respectively
            ell_ps: $m$-length array containing the percentage loss for each of the output phase shifters
            t_dc: $2\\times m(m-1)/2$ array containing the splitting ratio (T:R) of each directional coupler in the mesh, organized such that each column
                corresponds to one MZI, the top row being the first directional coupler and the bottom being the second, where the MZIs are ordered from
                top to bottom followed by left to right across the mesh
        """

        # fill in missing mesh properties
        if ell_mzi is None:
            ell_mzi = np.zeros((m, m), dtype=float)
        if ell_ps is None:
            ell_ps = np.zeros(m, dtype=float)
        if t_dc is None:
            t_dc = 0.5 * np.ones((2, m * (m - 1) // 2), dtype=float)

        # check the validity of the provided properties
        assert m > 1, "Cannot create any kind of interferometer with less than 2 modes"
        assert (ell_mzi.shape[0] == ell_mzi.shape[1]) and (
            ell_mzi.shape[0] == m
        ), "Loss per arm of the interferometer mesh must take the form of an m x m array"
        assert len(ell_ps) == m, "Loss per output phase shifter in the mesh must be an m-length array"
        assert (t_dc.shape[0] == 2) and (
            t_dc.shape[1] == m * (m - 1) // 2
        ), "Directional coupler splitting ratios (T:R) must be passed as a 2 x m(m-1)/2 array"

        # store the properties of the mesh internally
        self.m = m
        self.ell_mzi = jnp.asarray(ell_mzi)
        self.ell_ps = jnp.asarray(ell_ps)
        self.t_dc = jnp.asarray(t_dc)

    @partial(jit, static_argnums=(0,))
    def mzi(
        self,
        phi: float,
        theta: float,
    ) -> jnp.ndarray:
        """Construct $2\\times 2$ Mach-Zehnder interferometer transfer matrix.

        Args:
            phi: phase shift $\\phi$, in radians
            theta: phase shift $\\theta$, in radians

        Returns:
            A $2\\times 2$ complex 2D array representation of the Mach-Zehnder interferometer transfer matrix
        """
        T00 = jnp.exp(1j * phi) * jnp.sin(theta)
        T01 = jnp.cos(theta)
        T10 = jnp.exp(1j * phi) * jnp.cos(theta)
        T11 = -jnp.sin(theta)
        return 1j * jnp.exp(1j * theta) * jnp.array([[T00, T01], [T10, T11]])

    @partial(jit, static_argnums=(0,))
    def mzi_inv(
        self,
        phi: float,
        theta: float,
    ) -> jnp.ndarray:
        """Construct $2\\times 2$ inverse of a Mach-Zehnder interferometer transfer matrix.

        Args:
            phi: phase shift $\\phi$, in radians
            theta: phase shift $\\theta$, in radians

        Returns:
            A $2\\times 2$ complex 2D array representation of the inverse of the Mach-Zehnder interferometer transfer matrix
        """
        T00 = jnp.exp(-1j * phi) * jnp.sin(theta)
        T01 = jnp.exp(-1j * phi) * jnp.cos(theta)
        T10 = jnp.cos(theta)
        T11 = -jnp.sin(theta)
        return -1j * jnp.exp(-1j * theta) * jnp.array([[T00, T01], [T10, T11]])

    @partial(jit, static_argnums=(0,))
    def encode(
        self,
        phi: jnp.ndarray,
        theta: jnp.ndarray,
        delta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Encode a Mach-Zehnder interferometer mesh in the Clements configuration from an array of phase shifts.

        Args:
            phi: phase shifts, $\\phi$, for all MZIs in the mesh
            theta: phase shifts, $\\theta$, for all MZIs in the mesh
            delta: phase shifts, $\\delta$, applied in each mode at the output of the mesh

        Returns:
            An $m\\times m$ 2D array representative of the linear unitary transformation, $\\mathbf{U}(\\boldsymbol{\\phi}, \\boldsymbol{\\theta})$, enacted by the Clements mesh
        """

        # check the validity of the mesh and the provided phases
        assert self.m > 2, "Clements encoding is only relevant for m > 2, see 'mzi' otherwise"
        assert len(phi) == int(self.m * (self.m - 1) / 2), "There must be exactly m(m-1)/2 phi phase shifts"
        assert len(theta) == int(self.m * (self.m - 1) / 2), "There must be exactly m(m-1)/2 theta phase shifts"
        assert len(delta) == int(self.m), "There must be exactly m delta phase shifts"

        # ensure that the provided phases are jax arrays
        phi = jnp.asarray(phi)
        theta = jnp.asarray(theta)
        delta = jnp.asarray(delta)

        # iterate through each column of the mesh, constructing and multiplying the transformations for each MZI
        ind = 0
        m_2 = self.m // 2
        odd = self.m % 2  # equals 0 if even, 1 if odd
        even = (self.m + 1) % 2  # equals 0 if odd, 1 if even
        columns = []
        for col in range(self.m):
            # calculate whether MZIs should be inserted from mode 0 or mode 1
            m0 = col % 2

            # extract the parameters for the MZIs in this column
            theta_col = theta[ind : ind + m_2 - m0 * even]
            phi_col = phi[ind : ind + m_2 - m0 * even]
            t_dc_col = self.t_dc[:, ind : ind + m_2 - m0 * even]
            ind += m_2 - m0 * even

            # construct matrices that describe the transformations enacted by two full columns of T:R directional couplers, nominally 50:50
            dc_diag = jnp.diag(
                jnp.pad(
                    jnp.repeat(jnp.sqrt(t_dc_col[0]), 2),
                    (m0, (m0 + odd) % 2),
                    "constant",
                    constant_values=(1.0, 1.0),
                )
            )
            dc_offdiag_up = jnp.roll(
                jnp.diag(
                    jnp.pad(
                        jnp.dstack((1j * jnp.sqrt(1.0 - t_dc_col[0]), jnp.zeros(m_2 - m0 * even, dtype=complex))).flatten(),
                        (m0, (m0 + odd) % 2),
                        "constant",
                        constant_values=(0.0, 0.0),
                    )
                ),
                1,
            )
            dc_offdiag_down = jnp.roll(
                jnp.diag(
                    jnp.pad(
                        jnp.dstack((jnp.zeros(m_2 - m0 * even, dtype=complex), 1j * jnp.sqrt(1.0 - t_dc_col[0]))).flatten(),
                        (m0, (m0 + odd) % 2),
                        "constant",
                        constant_values=(0.0, 0.0),
                    )
                ),
                -1,
            )
            dc1 = dc_diag + dc_offdiag_up + dc_offdiag_down

            dc_diag = jnp.diag(
                jnp.pad(
                    jnp.repeat(jnp.sqrt(t_dc_col[1]), 2),
                    (m0, (m0 + odd) % 2),
                    "constant",
                    constant_values=(1.0, 1.0),
                )
            )
            dc_offdiag_up = jnp.roll(
                jnp.diag(
                    jnp.pad(
                        jnp.dstack((1j * jnp.sqrt(1.0 - t_dc_col[1]), jnp.zeros(m_2 - m0 * even, dtype=complex))).flatten(),
                        (m0, (m0 + odd) % 2),
                        "constant",
                        constant_values=(0.0, 0.0),
                    )
                ),
                1,
            )
            dc_offdiag_down = jnp.roll(
                jnp.diag(
                    jnp.pad(
                        jnp.dstack((jnp.zeros(m_2 - m0 * even, dtype=complex), 1j * jnp.sqrt(1.0 - t_dc_col[1]))).flatten(),
                        (m0, (m0 + odd) % 2),
                        "constant",
                        constant_values=(0.0, 0.0),
                    )
                ),
                -1,
            )
            dc2 = dc_diag + dc_offdiag_up + dc_offdiag_down

            # construct matrices that describe the transformations enacted by full columns of phi & 2theta phase shifters respectively
            ps_phi = jnp.diag(
                jnp.pad(
                    jnp.dstack((jnp.exp(1j * phi_col), jnp.ones(m_2 - m0 * even, dtype=complex))).reshape(self.m - odd - 2 * m0 * even),
                    (m0, (m0 + odd) % 2),
                    "constant",
                    constant_values=(1.0, 1.0),
                )
            )
            ps_2theta = jnp.diag(
                jnp.pad(
                    jnp.dstack((jnp.exp(2j * theta_col), jnp.ones(m_2 - m0 * even, dtype=complex))).reshape(self.m - odd - 2 * m0 * even),
                    (m0, (m0 + odd) % 2),
                    "constant",
                    constant_values=(1.0, 1.0),
                )
            )

            # construct matrix the describes the loss per arm of the interferometer mesh for this column
            loss = jnp.diag(jnp.sqrt(1.0 - self.ell_mzi[:, col]))

            # multiply each component of the MZI together to form the full transformation from this column
            column = loss @ dc2 @ ps_2theta @ dc1 @ ps_phi

            # add to list of column matrices that will multiplied together at the end
            columns.append(column)

        # construct matrix that describes the transformation enacted by the output phase shifters in each mode
        ps_delta = jnp.diag(jnp.exp(1j * delta))

        # construct matrix that describes the loss per output phase shifter
        loss = jnp.diag(jnp.sqrt(1.0 - self.ell_ps))

        # multiply all the columns, followed by the output phase shifters, to construct the full mesh transformation
        U = loss @ ps_delta @ reduce(jnp.matmul, columns[::-1])
        return U

    def decode(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Clements decomposition on a square $m\\times m$ unitary matrix.

        Args:
            U: $m\\times m$ unitary matrix, $\\mathbf{U}(\\boldsymbol{\\phi}, \\boldsymbol{\\theta}, \\boldsymbol{\\delta})$, to perform Clements decomposition on

        Returns:
            Tuple of three arrays, containing phase shifts $\\boldsymbol{\\phi}$, $\\boldsymbol{\\theta}$, $\\boldsymbol{\\delta}$ respectively, that yield $\\mathbf{U}$ when applied in the Clements mesh
        """

        # check that U is m x m
        assert len(U.shape) == 2, "Unitary matrix must be a 2D array"
        assert U.shape[0] == self.m, "Unitary matrix must be m x m"
        assert U.shape[1] == self.m, "Unitary matrix must be m x m"

        # initialize lists of MZIs and T_{p,q} applied from the left
        MZIs = []
        T_lefts = []

        # need to zero out m - 1 diagonal sections from the matrix U
        for i in range(self.m - 1):
            # for even i, multiply from the right
            if i % 2 == 0:
                for j in range(i + 1):
                    # store modes that T acts on
                    p = i - j
                    q = i - j + 1

                    # compute phi, theta to 0 out matrix element
                    phi = np.pi if U[self.m - j - 1, i - j + 1] == 0 else np.pi + np.angle(U[self.m - j - 1, i - j] / U[self.m - j - 1, i - j + 1])
                    theta = np.pi / 2 - np.arctan2(np.abs(U[self.m - j - 1, i - j]), np.abs(U[self.m - j - 1, i - j + 1]))

                    # from phi, theta, construct T_{p,q}^{-1}, then right-multiply
                    T_right = np.eye(self.m, dtype=complex)
                    T_right[p : q + 1, p : q + 1] = self.mzi_inv(phi, theta)
                    U = np.dot(U, T_right)

                    # append MZI to list, noting modes and phases
                    MZIs.append({"pq": (p, q), "phi": phi, "theta": theta})

            # for odd i, multiply from the left
            else:
                for j in range(i + 1):
                    # store modes that T acts on
                    p = self.m + j - i - 2
                    q = self.m + j - i - 1

                    # compute phi, theta to 0 out matrix element
                    phi = np.pi if U[self.m + j - i - 2, j] == 0 else np.pi + np.angle(-U[self.m + j - i - 1, j] / U[self.m + j - i - 2, j])
                    theta = np.pi / 2 - np.arctan2(np.abs(U[self.m + j - i - 1, j]), np.abs(U[self.m + j - i - 2, j]))

                    # from phi, theta, construct T_{p,q}, then left-multiply
                    T_left = np.eye(self.m, dtype=complex)
                    T_left[p : q + 1, p : q + 1] = self.mzi(phi, theta)
                    U = np.dot(T_left, U)

                    # append left-multiplying T_{p,q} to list, noting modes and phases
                    T_lefts.append({"pq": (p, q), "phi": phi, "theta": theta})

        # check that the resultant matrix, $D$, is diagonal
        assert np.allclose(np.abs(np.diag(U)), np.ones(self.m)), "Decomposition did not yield a diagonal matrix D"

        # rearrange the transformations to match the encoding scheme
        for T in reversed(T_lefts):
            # extract modes, phases for the T_{p,q}
            p, q = T["pq"]
            phi = T["phi"]
            theta = T["theta"]

            # construct T_{p,q}^{-1}, then left-multiply
            T_left_inv = np.eye(self.m, dtype=complex)
            T_left_inv[p : q + 1, p : q + 1] = self.mzi_inv(phi, theta)
            U = np.dot(T_left_inv, U)

            # compute phi, theta that allow T_{p,q}^{-1} to be multiplied on the right
            phi = np.pi if U[q, q] == 0 else np.pi + np.angle(U[q, p] / U[q, q])
            theta = np.pi / 2 - np.arctan2(np.abs(U[q, p]), np.abs(U[q, q]))

            # from phi, theta, construct T_{p,q}^{-1}, then right-multiply
            T_right = np.eye(self.m, dtype=complex)
            T_right[p : q + 1, p : q + 1] = self.mzi_inv(phi, theta)
            U = np.dot(U, T_right)

            # append MZI to list, noting modes and phases
            MZIs.append({"pq": (p, q), "phi": phi, "theta": theta})

        # check that the resultant matrix, $D'$, is diagonal
        assert np.allclose(np.abs(np.diag(U)), np.ones(self.m)), "Decomposition did not yield a diagonal matrix D'"

        # compute output phases from the diagonal of the resultant matrix U
        delta = np.angle(np.diag(U))

        # sort the MZIs by mode pair and the order in which they must be applied
        sorted_MZIs: list = [[] for _ in range(self.m - 1)]
        for MZI in MZIs:
            sorted_MZIs[MZI["pq"][0]].append(MZI)

        # extract the phi, theta phase shifts from the sorted MZIs in the correct order
        phi = np.zeros(int(self.m * (self.m - 1) / 2))
        theta = np.zeros(int(self.m * (self.m - 1) / 2))

        indp = 0
        indt = 0
        for i in range(self.m):
            m0 = i % 2
            for j in range(m0, self.m - 1, 2):
                MZI = sorted_MZIs[j].pop(0)
                phi[indp] = MZI["phi"]
                theta[indt] = MZI["theta"]

                indp += 1
                indt += 1

        return phi, theta, delta
