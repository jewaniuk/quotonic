"""
The `quotonic.qpnn` module includes ...
"""

from functools import partial, reduce
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.tree import map as tree_map
from jax.typing import DTypeLike

from quotonic.aa import SymmetricTransformer, asymmetric_transform
from quotonic.clements import Mesh
from quotonic.fock import build_symm_basis, calc_asymm_dim, calc_symm_dim
from quotonic.nl import build_kerr, build_photon_mp
from quotonic.utils import comp_indices_from_symm_fock


class QPNN:
    """Base class for a quantum photonic neural network (QPNN).

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the relevant Fock basis for $n$ photons and $m$ optical modes
    """

    def __init__(self, n: int, m: int, L: int, basis_type: str = "symmetric") -> None:
        """Initialization of a QPNN instance.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            basis_type: specifies whether the QPNN is resolved in the symmetric or asymmetric Fock basis
        """

        # check that basis_type is valid
        assert (basis_type == "symmetric") or (basis_type == "asymmetric"), "Basis type must be 'symmetric' or 'asymmetric'"

        # store the provided properties of the QPNN, compute others
        self.n = n
        self.m = m
        self.L = L
        self.N = calc_symm_dim(n, m) if basis_type == "symmetric" else calc_asymm_dim(n, m)


class IdealQPNN(QPNN):
    """Class for an idealized QPNN based on single-site Kerr-like nonlinearities.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the symmetric Fock basis for $n$ photons and $m$ optical modes
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer meshes) to be encoded
        transformer (SymmetricTransformer): object containing methods that compute multi-photon unitary transformations of the linear layers
        varphi (float): effective nonlinear phase shift, $\\varphi$
        kerr (jnp.ndarray): $N\\times N$ array, the matrix representation of the set of single-site Kerr-like nonlinearities resolved in the symmetric Fock basis
        K (int): number of input-target state pairs in the QPNN training set, defaults to 0 if none provided
        psi_in (jnp.ndarray): $K\\times N$ array containing the $K$ input states in the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        psi_targ (jnp.ndarray): $K\\times N$ array containing the $K$ target states in the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
    """

    def __init__(self, n: int, m: int, L: int, varphi: float = np.pi, training_set: Optional[tuple] = None) -> None:
        """Initialization of an Ideal QPNN instance.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            varphi: effective nonlinear phase shift, $\\varphi$
            training_set: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """

        super().__init__(n, m, L)

        # instantiate a Clements mesh to act as the pathway to encoding the linear layers
        self.mesh = Mesh(m)

        # instantiate symmetric transfomer required for the multi-photon unitary transformations of the linear layers
        self.transformer = SymmetricTransformer(n, m)

        # store the provided effective nonlinear phase shift, construct the corresponding nonlinear Kerr-like unitary
        self.varphi = varphi
        self.kerr = jnp.asarray(build_kerr(n, m, varphi))

        # prepare the training set attributes whether one was provided or not
        self.training_set = training_set if training_set is not None else (jnp.array(()), jnp.array(()))

    @property
    def training_set(self) -> tuple:
        """Training set of the QPNN.

        Returns:
            A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        return self.psi_in, self.psi_targ

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set of the QPNN.

        Args:
            tset: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        self.psi_in = jnp.asarray(tset[0])
        self.psi_targ = jnp.asarray(tset[1])
        self.K = 0 if self.psi_in.size == 0 else self.psi_in.shape[0]

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Build a matrix representation of the QPNN from all its layers and components.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            $N\\times N$ array, the matrix representation of the QPNN resolved in the symmetric Fock basis
        """

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = vmap(self.mesh.encode)(phi, theta, delta)

        # perform the multi-photon unitary transformations for each linear layer
        multi_photon_Us = vmap(self.transformer.transform)(single_photon_Us)

        # for each linear layer up to the last one, multiply the nonlinear unitary and multi-photon unitary together
        layers = vmap(lambda PhiU: self.kerr @ PhiU)(multi_photon_Us[0 : self.L - 1])

        # stack the layers together, including the final linear layer
        layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, self.N, self.N))))

        # multiply all the layers together
        S: jnp.ndarray = reduce(jnp.matmul, layers[::-1])
        return S

    @partial(jit, static_argnums=(0,))
    def calc_fidelity(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> DTypeLike:
        """Calculate the fidelity of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Fidelity of the QPNN
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function
        S = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.dot(S, psi))(self.psi_in)

        # compute the fidelity by first computing it for all K input-target pairs, then averaging
        Fs = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ, psi_out)
        F = jnp.sum(Fs) / self.K

        return F


class ImperfectQPNN(QPNN):
    """Class for experimental modelling of QPNNs based on single-site Kerr-like nonlinearities.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the symmetric Fock basis for $n$ photons and $m$ optical modes
        meshes (tuple): tuple of $L$ objects containing methods that allow each linear layer (i.e. rectangular Mach-Zehnder interferometer meshes) to be encoded
        ell_mzi (tuple): nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
        ell_ps (tuple): nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
        t_dc (tuple): directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
        transformer (SymmetricTransformer): object containing methods that compute multi-photon unitary transformations of the linear layers
        varphi (float): effective nonlinear phase shift, $\\varphi$
        nl (jnp.ndarray): $N\\times N$ array, the matrix representation of a set of single-site Kerr-like nonlinearities resolved in the symmetric Fock basis
        K (int): number of input-target state pairs in the QPNN training set, defaults to 0 if none provided
        psi_in (jnp.ndarray): $K\\times N$ array containing the $K$ input states in the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        psi_targ (jnp.ndarray): $K\\times N$ array containing the $K$ target states in the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        comp_indices (jnp.ndarray): $2^n$-length array whose elements are the indices of the symmetric Fock basis where dual-rail encoded computational basis states lie
    """

    def __init__(
        self,
        n: int,
        m: int,
        L: int,
        varphi: float = np.pi,
        ell_mzi: tuple = (0.0, 0.0),
        ell_ps: tuple = (0.0, 0.0),
        t_dc: tuple = (0.5, 0.0),
        training_set: Optional[tuple] = None,
    ) -> None:
        """Initialization of an Imperfect QPNN instance.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            varphi: effective nonlinear phase shift, $\\varphi$
            ell_mzi: nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
            ell_ps: nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
            t_dc: directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
            training_set: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """

        super().__init__(n, m, L)

        # instantiate L Clements meshes, with losses and routing errors, that act as the pathway to encoding the linear layers
        self.ell_mzi = ell_mzi
        self.ell_ps = ell_ps
        self.t_dc = t_dc
        self.meshes = tuple([Mesh(m) for _ in range(L)])
        self.prep_meshes()

        # instantiate symmetric transfomer required for the multi-photon unitary transformations of the linear layers
        self.transformer = SymmetricTransformer(n, m)

        # store the provided effective nonlinear phase shift, construct the corresponding nonlinear Kerr-like unitary
        self.varphi = varphi
        self.nl = jnp.asarray(build_kerr(n, m, varphi))

        # prepare the training set attributes whether one was provided or not
        self.training_set = training_set if training_set is not None else (jnp.array(()), jnp.array(()))

        # compute overhead for conditional fidelity and logical rate calculations
        self.comp_indices = jnp.asarray(comp_indices_from_symm_fock(build_symm_basis(n, m)))

    @property
    def training_set(self) -> tuple:
        """Training set of the QPNN.

        Returns:
            A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        return self.psi_in, self.psi_targ

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set of the QPNN.

        Args:
            tset: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        self.psi_in = jnp.asarray(tset[0])
        self.psi_targ = jnp.asarray(tset[1])
        self.K = 0 if self.psi_in.size == 0 else self.psi_in.shape[0]

    def prep_meshes(self) -> None:
        """Prepare the Mach-Zehnder interferometer meshes for the linear layers of the network."""

        # for each layer, compute and apply new loss and splitting ratio values from their respective distributions
        for i in range(self.L):
            ells_mzi = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_mzi[0]),
                self.ell_mzi[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_mzi[0]),
                self.m**2,
            ).reshape((self.m, self.m))
            ells_ps = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_ps[0]),
                self.ell_ps[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_ps[0]),
                self.m,
            )
            ts_dc = np.random.normal(self.t_dc[0], self.t_dc[1], self.m * (self.m - 1)).reshape((2, self.m * (self.m - 1) // 2))

            self.meshes[i].ell_mzi = jnp.asarray(ells_mzi)
            self.meshes[i].ell_ps = jnp.asarray(ells_ps)
            self.meshes[i].t_dc = jnp.asarray(ts_dc)

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Build a matrix representation of the QPNN from all its layers and components.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            $N\\times N$ array, the matrix representation of the QPNN resolved in the symmetric Fock basis
        """

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = jnp.array([self.meshes[i].encode(phi[i], theta[i], delta[i]) for i in range(self.L)], dtype=complex)

        # perform the multi-photon unitary transformations for each linear layer
        multi_photon_Us = vmap(self.transformer.transform)(single_photon_Us)

        # for each linear layer up to the last one, multiply the nonlinear unitary and multi-photon unitary together
        layers = vmap(lambda PhiU: self.nl @ PhiU)(multi_photon_Us[0 : self.L - 1])

        # stack the layers together, including the final linear layer
        layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, self.N, self.N))))

        # multiply all the layers together
        S: jnp.ndarray = reduce(jnp.matmul, layers[::-1])
        return S

    @partial(jit, static_argnums=(0,))
    def calc_unc_fidelity(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> DTypeLike:
        """Calculate the unconditional fidelity of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Unconditional fidelity of the QPNN
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function
        S = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.dot(S, psi))(self.psi_in)

        # compute the unconditional fidelity by first computing it for all K input-target pairs, then averaging
        Fus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ, psi_out)
        Fu = jnp.mean(Fus)

        return Fu

    @partial(jit, static_argnums=(0,))
    def calc_performance_measures(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Calculate the unconditional fidelity, conditional fidelity, and logical rate of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Unconditional fidelity, conditional fidelity, and logical rate of the QPNN, as a tuple
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function
        S = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.dot(S, psi))(self.psi_in)

        # compute the unconditional fidelity by first computing it for all K input-target pairs, then averaging
        Fus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ, psi_out)
        Fu = jnp.mean(Fus)

        # compute the logical rate by first computing it for all K input-target pairs, then averaging
        rates = vmap(lambda psi: jnp.sum(jnp.abs(psi[self.comp_indices]) ** 2))(psi_out)
        rate = jnp.mean(rates)

        # compute the conditional fidelity by first computing it for all K input-target pairs, then averaging
        Fcs = Fus / rates
        Fc = jnp.mean(Fcs)

        return Fu, Fc, rate


class MinimalQPNN(QPNN):
    """Class for experimental modelling of QPNNs based on single-site Kerr-like nonlinearities that may be turned on/off and varied in strength.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the symmetric Fock basis for $n$ photons and $m$ optical modes
        meshes (tuple): list of $L$ objects containing methods that allow each linear layer (i.e. rectangular Mach-Zehnder interferometer meshes) to be encoded
        ell_mzi (tuple): nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
        ell_ps (tuple): nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
        t_dc (tuple): directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
        transformer (SymmetricTransformer): object containing methods that compute multi-photon unitary transformations of the linear layers
        varphi (tuple): the first (second) element is the mean (standard deviation) of a normal distribution from which each effective nonlinear phase shift, $\\varphi$, is selected
        varphis (np.ndarray): $(L - 1)\\times m$ array that contains the effective nonlinear phase shifts, $\\varphi$, for each individual nonlinear element
        burnout_map (np.ndarray): $(L - 1)\\times m$ array, with either boolean or binary elements, specifying whether single-site nonlinearities are on/off for specific modes in each of the $L - 1$ nonlinear sections
        nls (jnp.ndarray): $(L - 1)\\times N\\times N$ array, the matrix representations of the sets of single-site Kerr-like nonlinearities resolved in the symmetric Fock basis for each of the $L - 1$ nonlinear sections
        K (int): number of input-target state pairs in the QPNN training set, defaults to 0 if none provided
        psi_in (jnp.ndarray): $K\\times N$ array containing the $K$ input states in the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        psi_targ (jnp.ndarray): $K\\times N$ array containing the $K$ target states in the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        comp_indices (jnp.ndarray): $2^n$-length array whose elements are the indices of the symmetric Fock basis where dual-rail encoded computational basis states lie
    """

    def __init__(
        self,
        n: int,
        m: int,
        L: int,
        varphi: tuple = (np.pi, 0.0),
        burnout_map: Optional[np.ndarray] = None,
        ell_mzi: tuple = (0.0, 0.0),
        ell_ps: tuple = (0.0, 0.0),
        t_dc: tuple = (0.5, 0.0),
        training_set: Optional[tuple] = None,
    ) -> None:
        """Initialization of a Minimal QPNN instance.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            varphi: nominal effective nonlinear phase shift, $\\varphi$, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nonlinear element is selected
            burnout_map: $(L - 1)\\times m$ array, with either boolean or binary elements, specifying whether single-site nonlinearities are on/off for specific modes in each of the $L - 1$ nonlinear sections
            ell_mzi: nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
            ell_ps: nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
            t_dc: directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
            training_set: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """

        super().__init__(n, m, L)

        # instantiate L Clements meshes, with losses and routing errors, that act as the pathway to encoding the linear layers
        self.ell_mzi = ell_mzi
        self.ell_ps = ell_ps
        self.t_dc = t_dc
        self.meshes = tuple([Mesh(m) for _ in range(L)])
        self.prep_meshes()

        # instantiate symmetric transfomer required for the multi-photon unitary transformations of the linear layers
        self.transformer = SymmetricTransformer(n, m)

        # generate the effective nonlinar phase shifts, store the burnout map, construct the corresponding nonlinear Kerr-like unitaries for each nonlinear section
        self.varphi = varphi
        self.varphis = np.random.normal(loc=varphi[0], scale=varphi[1], size=(L - 1, m))
        self.burnout_map = burnout_map
        nls = []
        for i in range(L - 1):
            bo_map = burnout_map[i] if burnout_map is not None else burnout_map
            nls.append(jnp.asarray(build_kerr(n, m, self.varphis[i], burnout_map=bo_map)))
        self.nls = jnp.array(nls, dtype=complex)

        # prepare the training set attributes whether one was provided or not
        self.training_set = training_set if training_set is not None else (jnp.array(()), jnp.array(()))

        # compute overhead for conditional fidelity and logical rate calculations
        self.comp_indices = jnp.asarray(comp_indices_from_symm_fock(build_symm_basis(n, m)))

    @property
    def training_set(self) -> tuple:
        """Training set of the QPNN.

        Returns:
            A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        return self.psi_in, self.psi_targ

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set of the QPNN.

        Args:
            tset: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        self.psi_in = jnp.asarray(tset[0])
        self.psi_targ = jnp.asarray(tset[1])
        self.K = 0 if self.psi_in.size == 0 else self.psi_in.shape[0]

    def prep_meshes(self) -> None:
        """Prepare the Mach-Zehnder interferometer meshes for the linear layers of the network."""

        # for each layer, compute and apply new loss and splitting ratio values from their respective distributions
        for i in range(self.L):
            ells_mzi = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_mzi[0]),
                self.ell_mzi[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_mzi[0]),
                self.m**2,
            ).reshape((self.m, self.m))
            ells_ps = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_ps[0]),
                self.ell_ps[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_ps[0]),
                self.m,
            )
            ts_dc = np.random.normal(self.t_dc[0], self.t_dc[1], self.m * (self.m - 1)).reshape((2, self.m * (self.m - 1) // 2))

            self.meshes[i].ell_mzi = jnp.asarray(ells_mzi)
            self.meshes[i].ell_ps = jnp.asarray(ells_ps)
            self.meshes[i].t_dc = jnp.asarray(ts_dc)

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Build a matrix representation of the QPNN from all its layers and components.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            $N\\times N$ array, the matrix representation of the QPNN resolved in the symmetric Fock basis
        """

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = jnp.array([self.meshes[i].encode(phi[i], theta[i], delta[i]) for i in range(self.L)], dtype=complex)

        # perform the multi-photon unitary transformations for each linear layer
        multi_photon_Us = vmap(self.transformer.transform)(single_photon_Us)

        # for each linear layer up to the last one, multiply the corresponding nonlinear unitary and multi-photon unitary together
        layers = vmap(lambda nl, PhiU: nl @ PhiU)(self.nls, multi_photon_Us[0 : self.L - 1])

        # stack the layers together, including the final linear layer
        layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, self.N, self.N))))

        # multiply all the layers together
        S: jnp.ndarray = reduce(jnp.matmul, layers[::-1])
        return S

    @partial(jit, static_argnums=(0,))
    def calc_unc_fidelity(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> DTypeLike:
        """Calculate the unconditional fidelity of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Unconditional fidelity of the QPNN
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function
        S = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.dot(S, psi))(self.psi_in)

        # compute the unconditional fidelity by first computing it for all K input-target pairs, then averaging
        Fus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ, psi_out)
        Fu = jnp.mean(Fus)

        return Fu

    @partial(jit, static_argnums=(0,))
    def calc_performance_measures(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Calculate the unconditional fidelity, conditional fidelity, and logical rate of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Unconditional fidelity, conditional fidelity, and logical rate of the QPNN, as a tuple
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function
        S = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.dot(S, psi))(self.psi_in)

        # compute the unconditional fidelity by first computing it for all K input-target pairs, then averaging
        Fus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ, psi_out)
        Fu = jnp.mean(Fus)

        # compute the logical rate by first computing it for all K input-target pairs, then averaging
        rates = vmap(lambda psi: jnp.sum(jnp.abs(psi[self.comp_indices]) ** 2))(psi_out)
        rate = jnp.mean(rates)

        # compute the conditional fidelity by first computing it for all K input-target pairs, then averaging
        Fcs = Fus / rates
        Fc = jnp.mean(Fcs)

        return Fu, Fc, rate


class JitterQPNN(QPNN):
    """Class for an QPNN based on single-site Kerr-like nonlinearities that operates on partially-distinguishable photons with finite wavepackets.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the asymmetric Fock basis for $n$ photons and $m$ optical modes
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer meshes) to be encoded
        ell_mzi (float): loss for the Mach-Zehnder interferometer in the time-bin-encoded mesh, as a decimal (i.e. percentage loss)
        ell_ps (float): loss for the output phase shifter in the time-bin-encoded mesh, as a decimal (i.e. percentage loss)
        ell_io (tuple): first element is the loss for the input switch for the time-bin-encoded mesh, as a decimal (i.e. percentage loss), while the second element is that for the output switch
        t_dc (tuple): directional coupler splitting ratios (T:R) for the two nominally 50:50 directional couplers that form the Mach-Zehnder interferometer, as decminal values
        loss_input (jnp.ndarray): $m\\times m$ array, a diagonal matrix that describes the loss in each mode from the input switch to a time-bin-encoded mesh
        loss_output (jnp.ndarray): $m\\times m$ array, a diagonal matrix that describes the loss in each mode from all passes through the MZI, the output phase shifter, and the output switch for a time-bin-encoded mesh
        varphi (float): effective nonlinear phase shift, $\\varphi$
        kerr (jnp.ndarray): $N\\times N$ array, the matrix representation of the set of single-site Kerr-like nonlinearities resolved in the asymmetric Fock basis
        t (jnp.ndarray): $N_t$-length array, the time domain on which the wavefunctions attached to the states in the QPNN training set are calculated, defaults to an empty array if none provided
        K (int): number of input-output state pairs in the QPNN training set, defaults to 0 if none provided
        psi_in (jnp.ndarray): $K\\times N\\times N_t\\times N_t$ array containing the $K$ input states in the QPNN training set, resolved in the $N$-dimensional asymmetric Fock basis, where each element is a $N_t\\times N_t$ wavefunction attached to the corresponding basis state, defaults to an empty array if none provided
        psi_targ (jnp.ndarray): $K\\times N\\times N_t\\times N_t$ array containing the $K$ target states in the QPNN training set, resolved in the $N$-dimensional asymmetric Fock basis, where each element is a $N_t\\times N_t$ wavefunction attached to the corresponding basis state, defaults to an empty array if none provided
        bins_targ (jnp.ndarray): $K\\times n!$ array containing the $n!$ indices of the asymmetric Fock basis, for each of the $K$ target states in the QPNN training set, where the photons are situated in the targeted time bins, defaults to an empty array if none provided
        comp_indices (jnp.ndarray): flattened $2^n\\times n!$ array where the $n!$ permutations in the asymmetric Fock basis that correspond to each of the $2^n$ computational basis states are catalogued, defaults to an empty array if none provided
    """

    def __init__(
        self,
        n: int,
        m: int,
        L: int,
        varphi: float = np.pi,
        ell_mzi: float = 0.0,
        ell_ps: float = 0.0,
        ell_io: tuple = (0.0, 0.0),
        t_dc: tuple = (0.5, 0.5),
        training_set: Optional[tuple] = None,
        comp_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Initialization of a Jitter QPNN instance.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            varphi: effective nonlinear phase shift, $\\varphi$
            ell_mzi: loss for the Mach-Zehnder interferometer in the time-bin-encoded mesh, as a decimal (i.e. percentage loss)
            ell_ps: loss for the output phase shifter in the time-bin-encoded mesh, as a decimal (i.e. percentage loss)
            ell_io: first element is the loss for the input switch for the time-bin-encoded mesh, as a decimal (i.e. percentage loss), while the second element is that for the output switch
            t_dc: directional coupler splitting ratios (T:R) for the two nominally 50:50 directional couplers that form the Mach-Zehnder interferometer, as decminal values
            training_set: a tuple including an $N_t$-length array that provides the time domain for the photon wavefunctions; two $K\\times N\\times N_t\\times N_t$ arrays, the first of which contains $K$ input states resolved in the asymmetric Fock basis,
                with attached wavefunctions, the second of which contains the corresponding target states; and one $K\\times n!$ array containing the $n!$ indices of the asymmetric Fock basis, for each of the $K$ target states in the QPNN training set, where
                the photons are situated in the targeted time bins
            comp_indices: $2^n\\times n!$ array where the $n!$ permutations in the asymmetric Fock basis that correspond to each of the $2^n$ computational basis states are catalogued
        """

        super().__init__(n, m, L, basis_type="asymmetric")

        # instantiate a Clements mesh to act as the pathway to encoding the linear layers, with imperfect directional coupler splitting ratios as specified
        self.mesh = Mesh(
            m,
            t_dc=np.vstack((t_dc[0] * np.ones(m * (m - 1) // 2, dtype=float), t_dc[1] * np.ones(m * (m - 1) // 2, dtype=float))),
        )

        # construct the loss matrices for the MZI, output phase shifter, and input/output switches used to realize a time-bin mesh
        loss_mzi = (np.sqrt(1.0 - ell_mzi) ** m) * np.ones(m, dtype=complex)
        loss_mzi[np.arange(0, m, 2)] *= np.sqrt(1.0 - ell_mzi)
        loss_mzi = np.diag(loss_mzi)
        loss_ps = np.diag(np.sqrt(1.0 - ell_ps) * np.ones(m, dtype=complex))
        loss_output_switch = np.diag(np.sqrt(1.0 - ell_io[1]) * np.ones(m, dtype=complex))
        self.loss_output = jnp.asarray(loss_output_switch @ loss_ps @ loss_mzi)
        self.loss_input = np.diag(np.sqrt(1.0 - ell_io[0]) * np.ones(m, dtype=complex))

        # store the provided loss and splitting ratio parameters
        self.ell_mzi = ell_mzi
        self.ell_ps = ell_ps
        self.ell_io = ell_io
        self.t_dc = t_dc

        # store the provided effective nonlinear phase shift, construct the corresponding nonlinear Kerr-like unitary
        self.varphi = varphi
        self.kerr = jnp.asarray(build_kerr(n, m, varphi, basis_type="asymmetric"))

        # prepare the training set attributes whether one was provided or not
        self.training_set = training_set if training_set is not None else (jnp.array(()), jnp.array(()), jnp.array(()), jnp.array(()))

        # compute overhead for conditional fidelity and logical rate calculations
        self.comp_indices = jnp.asarray(comp_indices).flatten() if comp_indices is not None else jnp.array(())

    @property
    def training_set(self) -> tuple:
        """Training set of the QPNN.

        Returns:
            A tuple including an $N_t$-length array that provides the time domain for the photon wavefunctions; two $K\\times N\\times N_t\\times N_t$ arrays, the first of which contains $K$ input states resolved in the asymmetric Fock basis,
                with attached wavefunctions, the second of which contains the corresponding target states; and one $K\\times n!$ array containing the $n!$ indices of the asymmetric Fock basis, for each of the $K$ target states in the QPNN training
                set, where the photons are situated in the targeted time bins
        """
        return self.t, self.psi_in, self.psi_targ, self.bins_targ

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set of the QPNN.

        Args:
            tset: a tuple including an $N_t$-length array that provides the time domain for the photon wavefunctions; two $K\\times N\\times N_t\\times N_t$ arrays, the first of which contains $K$ input states resolved in the asymmetric Fock basis,
                with attached wavefunctions, the second of which contains the corresponding target states; and one $K\\times n!$ array containing the $n!$ indices of the asymmetric Fock basis, for each of the $K$ target states in the QPNN training
                set, where the photons are situated in the targeted time bins
        """
        self.t = jnp.asarray(tset[0])
        self.psi_in = jnp.asarray(tset[1])
        self.psi_targ = jnp.asarray(tset[2])
        self.bins_targ = jnp.asarray(tset[3])
        self.K = 0 if self.psi_in.size == 0 else self.psi_in.shape[0]

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Build a matrix representation of the QPNN from all its layers and components.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            $N\\times N$ array, the matrix representation of the QPNN resolved in the symmetric Fock basis
        """

        @vmap
        def encode_with_loss(phi_l: jnp.ndarray, theta_l: jnp.ndarray, delta_l: jnp.ndarray) -> jnp.ndarray:
            """Wrapper for single-photon unitary encoding followed by the application of the loss model for time-bin-encoded meshes.

            Args:
                phi_l: $m(m-1)/2$ phase shifts, $\\phi$, for each MZI in the lth layer
                theta_l: $m(m-1)/2$ phase shifts, $\\theta$, for each MZI in the lth layer
                delta_l: $m$ phase shifts, $\\delta$, for each output phase shifter of the lth layer

            Returns:
                $m\\times m$ array, the matrix representation of the single-photon unitary for the lth layer of the QPNN
            """
            U = self.mesh.encode(phi_l, theta_l, delta_l)
            return self.loss_output @ U @ self.loss_input  # type: ignore

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = encode_with_loss(phi, theta, delta)

        @vmap
        def multi_transform(U: jnp.ndarray) -> jnp.ndarray:
            """Wrapper for multi-photon unitary transformation in the asymmetric Fock basis.

            Args:
                U: $m\\times m$ single-photon unitary, $\\mathbf{U}$, to transform

            Returns:
                $N\\times N$ multi-photon unitary, $\\boldsymbol{\\Phi(\\mathbf{U})}$, in the $N$-dimensional asymmetric Fock basis
            """
            return asymmetric_transform(U, self.n)  # type: ignore

        # perform the multi-photon unitary transformations for each linear layer
        multi_photon_Us = multi_transform(single_photon_Us)

        # for each linear layer up to the last one, multiply the nonlinear unitary and multi-photon unitary together
        layers = vmap(lambda PhiU: self.kerr @ PhiU)(multi_photon_Us[0 : self.L - 1])

        # stack the layers together, including the final linear layer
        layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, self.N, self.N))))

        # multiply all the layers together
        S: jnp.ndarray = reduce(jnp.matmul, layers[::-1])
        return S

    @partial(jit, static_argnums=(0,))
    def calc_unc_fidelities(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Calculate the unconditional wavepacket fidelity and unconditional bin fidelity of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Tuple including the unconditional wavepacket fidelity followed by the unconditional bin fidelity
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function
        S = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.tensordot(S, psi, axes=1))(self.psi_in)

        # compute the unconditional wavepacket fidelity by first computing it for all K input-target pairs, then averaging
        Fuws = vmap(lambda psit, psio: jnp.abs(jnp.sum(jnp.trapezoid(jnp.trapezoid(jnp.conj(psit) * psio, self.t), self.t))) ** 2)(
            self.psi_targ, psi_out
        )
        Fuw = jnp.mean(Fuws)

        @vmap
        def calc_Fubk(psio: jnp.ndarray, binst: jnp.ndarray) -> float:
            """Wrapper for computing the $k^\\mathrm{th}$ unconditional bin fidelity of the QPNN for the $k^\\mathrm{th}$ targeted time bins.

            Args:
                psio: $N\\times N_t\\times N_t$ array, the kth output state produced by applying the QPNN to the kth input state
                binst: $n$-length array containing indices for states in the asymmetric Fock basis where the photons are situated in the targeted time bins

            Returns:
                The $k^\\mathrm{th}$ unconditional bin fidelity of the QPNN for the $k^\\mathrm{th}$ targeted time bins
            """

            # compute the probabilities of the photons being situated in each of the targeted time bin states of the asymmetric Fock basis, then sum
            probs = vmap(lambda ind: jnp.trapezoid(jnp.trapezoid(jnp.abs(psio[ind]) ** 2, self.t), self.t))(binst)
            return jnp.sum(probs)  # type: ignore

        # compute the unconditional bin fidelity by summing the probabilities of measuring the output photons in the targeted time bins, for each input-target pair, then average
        Fubs = calc_Fubk(psi_out, self.bins_targ)
        Fub = jnp.mean(Fubs)

        return Fuw, Fub

    @partial(jit, static_argnums=(0,))
    def calc_performance_measures(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Calculate the unconditional and conditional wavepacket and bin fidelities, as well as the logical rate for the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Tuple including the unconditional wavepacket and bin fidelities, followed by the conditional wavepacket and bin fidelities, followed by the logical rate
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function
        S = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.tensordot(S, psi, axes=1))(self.psi_in)

        # compute the wavepacket fidelity by first computing it for all K input-target pairs, then averaging
        Fuws = vmap(lambda psit, psio: jnp.abs(jnp.sum(jnp.trapezoid(jnp.trapezoid(jnp.conj(psit) * psio, self.t), self.t))) ** 2)(
            self.psi_targ, psi_out
        )
        Fuw = jnp.mean(Fuws)

        @vmap
        def calc_Fubk(psio: jnp.ndarray, binst: jnp.ndarray) -> float:
            """Wrapper for computing the $k^\\mathrm{th}$ unconditional bin fidelity of the QPNN for the $k^\\mathrm{th}$ targeted time bins.

            Args:
                psio: $N\\times N_t\\times N_t$ array, the kth output state produced by applying the QPNN to the kth input state
                binst: $n$-length array containing indices for states in the asymmetric Fock basis where the photons are situated in the targeted time bins

            Returns:
                The $k^\\mathrm{th}$ unconditional bin fidelity of the QPNN for the $k^\\mathrm{th}$ targeted time bins
            """

            # compute the probabilities of the photons being situated in each of the targeted time bin states of the asymmetric Fock basis, then sum
            probs = vmap(lambda ind: jnp.trapezoid(jnp.trapezoid(jnp.abs(psio[ind]) ** 2, self.t), self.t))(binst)
            return jnp.sum(probs)  # type: ignore

        # compute the bin fidelity by summing the probabilities of measuring the output photons in the targeted time bins, for each input-target pair, then average
        Fubs = calc_Fubk(psi_out, self.bins_targ)
        Fub = jnp.mean(Fubs)

        @vmap
        def calc_rate(psio: jnp.ndarray) -> float:
            """Wrapper for computing the $k^\\mathrm{th}$ logical rate of the QPNN for the $k^\\mathrm{th}$ input-target pair.

            Args:
                psio: $N\\times N_t\\times N_t$ array, the kth output state produced by applying the QPNN to the kth input state

            Returns:
                The $k^\\mathrm{th}$ logical rate of the QPNN for the $k^\\mathrm{th}$ input-target pair
            """

            # compute the probabilities of the photons being situated in each of the computational basis states of the asymmetric Fock basis, then sum
            probs = vmap(lambda ind: jnp.trapezoid(jnp.trapezoid(jnp.abs(psio[ind]) ** 2, self.t), self.t))(self.comp_indices)
            return jnp.sum(probs)  # type: ignore

        # compute the logical rates for each input-target pair, then average
        rates = calc_rate(psi_out)
        rate = jnp.mean(rates)

        # compute the conditional wavepacket fidelities for each input-target pair, then average
        Fcws = Fuws / rates
        Fcw = jnp.mean(Fcws)

        # compute the conditional bin fidelities for each input-target pair, then average
        Fcbs = Fubs / rates
        Fcb = jnp.mean(Fcbs)

        return Fuw, Fub, Fcw, Fcb, rate


class TreeQPNN(QPNN):
    """Class for experimental modelling of QPNNs based on three-level system photon subtraction/addition nonlinearities that power a tree-type photonic cluster state generation protocol.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        b (int): number of branches in the tree, $b$
        N (int): dimension of the symmetric Fock basis for $n$ photons and $m$ optical modes
        meshes (list): list of $L$ objects containing methods that allow each linear layer (i.e. rectangular Mach-Zehnder interferometer meshes) to be encoded
        ell_mzi (tuple): nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
        ell_ps (tuple): nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
        t_dc (tuple): directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
        transformer (SymmetricTransformer): object containing methods that compute multi-photon unitary transformations of the linear layers
        varphi (tuple): tuple of the phase shifts applied to the subtracted photon, followed by that applied to the remaining photons, for the 3LS photon $\\mp$ nonlinearity, in $\\text{rad}$, $(\\varphi_1, \\varphi2)$
        nl (jnp.ndarray): $N\\times N$ array, the matrix representation of a set of single-site 3LS photon $\\mp$ nonlinearities resolved in the symmetric Fock basis
        K (int): number of input-target state pairs in the unit cell portion of the QPNN training set, defaults to 0 if none provided
        psi_in (jnp.ndarray): $K\\times N$ array containing the $K$ input states in the unit cell portion of the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        psi_targ (jnp.ndarray): $K\\times N$ array containing the $K$ target states in the unit cell portion of the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        K_route (int): number of input-target state pairs in the routing portion of the QPNN training set, defaults to 0 if none provided
        psi_in_route (jnp.ndarray): $K\\times m$ array containing the $K$ input states in the routing portion of the QPNN training set, resolved in the $m$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        psi_targ_route (jnp.ndarray): $K\\times m$ array containing the $K$ target states in the routing portion of the QPNN training set, resolved in the $m$-dimensional symmetric Fock basis, defaults to an empty array if none provided
        comp_indices (jnp.ndarray): $2^n$-length array whose elements are the indices of the symmetric Fock basis where dual-rail encoded computational basis states lie
    """

    def __init__(
        self,
        b: int,
        L: int,
        varphi: tuple = (0.0, np.pi),
        ell_mzi: tuple = (0.0, 0.0),
        ell_ps: tuple = (0.0, 0.0),
        t_dc: tuple = (0.5, 0.0),
        training_set: Optional[tuple] = None,
        training_set_route: Optional[tuple] = None,
    ) -> None:
        """Initialization of a Tree QPNN instance.

        Args:
            b: number of branches in the tree, $b$
            L: number of layers, $L$
            varphi: tuple of the phase shifts applied to the subtracted photon, followed by that applied to the remaining photons, for the 3LS photon $\\mp$ nonlinearity, in $\\text{rad}$, $(\\varphi_1, \\varphi2)$
            ell_mzi: nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
            ell_ps: nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
            t_dc: directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
            training_set: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states, for the unit cell functionality of the QPNN
            training_set_route: a tuple including two $K\\times m$ arrays, the first of which contains $K$ input states resolved in the $m$-dimensional symmetric Fock basis, the second of which contains the corresponding target states, for the routing functionality of the QPNN
        """

        n = b + 1
        m = 2 * n
        self.b = b
        super().__init__(n, m, L)

        # instantiate L Clements meshes, with losses and routing errors, that act as the pathway to encoding the linear layers
        self.ell_mzi = ell_mzi
        self.ell_ps = ell_ps
        self.t_dc = t_dc
        self.meshes: list[Mesh] = []
        self.prep_meshes()

        # instantiate symmetric transfomer required for the multi-photon unitary transformations of the linear layers
        self.transformer = SymmetricTransformer(n, m)

        # store the provided nonlinear phase shifts and construct the corresponding 3LS photon -/+ nonlinear unitary
        self.varphi = varphi
        self.nl = jnp.asarray(build_photon_mp(n, m, *varphi))

        # prepare the training set attributes whether they were provided or not
        self.training_set = training_set if training_set is not None else (jnp.array(()), jnp.array(()))
        self.training_set_route = training_set_route if training_set_route is not None else (jnp.array(()), jnp.array(()))

        # compute overhead for conditional fidelity and logical rate calculations
        self.comp_indices = jnp.asarray(comp_indices_from_symm_fock(build_symm_basis(n, m)))

    @property
    def training_set(self) -> tuple:
        """Training set for the unit cell generation functionality of the QPNN.

        Returns:
            A tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        return self.psi_in, self.psi_targ

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set for the unit cell generation functionality of the QPNN.

        Args:
            tset: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        self.psi_in = jnp.asarray(tset[0])
        self.psi_targ = jnp.asarray(tset[1])
        self.K = 0 if self.psi_in.size == 0 else self.psi_in.shape[0]

    @property
    def training_set_route(self) -> tuple:
        """Training set for the single-photon routing functionality of the QPNN.

        Returns:
            A tuple including two $K\\times m$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        return self.psi_in_route, self.psi_targ_route

    @training_set_route.setter
    def training_set_route(self, tset: tuple) -> None:
        """Training set for the single-photon routing functionality of the QPNN.

        Args:
            tset: a tuple including two $K\\times m$ arrays, the first of which contains $K$ input states resolved in the symmetric Fock basis, the second of which contains the corresponding target states
        """
        self.psi_in_route = jnp.asarray(tset[0])
        self.psi_targ_route = jnp.asarray(tset[1])
        self.K_route = 0 if self.psi_in_route.size == 0 else self.psi_in_route.shape[0]

    def prep_meshes(self) -> None:
        """Prepare the Mach-Zehnder interferometer meshes for the linear layers of the network."""

        # check if meshes have previously been instantiated
        new = len(self.meshes) == 0

        # for each layer, compute and apply new loss and splitting ratio values from their respective distributions
        for i in range(self.L):
            ells_mzi = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_mzi[0]),
                self.ell_mzi[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_mzi[0]),
                self.m**2,
            ).reshape((self.m, self.m))
            ells_ps = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_ps[0]),
                self.ell_ps[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_ps[0]),
                self.m,
            )
            ts_dc = np.random.normal(self.t_dc[0], self.t_dc[1], self.m * (self.m - 1)).reshape((2, self.m * (self.m - 1) // 2))

            if new:
                self.meshes.append(Mesh(self.m, ells_mzi, ells_ps, ts_dc))
            else:
                self.meshes[i].ell_mzi = jnp.asarray(ells_mzi)
                self.meshes[i].ell_ps = jnp.asarray(ells_ps)
                self.meshes[i].t_dc = jnp.asarray(ts_dc)

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Build matrix representations of the QPNN from all its layers and components, for operation on $n$ photons and one photon respectively.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            A tuple including an $N\\times N$ array, the matrix representation of the QPNN resolved in the $N$-dimensional symmetric Fock basis, and an $m\\times m$ array,
                the matrix representation of the QPNN resolved in the $m$-dimensional symmetric Fock basis
        """

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = jnp.array([self.meshes[i].encode(phi[i], theta[i], delta[i]) for i in range(self.L)], dtype=complex)

        # perform the multi-photon unitary transformations for each linear layer
        multi_photon_Us = vmap(self.transformer.transform)(single_photon_Us)

        # for each linear layer up to the last one, multiply the nonlinear unitary and multi-photon unitary together
        layers = vmap(lambda PhiU: self.nl @ PhiU)(multi_photon_Us[0 : self.L - 1])

        # stack the layers together, including the final linear layer
        layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, self.N, self.N))))

        # multiply all the layers together
        S: jnp.ndarray = reduce(jnp.matmul, layers[::-1])
        S_route: jnp.ndarray = reduce(jnp.matmul, single_photon_Us[::-1])
        return S, S_route

    @partial(jit, static_argnums=(0,))
    def calc_unc_fidelity(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> DTypeLike:
        """Calculate the full unconditional fidelity of the QPNN, including both the unit cell and routing functionalities.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Unconditional fidelity of the QPNN, including both the unit cell and routing functionalities
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function, both in the $N$-dimensional and $m$-dimensional symmetric Fock bases, for the unit cell and routing functionalities respectively
        S, S_route = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.dot(S, psi))(self.psi_in)
        psi_out_route = vmap(lambda psi: jnp.dot(S_route, psi))(self.psi_in_route)

        # compute the unconditional fidelity by first computing it for all K + K_route input-target pairs, then averaging
        Fus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ, psi_out)
        Fus_route = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ_route, psi_out_route)
        Fu = jnp.mean(jnp.hstack((Fus, Fus_route)))

        return Fu

    @partial(jit, static_argnums=(0,))
    def calc_performance_measures(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Calculate the unconditional fidelities, conditional fidelity, and logical rate of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Full unconditional fidelity, routing unconditional fidelity, unit cell unconditional fidelity, conditional fidelity, and logical rate of the QPNN, as a tuple
        """

        # check that a training set has been provided
        assert self.K > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function, both in the $N$-dimensional and $m$-dimensional symmetric Fock bases, for the unit cell and routing functionalities respectively
        S, S_route = self.build(phi, theta, delta)

        # apply the QPNN to the input states to produce the output states
        psi_out = vmap(lambda psi: jnp.dot(S, psi))(self.psi_in)
        psi_out_route = vmap(lambda psi: jnp.dot(S_route, psi))(self.psi_in_route)

        # compute the unconditional fidelity for the routing functionality by first computing it for all K_route input-target pairs, then averaging
        Furs = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ_route, psi_out_route)
        Fur = jnp.mean(Furs)

        # compute the unconditional fidelity for the unit cell functionality by first computing it for all K input-target pairs, then averaging
        Fuus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(self.psi_targ, psi_out)
        Fuu = jnp.mean(Fuus)

        # compute the full unconditional fidelity by averaging over all K + K_route input-target pairs
        Fu = jnp.mean(jnp.hstack((Fuus, Furs)))

        # compute the logical rate by first computing it for all K input-target pairs, then averaging
        rates = vmap(lambda psi: jnp.sum(jnp.abs(psi[self.comp_indices]) ** 2))(psi_out)
        rate = jnp.mean(rates)

        # compute the conditional fidelity by first computing it for all K input-target pairs, then averaging
        Fcs = Fuus / rates
        Fc = jnp.mean(Fcs)

        return Fu, Fur, Fuu, Fc, rate


class TreeQPNNExtended(QPNN):
    """Class for experimental modelling of QPNNs based on three-level system photon subtraction/addition nonlinearities that power a tree-type photonic cluster state generation protocol.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        b (int): number of branches in the tree, $b$
        N (int): dimension of the symmetric Fock basis for $n$ photons and $m$ optical modes
        Ns (tuple): tuple of $b + 1$ dimensions of the symmetric Fock basis for $n$ photons and $m$ optical modes for all $1 \\leq n \\leq b + 1$
        meshes (tuple): tuple of $L$ objects containing methods that allow each linear layer (i.e. rectangular Mach-Zehnder interferometer meshes) to be encoded
        ell_mzi (tuple): nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
        ell_ps (tuple): nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
        t_dc (tuple): directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
        transformers (tuple): tuple of $b + 1$ objects containing methods that compute multi-photon unitary transformations of the linear layers for all $1 \\leq n \\leq b + 1$
        varphi (tuple): tuple of the phase shifts applied to the subtracted photon, followed by that applied to the remaining photons, for the 3LS photon $\\mp$ nonlinearity, in $\\text{rad}$, $(\\varphi_1, \\varphi2)$
        nls (tuple): tuple of $b + 1$ $N\\times N$ arrays, the $b + 1$ matrix representations of a set of single-site 3LS photon $\\mp$ nonlinearities resolved in the symmetric Fock bases for all $1 \\leq n \\leq b + 1$
        K (tuple): the numbers of input-target state pairs in the QPNN training set for each $1 \\leq n \\leq b + 1$, defaults to a tuple of zeros if none provided
        psi_in (tuple): all $K\\times N$ arrays containing the $K$ input states of the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, for each $1 \\leq n \\leq b + 1$, defaults to a tuple of empty arrays if none provided
        psi_targ (tuple): all $K\\times N$ arrays containing the $K$ target states of the QPNN training set, resolved in the $N$-dimensional symmetric Fock basis, for each $1 \\leq n \\leq b + 1$, defaults to a tuple of empty arrays if none provided
        comp_indices (jnp.ndarray): $2^n$-length array whose elements are the indices of the symmetric Fock basis where dual-rail encoded computational basis states lie
    """

    def __init__(
        self,
        b: int,
        L: int,
        varphi: tuple = (0.0, np.pi),
        ell_mzi: tuple = (0.0, 0.0),
        ell_ps: tuple = (0.0, 0.0),
        t_dc: tuple = (0.5, 0.0),
        training_set: Optional[tuple] = None,
    ) -> None:
        """Initialization of a Tree QPNN instance.

        Args:
            b: number of branches in the tree, $b$
            L: number of layers, $L$
            varphi: tuple of the phase shifts applied to the subtracted photon, followed by that applied to the remaining photons, for the 3LS photon $\\mp$ nonlinearity, in $\\text{rad}$, $(\\varphi_1, \\varphi2)$
            ell_mzi: nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual interferometer is selected
            ell_ps: nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual output phase shifter is selected
            t_dc: directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50 coupler is selected
            training_set: tuple of two tuples of $K\\times N$ arrays, the first of which contains the $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states, for each $1 \\leq n \\leq b + 1$
        """

        n = b + 1
        m = 2 * n
        self.b = b
        super().__init__(n, m, L)

        # instantiate L Clements meshes, with losses and routing errors, that act as the pathway to encoding the linear layers
        self.ell_mzi = ell_mzi
        self.ell_ps = ell_ps
        self.t_dc = t_dc
        self.meshes = tuple([Mesh(m) for _ in range(L)])
        self.prep_meshes()

        # instantiate symmetric transfomers required for the multi-photon unitary transformations of the linear layers, for all 1 <= n <= b + 1
        transformers = []
        Ns = []
        for _n in range(1, n + 1):
            transformers.append(SymmetricTransformer(_n, m))
            Ns.append(transformers[-1].N)
        self.transformers = tuple(transformers)
        self.Ns = tuple(Ns)

        # store the provided nonlinear phase shifts and construct the corresponding 3LS photon -/+ nonlinear unitaries for all 1 <= n <= b + 1
        self.varphi = varphi
        nls = []
        for _n in range(1, n + 1):
            nls.append(jnp.asarray(build_photon_mp(_n, m, *varphi)))
        self.nls = tuple(nls)

        # prepare the training set attributes whether they were provided or not
        self.training_set = training_set if training_set is not None else ((), ())

        # compute overhead for conditional fidelity and logical rate calculations
        self.comp_indices = jnp.asarray(comp_indices_from_symm_fock(build_symm_basis(n, m)))

    @property
    def training_set(self) -> tuple:
        """Training set for the unit cell generation functionality of the QPNN.

        Returns:
            A tuple of two tuples of $K\\times N$ arrays, the first of which contains the $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states, for each $1 \\leq n \\leq b + 1$
        """
        return self.psi_in, self.psi_targ

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set for the unit cell generation functionality of the QPNN.

        Args:
            tset: tuple of two tuples of $K\\times N$ arrays, the first of which contains the $K$ input states resolved in the $N$-dimensional symmetric Fock basis, the second of which contains the corresponding target states, for each $1 \\leq n \\leq b + 1$
        """
        if len(tset[0]) == 0:
            psi_in = [jnp.array(())] * self.n
            psi_targ = [jnp.array(())] * self.n
            K = [0] * self.n
        else:
            psi_in = []
            psi_targ = []
            K = []
            for i in range(self.n):
                psi_in.append(jnp.asarray(tset[0][i]))
                psi_targ.append(jnp.asarray(tset[1][i]))
                K.append(psi_in[-1].shape[0])

        self.psi_in = tuple(psi_in)
        self.psi_targ = tuple(psi_targ)
        self.K = tuple(K)

    def prep_meshes(self) -> None:
        """Prepare the Mach-Zehnder interferometer meshes for the linear layers of the network."""

        # for each layer, compute and apply new loss and splitting ratio values from their respective distributions
        for i in range(self.L):
            ells_mzi = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_mzi[0]),
                self.ell_mzi[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_mzi[0]),
                self.m**2,
            ).reshape((self.m, self.m))
            ells_ps = np.random.normal(
                1.0 - 10 ** (-0.1 * self.ell_ps[0]),
                self.ell_ps[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_ps[0]),
                self.m,
            )
            ts_dc = np.random.normal(self.t_dc[0], self.t_dc[1], self.m * (self.m - 1)).reshape((2, self.m * (self.m - 1) // 2))

            self.meshes[i].ell_mzi = jnp.asarray(ells_mzi)
            self.meshes[i].ell_ps = jnp.asarray(ells_ps)
            self.meshes[i].t_dc = jnp.asarray(ts_dc)

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Build matrix representations of the QPNN from all its layers and components, for operation on $1 \\leq n \\leq b + 1$ photons.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            A tuple of $b + 1$ $N\\times N$ arrays, the matrix representations of the QPNN resolved in the $N$-dimensional symmetric Fock bases for all $1 \\leq n \\leq b + 1$
        """

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = jnp.array([self.meshes[i].encode(phi[i], theta[i], delta[i]) for i in range(self.L)], dtype=complex)

        def n_photon_S(transformer: SymmetricTransformer, nl: jnp.ndarray, N: int) -> jnp.ndarray:
            # perform the multi-photon unitary transformations for each linear layer
            multi_photon_Us = vmap(transformer.transform)(single_photon_Us)

            # for each linear layer up to the last one, multiply the nonlinear unitary and multi-photon unitary together
            layers = vmap(lambda PhiU: nl @ PhiU)(multi_photon_Us[0 : self.L - 1])

            # stack the layers together, including the final linear layer
            layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, N, N))))

            # multiply all the layers together
            Sn: jnp.ndarray = reduce(jnp.matmul, layers[::-1])
            return Sn

        # construct the matrix representations for all numbers of photons, 1 <= n <= b + 1
        S: tuple = tree_map(n_photon_S, self.transformers, self.nls, self.Ns)

        return S

    @partial(jit, static_argnums=(0,))
    def calc_unc_fidelity(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> DTypeLike:
        """Calculate the full unconditional fidelity of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Unconditional fidelity of the QPNN
        """

        # check that a training set has been provided
        assert self.K[0] > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function in all $N$-dimensional symmetric Fock bases for all 1 <= n <= b + 1
        S = self.build(phi, theta, delta)

        def n_photon_Fus(Sn: jnp.ndarray, psi_in_n: jnp.ndarray, psi_targ_n: jnp.ndarray) -> jnp.ndarray:
            psi_out_n = vmap(lambda psi: Sn @ psi)(psi_in_n)
            Fus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(psi_targ_n, psi_out_n)
            return Fus

        # compute the unconditional fidelities for each 1 <= n <= b + 1
        Fus = tree_map(n_photon_Fus, S, self.psi_in, self.psi_targ)

        # put everything together and take the mean
        Fu = jnp.mean(jnp.hstack(Fus))

        return Fu

    @partial(jit, static_argnums=(0,))
    def calc_performance_measures(self, phi: jnp.ndarray, theta: jnp.ndarray, delta: jnp.ndarray) -> tuple:
        """Calculate the unconditional fidelities, conditional fidelity, and logical rate of the QPNN.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output of the mesh in the ith layer

        Returns:
            Full unconditional fidelity, n = b + 1 unconditional fidelity, conditional fidelity, and logical rate of the QPNN, as a tuple
        """

        # check that a training set has been provided
        assert self.K[0] > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function in all $N$-dimensional symmetric Fock bases for all 1 <= n <= b + 1
        S = self.build(phi, theta, delta)

        def n_photon_Fus(Sn: jnp.ndarray, psi_in_n: jnp.ndarray, psi_targ_n: jnp.ndarray) -> jnp.ndarray:
            psi_out_n = vmap(lambda psi: Sn @ psi)(psi_in_n)
            Fus = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(psi_targ_n, psi_out_n)
            return Fus

        # compute the unconditional fidelities for each 1 <= n <= b + 1
        Fus_full = tree_map(n_photon_Fus, S, self.psi_in, self.psi_targ)

        # put everything together and take the mean
        Fu_full = jnp.mean(jnp.hstack(Fus_full))

        # compute the unconditional fidelity specifically for n = b + 1
        Fus = jnp.array(Fus_full[-1], dtype=float)
        Fu = jnp.mean(Fus)

        # compute the logical rate for n = b + 1 by first computing it for all K input-target pairs, then averaging
        psi_out = vmap(lambda psi: S[-1] @ psi)(self.psi_in[-1])
        rates = vmap(lambda psi: jnp.sum(jnp.abs(psi[self.comp_indices]) ** 2))(psi_out)
        rate = jnp.mean(rates)

        # compute the conditional fidelity by first computing it for all K input-target pairs, then averaging
        Fcs = Fus / rates
        Fc = jnp.mean(Fcs)

        return Fu_full, Fu, Fc, rate
