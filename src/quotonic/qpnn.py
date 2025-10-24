"""
The `quotonic.qpnn` module includes classes that contain different models of quantum photonic neural networks (QPNNs),
each designed to explore different capabilities in detail. `QPNN` serves as a template for these models and thus
includes attributes that are relevant to all. The other classes inherit `QPNN` and build from it, yet remain
organized similarly to each other in many ways, as will be discussed further below.

QPNNs are brain-inspired, nonlinear photonic circuits that have been predicted to near-deterministically (i.e. with
near 100% success rates) generate, and process, quantum entanglement. Driven with light, these networks leverage the
multiplexing, low latency, and ultra-low operational powers of mature photonic integrated circuits similarly to their
classical counterparts. By adding optical nonlinearities to the processing toolkit, QPNNs feature components that not
only induce the necessary photon-photon interactions for efficient processing, but also act analogous to the
activation (i.e. learning) function of conventional neural networks, allowing QPNNs to be trained to perform
specified input-output mappings between quantum photonic states. Building on this operating principle, QPNNs have
been considered for conducting quantum simulation, accelerating quantum state tomography, and even speeding up more
common machine learning tasks like image recognition or natural language processing.

As displayed in the exemplary four-mode, two-layer network shown below, each QPNN is constructed from $m$ optical
modes and $L$ layers, where each layer is realized by a linear, rectangular $m\\times m$ Mach-Zehnder inteferometer
(MZI) mesh. By selecting the two controllable phase shifters $(\\phi, \\theta)$ in each MZI, each layer can be
programmed to perform any arbitrary linear unitary transformation $\\mathbf{U}$ on the optical modes of the photons
(see [clements](clements.md) for more details). Single-site few-photon optical nonlinearities $\\Sigma(\\varphi)$,
of effective nonlinear phase shift $\\varphi$ (ideally $\\pi$, see [nl](nl.md) for more details), are placed between
consecutive layers. These elements are key to network operation as they provide the learning capabilities to the
neural network and allow it to realize near-deterministic entangling operations.

<p align="center">
<img width="800" src="..img/light/qpnn.png" class="only-light">
<img width="800" src="..img/dark/qpnn.png" class="only-dark">
</p>

By simply piecing all the sections of the network together, a QPNN can be described by the transfer function,

$$ \\mathbf{S} = \\mathbf{U}(\\boldsymbol{\\phi}_L, \\boldsymbol{\\theta}_L) \\cdot \\prod_{i = 1}^{L - 1} \\boldsymbol{
\\Sigma}(\\varphi) \\cdot \\mathbf{U}(\\boldsymbol{\\phi}_i, \\boldsymbol{\\theta}_i), $$

where each $\\boldsymbol{\\phi}_L, \\boldsymbol{\\theta}_L$ are vectors that contain all the $\\phi, \\theta$ phase
shifts in each MZI for the $i^\\text{th}$ layer. This transfer function will act on the $k^\\text{th}$ input state
$\\left|\\mathrm{in}\\right\\rangle_k$ to produce an output state $\\left|\\mathrm{out}\\right\\rangle_k = \\mathbf{
S}\\left|\\mathrm{in}\\right\\rangle_k$. Comparing the output with the target state $\\left|\\mathrm{
targ}\\right\\rangle_k$, (i.e. according to a truth table), the unconditional fidelity (or equivalently the success
rate) for the $k^\\text{th}$ input-target pair is given by

$$ \\mathcal{F}_k^{(\\mathrm{unc})} = \\left|{}_k\\!\\left\\langle\\mathrm{targ}\\right|\\mathbf{S}\\left|\\mathrm{
in}\\right\\rangle\\!{}_k\\right|^2, $$

which describes the chance that the network produces the targeted output state for any given input state without
conditions. To train the QPNN, an optimization algorithm maximizes the unconditional fidelity (or equivalently
minimizes the cost/network error $\\mathcal{C}^{(\\mathrm{unc})} = 1 - \\mathcal{F}^{(\\mathrm{unc})}$) using the
variational phase shift parameters from its $L$ layers (see [trainer](trainer.md) for more details on training).

Alternatively, the success of a QPNN operation may be conditioned on the detection of a logical output, that is,
one where the photons are detected in a combination of output modes that corresponds to the qubit encoding scheme. In
the image above, dual-rail encoding for the photonic qubits is considered such that a logical output is one where a
single photon is detected in one of the upper two modes while the other is detected in one of the bottom two modes.
This measure is termed the conditional fidelity  (or equivalently just fidelity) $\\mathcal{F}_k^{(\\mathrm{con})}$,
each $k^\\text{th}$ term of which can be multiplied by the corresponding probability that the network produces a
logical output in the computational basis $\\mathcal{P}_k^{(\\mathrm{cb})}$ (or equivalently logical rate),

$$ \\mathcal{F}_k^{(\\mathrm{unc})} = \\mathcal{F}_k^{(\\mathrm{con})}\\mathcal{P}_k^{(\\mathrm{cb})}, $$

to retrieve the $k^\\text{th}$ term of the unconditional fidelity.

With this background in mind, the structure of the classes for each QPNN model can be described more clearly. Upon
instantiation, each model prepares all pieces of the network architecture, from the linear layers to the nonlinear
sections, though the specific way that these components are prepared may differ slightly between models. Each class
has a property called `training_set` that can be used to set and retrieve a training set for an instantiated QPNN.
They also include a method called `build` to construct the system function $\\mathbf{S}$ in full. With the system
function and a training set, the aforementioned performance measures can be calculated, and these calculations tend
to make up the remainder of the class methods. All of these properties and methods differ slightly between models
which is why they are not written into the base class `QPNN` itself. If you decide to use `quotonic` to perform
research on QPNNs, feel free to develop your own model and a corresponding class to go with it. That is essentially
the best way to use this part of the package from a research standpoint. Also, we'd be happy to add it if it fits the
format appropriately, so please reach out!
"""

from functools import partial, reduce

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.tree import map as tree_map
from jax.tree import structure as tree_structure
from jax.tree import transpose as tree_transpose
from jax.typing import DTypeLike

from quotonic.aa import SecqTransformer
from quotonic.clements import Mesh
from quotonic.fock import build_secq_basis, calc_firq_dim, calc_secq_dim
from quotonic.nl import build_kerr, build_photon_mp
from quotonic.types import jnp_ndarray, np_ndarray
from quotonic.utils import comp_indices_from_secq

DEFAULT = None


class QPNN:
    """Base class for a quantum photonic neural network (QPNN).

    This is effectively a template that prepares the most fundamental attributes for any QPNN. Each QPNN is designed to
    operate on a certain number of photons, $n$ with a certain number of optical modes $m$, and features $L$ layers.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the relevant Fock basis for $n$ photons and $m$ optical modes
    """

    def __init__(self, n: int, m: int, L: int, basis_type: str = "secq") -> None:
        """Initialization of a QPNN instance.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            basis_type: specifies whether the QPNN is resolved in the first or second-quantized basis
        """

        # check that basis_type is valid
        assert (basis_type == "secq") or (basis_type == "firq"), "Basis type must be 'secq' or 'firq'"

        # store the provided properties of the QPNN, compute others
        self.n = n
        self.m = m
        self.L = L
        self.N = calc_secq_dim(n, m) if basis_type == "secq" else calc_firq_dim(n, m)


class IdealQPNN(QPNN):
    """Class for an idealized QPNN based on single-site Kerr-like nonlinearities.

    Here, the QPNN is modelled as it was originally proposed in [G. R. Steinbrecher *et al*., “Quantum optical
    neural networks”, *npj Quantum Inf* **5**, 60 (2019)](https://doi.org/10.1038/s41534-019-0174-7). Linear layers are
    spatial meshes of Mach-Zehnder interferometers, and the single-site nonlinearities are based on the optical Kerr
    effect. A provided truth table defines the training set.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the second quantization Fock basis for $n$ photons and $m$ optical modes
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer
            meshes) to be encoded
        transformer (SecqTransformer): object containing methods that compute multi-photon unitary transformations
            of the linear layers
        varphi (float): effective nonlinear phase shift, $\\varphi$
        kerr (jnp_ndarray): $N\\times N$ array, the matrix representation of the set of single-site Kerr-like
            nonlinearities resolved in the second quantization Fock basis
        K (int): number of input-target state pairs in the QPNN training set, defaults to 0 if none provided
        psi_in (jnp_ndarray): $K\\times N$ array containing the $K$ input states in the QPNN training set, resolved in
            the $N$-dimensional second quantization Fock basis, defaults to an empty array if none provided
        psi_targ (jnp_ndarray): $K\\times N$ array containing the $K$ target states in the QPNN training set, resolved
            in the $N$-dimensional second quantization Fock basis, defaults to an empty array if none provided
    """

    def __init__(self, n: int, m: int, L: int, varphi: float = np.pi, training_set: tuple | None = None) -> None:
        """Initialization of an Ideal QPNN instance.

        Each piece of the QPNN architecture is instantiated and stored as an attribute alongside relevant parameters.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            varphi: effective nonlinear phase shift, $\\varphi$
            training_set: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states
                resolved in the second quantization Fock basis, the second of which contains the corresponding
                target states
        """

        super().__init__(n, m, L)

        # instantiate a Clements mesh to act as the pathway to encoding the linear layers
        self.mesh = Mesh(m)

        # instantiate transfomer required for the multi-photon unitary transformations of the linear layers
        self.transformer = SecqTransformer(n, m)

        # store the provided effective nonlinear phase shift, construct the corresponding nonlinear Kerr-like unitary
        self.varphi = varphi
        self.kerr = jnp.asarray(build_kerr(n, m, varphi))

        # prepare the training set attributes whether one was provided or not
        self.training_set = training_set if training_set is not None else (jnp.array(()), jnp.array(()))

    @property
    def training_set(self) -> tuple[np_ndarray, np_ndarray]:
        """Training set of the QPNN.

        Returns:
            psi_in: $K\\times N$ array containing the $K$ input states resolved in the second quantization Fock basis
            psi_targ: $K\\times N$ array containing the $K$ target states resolved in the second quantization Fock basis
        """
        return np.asarray(self.psi_in), np.asarray(self.psi_targ)

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set of the QPNN.

        Args:
            tset: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in
                the second quantization Fock basis, the second of which contains the corresponding target states
        """
        self.psi_in = jnp.asarray(tset[0])
        self.psi_targ = jnp.asarray(tset[1])
        self.K = 0 if self.psi_in.size == 0 else self.psi_in.shape[0]

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> jnp_ndarray:
        """Build a matrix representation of the QPNN from all its layers and components.

        This method calculates the system function of the QPNN as introduced at the top of this module.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            S: $N\\times N$ array, the matrix representation of the QPNN resolved in the second quantization Fock basis
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
        S: jnp_ndarray = reduce(jnp.matmul, layers[::-1])
        return S

    @partial(jit, static_argnums=(0,))
    def calc_fidelity(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> DTypeLike:
        """Calculate the fidelity of the QPNN.

        This method calculates the fidelity of the QPNN as introduced at the top of this module. In this idealized
        model, the logical rate is always unity. Therefore, the unconditional and conditional fidelities are
        equivalent. This method relies on a training set and will thus throw an error if one has not been provided.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            F: fidelity of the QPNN
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

    Here, we add relevant experimental imperfections to the QPNN model including non-uniform photon loss and
    imbalanced routing caused by non-ideal directional coupler splitting ratios. This model corresponds to that
    presented in [J. Ewaniuk *et al*., “Imperfect Quantum Photonic Neural Networks”, *Adv Quantum Technol.* **6**,
    2200125 (2023)](https://doi.org/10.1002/qute.202200125). As in `IdealQPNN`, a provided truth table defines the
    training set.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        N (int): dimension of the second quantization Fock basis for $n$ photons and $m$ optical modes
        meshes (tuple): tuple of $L$ objects containing methods that allow each linear layer (i.e. rectangular
            Mach-Zehnder interferometer meshes) to be encoded
        ell_mzi (tuple): nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the
            mean (standard deviation) of a normal distribution from which those for each individual interferometer is
            selected
        ell_ps (tuple): nominal loss for a phase shifter in dB, where the first (second) element is the mean
            (standard deviation) of a normal distribution from which those for each individual output phase shifter
            is selected
        t_dc (tuple): directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is
            the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50
            coupler is selected
        transformer (SecqTransformer): object containing methods that compute multi-photon unitary transformations of
            the linear layers
        varphi (float): effective nonlinear phase shift, $\\varphi$
        nl (jnp_ndarray): $N\\times N$ array, the matrix representation of a set of single-site Kerr-like nonlinearities
            resolved in the second quantization Fock basis
        K (int): number of input-target state pairs in the QPNN training set, defaults to 0 if none provided
        psi_in (jnp_ndarray): $K\\times N$ array containing the $K$ input states in the QPNN training set, resolved in
            the $N$-dimensional second quantization Fock basis, defaults to an empty array if none provided
        psi_targ (jnp_ndarray): $K\\times N$ array containing the $K$ target states in the QPNN training set, resolved
            in the $N$-dimensional second quantization Fock basis, defaults to an empty array if none provided
        comp_indices (jnp_ndarray): $2^n$-length array whose elements are the indices of the second quantization Fock
            basis where dual-rail encoded computational basis states lie
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
        training_set: tuple | None = None,
    ) -> None:
        """Initialization of an Imperfect QPNN instance.

        Each piece of the QPNN architecture is instantiated and stored as an attribute alongside relevant parameters.

        Args:
            n: number of photons, $n$
            m: number of optical modes, $m$
            L: number of layers, $L$
            varphi: effective nonlinear phase shift, $\\varphi$
            ell_mzi: nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean
                (standard deviation) of a normal distribution from which those for each individual interferometer is
                selected
            ell_ps: nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard
                deviation) of a normal distribution from which those for each individual output phase shifter is
                selected
            t_dc: directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the
                mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50
                coupler is selected
            training_set: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states
                resolved in the second quantization Fock basis, the second of which contains the corresponding target
                states
        """

        super().__init__(n, m, L)

        # instantiate L Clements meshes, with losses and routing errors, for encoding the linear layers
        self.ell_mzi = ell_mzi
        self.ell_ps = ell_ps
        self.t_dc = t_dc
        self.meshes = tuple([Mesh(m) for _ in range(L)])
        self.imperfections = DEFAULT

        # instantiate transfomer required for the multi-photon unitary transformations of the linear layers
        self.transformer = SecqTransformer(n, m)

        # store the provided effective nonlinear phase shift, construct the corresponding nonlinear Kerr-like unitary
        self.varphi = varphi
        self.nl = jnp.asarray(build_kerr(n, m, varphi))

        # prepare the training set attributes whether one was provided or not
        self.training_set = training_set if training_set is not None else (jnp.array(()), jnp.array(()))

        # compute overhead for conditional fidelity and logical rate calculations
        self.comp_indices = jnp.asarray(comp_indices_from_secq(build_secq_basis(n, m)))

    @property
    def training_set(self) -> tuple[np_ndarray, np_ndarray]:
        """Training set of the QPNN.

        Returns:
            psi_in: $K\\times N$ array containing the $K$ input states resolved in the second quantization Fock basis
            psi_targ: $K\\times N$ array containing the $K$ target states resolved in the second quantization Fock basis
        """
        return np.asarray(self.psi_in), np.asarray(self.psi_targ)

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set of the QPNN.

        Args:
            tset: a tuple including two $K\\times N$ arrays, the first of which contains $K$ input states resolved in
                the second quantization Fock basis, the second of which contains the corresponding target states
        """
        self.psi_in = jnp.asarray(tset[0])
        self.psi_targ = jnp.asarray(tset[1])
        self.K = 0 if self.psi_in.size == 0 else self.psi_in.shape[0]

    @property
    def imperfections(self) -> tuple[np_ndarray, np_ndarray, np_ndarray]:
        """Component-level imperfection values for each interferometer mesh in the QPNN.

        Realistic MZI meshes are modelled by allowing each individual interferometer to suffer unique imperfections,
        resulting in unbalanced, photon-path-dependent errors. The transmittance of each directional coupler is
        randomly selected from a normal distribution with a mean and a standard deviation as defined by attribute
        `t_dc`. Similarly, the photon losses introduced by each MZI and each output phase shifter in each mesh are
        selected from normal distributions as well, with means and standard deviations defined by attributes
        `ell_mzi` and `ell_ps`, respectively. The transmittance attribute is a decimal value between 0 and 1 (e.g.
        0.5 corresponds to 50% transmission). Conversely, the loss attributes are provided as a positive dB value,
        and thus must be converted to a decimal value between 0 and 1 that defines the fraction of light lost. For
        some positive-valued dB loss $\\ell_\\mathrm{dB}$, this fraction is given by

        $$ \\ell = 1 - 10^{-\\ell_\\mathrm{dB}/10}. $$

        From the standard deviation of the dB loss $\\sigma_\\mathrm{dB}$, the standard deviation of the fractional
        loss is given by

        $$ \\sigma = \\frac{1}{10}\\sigma_\\mathrm{dB}\\ln{(10)}\\cdot 10^{-\\ell_\\mathrm{dB}/10}. $$

        When setting this property, you can provide a tuple with elements including arrays of the MZI losses,
        phase shifter losses, and directional coupler transmissivities, respectively (all as decimal values).
        Alternatively, you can pass `quotonic.qpnn.DEFAULT = None` to instruct the function to sample imperfection
        values from the distributions defined by the internal attributes.

        Returns:
            ells_mzi: $L\\times m\\times m$ array containing the fractinal loss per arm of each of the $L$
                interferometer meshes, for each column of MZIs respectively
            ells_ps: $L\\times m$ array containing the fractional loss for each of the output phase shifters
                in each of the $L$ interferometer meshes
            ts_dc: $L\\times 2\\times m(m-1)/2$ array containing the splitting ratio (T:R) of each directional
                coupler in each of the $L$ interferometer meshes, organized such that each column corresponds to one
                MZI, the top row being the first directional coupler and the bottom being the second, where the MZIs
                are ordered from top to bottom followed by left to right across each mesh
        """
        ells_mzi = np.zeros((self.L, self.m, self.m), dtype=float)
        ells_ps = np.zeros((self.L, self.m), dtype=float)
        ts_dc = np.zeros((self.L, 2, self.m * (self.m - 1) // 2), dtype=float)
        for i in range(self.L):
            ells_mzi[i] = self.meshes[i].ell_mzi
            ells_ps[i] = self.meshes[i].ell_ps
            ts_dc[i] = self.meshes[i].t_dc
        return ells_mzi, ells_ps, ts_dc

    @imperfections.setter
    def imperfections(self, imp: tuple | None) -> None:
        """Component-level imperfection values for each interferometer mesh in the QPNN.

        Args:
            imp: Tuple of arrays, the first of which is an $L\\times m\\times m$ array containing the
                percentage loss per arm of each of the $L$ interferometer meshes, for each column of MZIs
                respectively; the second of which is an $L\\times m$ array containing the percentage loss for each of
                the output phase shifters in each of the $L$ interferometer meshes; the third of which is an
                $L\\times 2\\times m(m-1)/2$ array containing the splitting ratio (T:R) of each directional coupler
                in each of the $L$ interferometer meshes, organized such that each column corresponds to one MZI,
                the top row being the first directional coupler and the bottom being the second, where the MZIs are
                ordered from top to bottom followed by left to right across each mesh; if None, then this function
                will use the nominal imperfection attributes to generate the component-level imperfection values
        """
        if imp is None:
            # for each layer, compute and apply new loss and splitting ratio values from their respective distributions
            ells_mzi = np.zeros((self.L, self.m, self.m), dtype=float)
            ells_ps = np.zeros((self.L, self.m), dtype=float)
            ts_dc = np.zeros((self.L, 2, self.m * (self.m - 1) // 2), dtype=float)
            for i in range(self.L):
                ells_mzi[i] = np.random.normal(
                    1.0 - 10 ** (-0.1 * self.ell_mzi[0]),
                    self.ell_mzi[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_mzi[0]),
                    self.m**2,
                ).reshape((self.m, self.m))
                ells_ps[i] = np.random.normal(
                    1.0 - 10 ** (-0.1 * self.ell_ps[0]),
                    self.ell_ps[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_ps[0]),
                    self.m,
                )
                ts_dc[i] = np.random.normal(self.t_dc[0], self.t_dc[1], self.m * (self.m - 1)).reshape(
                    (2, self.m * (self.m - 1) // 2)
                )
        else:
            ells_mzi, ells_ps, ts_dc = imp

        for i in range(self.L):
            self.meshes[i].ell_mzi = jnp.asarray(ells_mzi[i])
            self.meshes[i].ell_ps = jnp.asarray(ells_ps[i])
            self.meshes[i].t_dc = jnp.asarray(ts_dc[i])

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> jnp_ndarray:
        """Build a matrix representation of the QPNN from all its layers and components.

        This method calculates the system function of the QPNN as introduced at the top of this module.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            S: $N\\times N$ array, the matrix representation of the QPNN resolved in the second quantization Fock basis
        """

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = jnp.array(
            [self.meshes[i].encode(phi[i], theta[i], delta[i]) for i in range(self.L)], dtype=complex
        )

        # perform the multi-photon unitary transformations for each linear layer
        multi_photon_Us = vmap(self.transformer.transform)(single_photon_Us)

        # for each linear layer up to the last one, multiply the nonlinear unitary and multi-photon unitary together
        layers = vmap(lambda PhiU: self.nl @ PhiU)(multi_photon_Us[0 : self.L - 1])

        # stack the layers together, including the final linear layer
        layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, self.N, self.N))))

        # multiply all the layers together
        S: jnp_ndarray = reduce(jnp.matmul, layers[::-1])
        return S

    @partial(jit, static_argnums=(0,))
    def calc_unc_fidelity(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> DTypeLike:
        """Calculate the unconditional fidelity of the QPNN.

        This method calculates the unconditional fidelity of the QPNN as introduced at the top of this module. It
        relies on a training set and will thus throw an error if one has not been provided.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith
                layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the ith
                layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            Func: unconditional fidelity of the QPNN
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
    def calc_performance_measures(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray
    ) -> tuple[DTypeLike, DTypeLike, DTypeLike]:
        """Calculate the unconditional fidelity, conditional fidelity, and logical rate of the QPNN.

        This method calculates the unconditional fidelity, conditional fidelity, and logical rate of the QPNN as
        introduced at the top of this module. It relies on a training set and will thus throw an error if one has not
        been provided.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in
                the ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            Func: unconditional fidelity $\\mathcal{F}^{(\\mathrm{unc})} of the QPNN
            Fcon: conditinal fidelity $\\mathcal{F}^{(\\mathrm{con})} of the QPNN
            Pcb: logical rate $\\mathcal{P}^{(\\mathrm{cb})} of the QPNN
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


class TreeQPNN(QPNN):
    """Class for experimental modelling of QPNNs based on three-level system photon subtraction/addition nonlinearities
    that power a tree-type photonic cluster state generation protocol.

    Here, we extend upon the QPNN model further, now incorporating both imperfections as well as the cavity-assisted
    three-level system scattering nonlinearity introduced in [J. R. Basani *et al*., "Universal logical quantum
    photonic neural network processor via cavity-assisted interactions", *npj Quantum Inf* **11**, 142 (2025)](
    https://doi.org/10.1038/s41534-025-01096-9). Additionally, this model is specifically designed for compatibility
    with the tree-type photonic cluster state generation protocol outlined in [J. Ewaniuk *et al*., "Large-Scale
    Tree-Type Photonic Cluster State Generation with Recurrent Quantum Photonic Neural Networks", arXiv:2505.14628
    [quant-ph]](https://doi.org/10.48550/arXiv.2505.14628). As in `IdealQPNN`, a provided truth table defines the
    training set, however, this set is more involved than the previous models. When generating trees, the QPNN must
    perform photon-number-dependent operations, responding only to the input photons without active adjustment. As a
    result, there are multiple training subsets for different numbers of photons, and more specifically different
    tree unit cells that must be formed.

    Attributes:
        n (int): number of photons, $n$
        m (int): number of optical modes, $m$
        L (int): number of layers, $L$
        b (int): maximum number of branches in the tree, $b\\equiv\\max\\{\\vec{b}\\}$
        N (int): dimension of the second quantization Fock basis for $n$ photons and $m$ optical modes
        Ns (tuple): tuple of $b + 1$ dimensions of the second quantization Fock bases for $n$ photons and $m$ optical
            modes for all $1 \\leq n \\leq b + 1$
        meshes (tuple): tuple of $L$ objects containing methods that allow each linear layer (i.e. rectangular
            Mach-Zehnder interferometer meshes) to be encoded
        ell_mzi (tuple): nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the
            mean (standard deviation) of a normal distribution from which those for each individual interferometer is
            selected
        ell_ps (tuple): nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard
            deviation) of a normal distribution from which those for each individual output phase shifter is selected
        t_dc (tuple): directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is
            the mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50
            coupler is selected
        transformers (tuple): tuple of $b + 1$ objects containing methods that compute multi-photon unitary
            transformations of the linear layers for all $1 \\leq n \\leq b + 1$
        varphi (tuple): tuple of the phase shifts applied to the subtracted photon, followed by that applied to the
            remaining photons, for the 3LS photon $\\mp$ nonlinearity, in $\\text{rad}$, $(\\varphi_1, \\varphi2)$
        nls (tuple): tuple of $b + 1$ $N\\times N$ arrays, the $b + 1$ matrix representations of a set of single-site
            3LS photon $\\mp$ nonlinearities resolved in the Fock bases for all $1 \\leq n \\leq b + 1$
        K (tuple): arrays containing the numbers of input-target state pairs in the QPNN training set for each
            $1 \\leq n \\leq b + 1$, per unit cell operation, defaults to a tuple of zeros if none provided
        psi_in (tuple): arrays containing the input states of the QPNN training set, resolved in the $2^n$-dimensional
            computational bases, for each $1 \\leq n \\leq b + 1$, defaults to a tuple of empty arrays if none provided
        psi_targ (tuple): arrays containing the target states of the QPNN training set, resolved in the
            $2^n$-dimensional computational bases, for each $1 \\leq n \\leq b + 1$, defaults to a tuple of empty
            arrays if none provided
        comp_indices (tuple): arrays containing the indices of each second quantization Fock basis, for each
            $1 \\leq n \\leq b + 1$, that correspond to each possible unit cell operation, defaults to a tuple of empty
            arrays if none provided
    """

    def __init__(
        self,
        b: int,
        L: int,
        varphi: tuple = (0.0, np.pi),
        ell_mzi: tuple = (0.0, 0.0),
        ell_ps: tuple = (0.0, 0.0),
        t_dc: tuple = (0.5, 0.0),
        training_set: tuple | None = None,
    ) -> None:
        """Initialization of a Tree QPNN instance.

        Each piece of the QPNN architecture is instantiated and stored as an attribute alongside relevant parameters.

        Args:
            b: number of branches in the tree, $b$
            L: number of layers, $L$
            varphi: tuple of the phase shifts applied to the subtracted photon, followed by that applied to the
                remaining photons, for the 3LS photon $\\mp$ nonlinearity, in $\\text{rad}$, $(\\varphi_1, \\varphi2)$
            ell_mzi: nominal loss for a Mach-Zehnder interferometer in dB, where the first (second) element is the mean
                (standard deviation) of a normal distribution from which those for each individual interferometer is
                selected
            ell_ps: nominal loss for a phase shifter in dB, where the first (second) element is the mean (standard
                deviation) of a normal distribution from which those for each individual output phase shifter is
                selected
            t_dc: directional coupler splitting ratios (T:R) as decimal values, where the first (second) element is the
                mean (standard deviation) of a normal distribution from which those for each individual nominally 50:50
                coupler is selected
            training_set: tuple of three tuples, the first two of which are the input and target states resolved in
                the computational basis for each $1 \\leq n \\leq b + 1$, the last of which contains the
                computational basis indices for each unit cell operation that exists for each $n$
        """

        n = b + 1
        m = 2 * n
        self.b = b
        super().__init__(n, m, L)

        # instantiate L Clements meshes, with losses and routing errors, for encoding the linear layers
        self.ell_mzi = ell_mzi
        self.ell_ps = ell_ps
        self.t_dc = t_dc
        self.meshes = tuple([Mesh(m) for _ in range(L)])
        self.imperfections = DEFAULT

        # instantiate transfomers for the multi-photon unitary transformations of the layers, for all 1 <= n <= b + 1
        transformers = []
        Ns = []
        for _n in range(1, n + 1):
            transformers.append(SecqTransformer(_n, m))
            Ns.append(transformers[-1].N)
        self.transformers = tuple(transformers)
        self.Ns = tuple(Ns)

        # store nonlinear phase shifts, construct the 3LS photon -/+ nonlinear unitaries for all 1 <= n <= b + 1
        self.varphi = varphi
        nls = []
        for _n in range(1, n + 1):
            nls.append(jnp.asarray(build_photon_mp(_n, m, *varphi)))
        self.nls = tuple(nls)

        # prepare the training set attributes whether they were provided or not
        self.training_set = training_set if training_set is not None else ((), (), ())

    @property
    def training_set(self) -> tuple[tuple, tuple, tuple]:
        """Training set for the unit cell generation functionality of the QPNN.

        Returns:
            psi_in: tuple of the input states resolved in the computational basis for each $1 \\leq n \\leq b + 1$
            psi_targ: tuple of the target states resolved in the computational basis for each $1 \\leq n \\leq b + 1$
            comp_indices: computational basis indices for each unit cell operation that exists for each $n$
        """
        psi_in = []
        psi_targ = []
        comp_indices = []
        for i in range(self.n):
            psi_in.append(np.asarray(self.psi_in[i]))
            psi_targ.append(np.asarray(self.psi_targ[i]))
            comp_indices.append(np.asarray(self.comp_indices[i]))
        return tuple(psi_in), tuple(psi_targ), tuple(comp_indices)

    @training_set.setter
    def training_set(self, tset: tuple) -> None:
        """Training set for the unit cell generation functionality of the QPNN.

        Args:
            tset: tuple of three tuples, the first two of which are the input and target states resolved in the
                computational basis for each $1 \\leq n \\leq b + 1$, the last of which contains the computational
                basis indices for each unit cell operation that exists for each $n$
        """
        if len(tset[0]) == 0:
            psi_in = [jnp.array(())] * self.n
            psi_targ = [jnp.array(())] * self.n
            comp_indices = [jnp.array(())] * self.n
            K = [0] * self.n
        else:
            psi_in = []
            psi_targ = []
            comp_indices = []
            K = []
            for i in range(self.n):
                psi_in.append(jnp.asarray(tset[0][i]))
                psi_targ.append(jnp.asarray(tset[1][i]))
                comp_indices.append(jnp.asarray(tset[2][i]))
                K.append(psi_in[-1].shape[0])

        self.psi_in = tuple(psi_in)
        self.psi_targ = tuple(psi_targ)
        self.comp_indices = tuple(comp_indices)
        self.K = tuple(K)

    @property
    def imperfections(self) -> tuple:
        """Component-level imperfection values for each interferometer mesh in the QPNN.

        See `ImperfectQPNN.imperfections` for more details.

        Returns:
            ells_mzi: $L\\times m\\times m$ array containing the fractinal loss per arm of each of the $L$
                interferometer meshes, for each column of MZIs respectively
            ells_ps: $L\\times m$ array containing the fractional loss for each of the output phase shifters
                in each of the $L$ interferometer meshes
            ts_dc: $L\\times 2\\times m(m-1)/2$ array containing the splitting ratio (T:R) of each directional
                coupler in each of the $L$ interferometer meshes, organized such that each column corresponds to one
                MZI, the top row being the first directional coupler and the bottom being the second, where the MZIs
                are ordered from top to bottom followed by left to right across each mesh
        """
        ells_mzi = np.zeros((self.L, self.m, self.m), dtype=float)
        ells_ps = np.zeros((self.L, self.m), dtype=float)
        ts_dc = np.zeros((self.L, 2, self.m * (self.m - 1) // 2), dtype=float)
        for i in range(self.L):
            ells_mzi[i] = self.meshes[i].ell_mzi
            ells_ps[i] = self.meshes[i].ell_ps
            ts_dc[i] = self.meshes[i].t_dc
        return ells_mzi, ells_ps, ts_dc

    @imperfections.setter
    def imperfections(self, imp: tuple | None) -> None:
        """Component-level imperfection values for each interferometer mesh in the QPNN.

        Args:
            imp: Tuple of arrays, the first of which is an $L\\times m\\times m$ array containing the
                percentage loss per arm of each of the $L$ interferometer meshes, for each column of MZIs
                respectively; the second of which is an $L\\times m$ array containing the percentage loss for each of
                the output phase shifters in each of the $L$ interferometer meshes; the third of which is an
                $L\\times 2\\times m(m-1)/2$ array containing the splitting ratio (T:R) of each directional coupler
                in each of the $L$ interferometer meshes, organized such that each column corresponds to one MZI,
                the top row being the first directional coupler and the bottom being the second, where the MZIs are
                ordered from top to bottom followed by left to right across each mesh; if None, then this function
                will use the nominal imperfection attributes to generate the component-level imperfection values
        """
        if imp is None:
            # for each layer, compute and apply new loss and splitting ratio values from their respective distributions
            ells_mzi = np.zeros((self.L, self.m, self.m), dtype=float)
            ells_ps = np.zeros((self.L, self.m), dtype=float)
            ts_dc = np.zeros((self.L, 2, self.m * (self.m - 1) // 2), dtype=float)
            for i in range(self.L):
                ells_mzi[i] = np.random.normal(
                    1.0 - 10 ** (-0.1 * self.ell_mzi[0]),
                    self.ell_mzi[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_mzi[0]),
                    self.m**2,
                ).reshape((self.m, self.m))
                ells_ps[i] = np.random.normal(
                    1.0 - 10 ** (-0.1 * self.ell_ps[0]),
                    self.ell_ps[1] * 0.1 * np.log(10) * 10 ** (-0.1 * self.ell_ps[0]),
                    self.m,
                )
                ts_dc[i] = np.random.normal(self.t_dc[0], self.t_dc[1], self.m * (self.m - 1)).reshape(
                    (2, self.m * (self.m - 1) // 2)
                )
        else:
            ells_mzi, ells_ps, ts_dc = imp

        for i in range(self.L):
            self.meshes[i].ell_mzi = jnp.asarray(ells_mzi[i])
            self.meshes[i].ell_ps = jnp.asarray(ells_ps[i])
            self.meshes[i].t_dc = jnp.asarray(ts_dc[i])

    @partial(jit, static_argnums=(0,))
    def build(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> tuple:
        """Build matrix representations of the QPNN from all its layers and components, for operation on
        $1 \\leq n \\leq b + 1$ photons.

        This method calculates the system function of the QPNN as introduced at the top of this module, yet does so
        for each potential number of input photons, $1 \\leq n \\leq b + 1$.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            S: a tuple of $b + 1$ $N\\times N$ arrays, the matrix representations of the QPNN resolved in the
                $N$-dimensional second quantization Fock bases for all $1 \\leq n \\leq b + 1$
        """

        # encode the single-photon unitary matrices for each linear layer in the Clements configuration
        single_photon_Us = jnp.array(
            [self.meshes[i].encode(phi[i], theta[i], delta[i]) for i in range(self.L)], dtype=complex
        )

        def n_photon_S(transformer: SecqTransformer, nl: jnp_ndarray, N: int) -> jnp_ndarray:
            # perform the multi-photon unitary transformations for each linear layer
            multi_photon_Us = vmap(transformer.transform)(single_photon_Us)

            # for each linear layer up to the last one, multiply the nonlinear unitary and multi-photon unitary together
            layers = vmap(lambda PhiU: nl @ PhiU)(multi_photon_Us[0 : self.L - 1])

            # stack the layers together, including the final linear layer
            layers = jnp.vstack((layers, multi_photon_Us[-1].reshape((1, N, N))))

            # multiply all the layers together
            Sn: jnp_ndarray = reduce(jnp.matmul, layers[::-1])
            return Sn

        # construct the matrix representations for all numbers of photons, 1 <= n <= b + 1
        S: tuple = tree_map(n_photon_S, self.transformers, self.nls, self.Ns)

        return S

    @partial(jit, static_argnums=(0,))
    def calc_cost(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> DTypeLike:
        """Calculate the cost function for the QPNN.

        This method calculates the cost function of the QPNN as introduced at the top of this module. It relies on a
        training set and will thus throw an error if one has not been provided. Specifically, it includes all
        input-target pairs for all unit cell operations for all numbers of photons $1 \\leq n \\leq b + 1$,
        before averaging.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            C: cost (i.e. network error) of the QPNN
        """

        # check that a training set has been provided
        assert self.K[0] > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function in all $N$-dimensional Fock bases for all 1 <= n <= b + 1
        S = self.build(phi, theta, delta)

        def n_photon_succ_rates(
            Sn: jnp_ndarray, psi_in_n: jnp_ndarray, psi_targ_n: jnp_ndarray, comp_inds_n: jnp_ndarray
        ) -> jnp_ndarray:
            @vmap
            def n_photon_unit_cell_succ_rates(inds: jnp_ndarray) -> jnp_ndarray:
                psi_out_n = vmap(lambda psi: Sn[jnp.ix_(inds, inds)] @ psi)(psi_in_n)
                succ_uc = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(psi_targ_n, psi_out_n)
                return succ_uc

            succ_rates = n_photon_unit_cell_succ_rates(comp_inds_n)
            return jnp.hstack(succ_rates)

        # compute the success rates for each 1 <= n <= b + 1
        succ_rates = tree_map(n_photon_succ_rates, S, self.psi_in, self.psi_targ, self.comp_indices)

        # put everything together, take the mean, then calculate cost
        cost = 1 - jnp.mean(jnp.hstack(succ_rates))

        return cost

    @partial(jit, static_argnums=(0,))
    def calc_overall_performance_measures(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray
    ) -> tuple[DTypeLike, DTypeLike, DTypeLike]:
        """Calculate the overall fidelity, success rate and logical rate of the QPNN.

        This method calculates the fidelity, success rate, and logical rate of the QPNN as introduced at the top of
        this module. It relies on a training set and will thus throw an error if one has not been provided.
        Specifically, it includes all input-target pairs for all unit cell operations for all numbers of photons $1
        \\leq n \\leq b + 1$, before averaging.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            fid: overall fidelity of the QPNN
            succ_rate: overall success rate of the QPNN
            logi_rate: overall logical rate of the QPNN
        """

        # check that a training set has been provided
        assert self.K[0] > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function in all $N$-dimensional Fock bases for all 1 <= n <= b + 1
        S = self.build(phi, theta, delta)

        def measures(
            Sn: jnp_ndarray, psi_in_n: jnp_ndarray, psi_targ_n: jnp_ndarray, comp_inds_n: jnp_ndarray
        ) -> tuple:
            @vmap
            def measures_per_unit_cell(inds: jnp_ndarray) -> tuple:
                psi_out_n = vmap(lambda psi: Sn[jnp.ix_(inds, inds)] @ psi)(psi_in_n)
                succ = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(psi_targ_n, psi_out_n)
                logi = vmap(lambda psio: jnp.sum(jnp.abs(psio) ** 2))(psi_out_n)
                return succ, logi

            succ_rates, logi_rates = measures_per_unit_cell(comp_inds_n)
            fids = succ_rates / logi_rates
            return jnp.hstack(fids), jnp.hstack(succ_rates), jnp.hstack(logi_rates)

        # map through the different numbers of photons and unit cell operations, evaluating performance on the way
        meas = tree_map(measures, S, self.psi_in, self.psi_targ, self.comp_indices)
        meas_T = tree_transpose(
            outer_treedef=tree_structure(S),
            inner_treedef=tree_structure(meas[0]),
            pytree_to_transpose=meas,
        )
        fids, succ_rates, logi_rates = meas_T

        # compute the overall fidelity, success rate & logical rate, including all operations in the mean
        fid = jnp.mean(jnp.hstack(fids))
        succ_rate = jnp.mean(jnp.hstack(succ_rates))
        logi_rate = jnp.mean(jnp.hstack(logi_rates))

        return fid, succ_rate, logi_rate

    @partial(jit, static_argnums=(0,))
    def calc_unit_cell_performance_measures(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray
    ) -> tuple[tuple, tuple, tuple]:
        """Calculate the fidelities, success rates and logical rates of the QPNN for each individual unit cell
        operation required for tree formation.

        This method calculates the fidelity, success rate, and logical rate of the QPNN as introduced at the top of
        this module. It relies on a training set and will thus throw an error if one has not been provided.
        Specifically, each returned value includes only the input-target pairs for a specific unit cell operation.
        For each measure, a tuple is returned. The elements of this tuple are arrays, each for a specific number of
        photons in increasing order for all $1 \\leq n \\leq b + 1$. There may be multiple unit cell operations
        required for a given $n$, so these arrays may contain multiple values. The specific dimensions depend in a
        complicated way on $b$, which is why they are not provided here in general.

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer

        Returns:
            fids: tuple containing the fidelities of the QPNN for each unit cell operation and each $n$
            succ_rates: tuple containing the success rates of the QPNN for each unit cell operation and each $n$
            logi_rates: tuple containing the logical rates of the QPNN for each unit cell operation and each $n$
        """

        # check that a training set has been provided
        assert self.K[0] > 0, "No training set was provided for the QPNN."

        # construct the QPNN system function in all $N$-dimensional Fock bases for all 1 <= n <= b + 1
        S = self.build(phi, theta, delta)

        def measures(
            Sn: jnp_ndarray, psi_in_n: jnp_ndarray, psi_targ_n: jnp_ndarray, comp_inds_n: jnp_ndarray
        ) -> tuple:
            @vmap
            def measures_per_unit_cell(inds: jnp_ndarray) -> tuple:
                psi_out_n = vmap(lambda psi: Sn[jnp.ix_(inds, inds)] @ psi)(psi_in_n)
                succ_uc = vmap(lambda psit, psio: jnp.abs(jnp.dot(jnp.conj(psit), psio)) ** 2)(psi_targ_n, psi_out_n)
                logi_uc = vmap(lambda psio: jnp.sum(jnp.abs(psio) ** 2))(psi_out_n)
                return succ_uc, logi_uc

            succ_rates, logi_rates = measures_per_unit_cell(comp_inds_n)
            succ_rate = vmap(lambda succ: jnp.mean(succ))(succ_rates)
            logi_rate = vmap(lambda logi: jnp.mean(logi))(logi_rates)
            fid = vmap(lambda succ, logi: jnp.mean(succ / logi))(succ_rates, logi_rates)
            return fid, succ_rate, logi_rate

        # map through the different numbers of photons and unit cell operations, evaluating performance on the way
        meas = tree_map(measures, S, self.psi_in, self.psi_targ, self.comp_indices)
        meas_T = tree_transpose(
            outer_treedef=tree_structure(S),
            inner_treedef=tree_structure(meas[0]),
            pytree_to_transpose=meas,
        )
        fids, succ_rates, logi_rates = meas_T

        return fids, succ_rates, logi_rates
