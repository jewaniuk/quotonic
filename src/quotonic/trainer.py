"""
The `quotonic.trainer` module includes classes that contain methods required to train the respective quantum photonic
neural network (QPNN) models defined in [qpnn](qpnn.md). `Trainer` serves as a template class and thus includes
attributes and methods that are relevant to training any model. The other trainer classes inherit `Trainer` and build
from it, yet remain organized similarly to each other in many ways, as will be discussed further below.

When training a QPNN, the goal is to minimize the network cost function by adjusting the variational phase shift
parameters throughout the network architecture. Since most functionalities in `quotonic` are written for use with
`jax`, we naturally turn to the version of `autograd` native to jax for gradient computation, and manage optimization
trials using `optax`. Currently, we typically apply the default [Adam optimizer](https://arxiv.org/abs/1412.6980) and
exponential decay scheduler from `optax`, as inspired by [Cascaded Optical Systems Approach to Neural Networks
(CasOptAx)]( https://github.com/JasvithBasani/CasOptAx) as originally designed for use in [J. R. Basani *et al*.,
"Universal logical quantum photonic neural network processor via cavity-assisted interactions", *npj Quantum Inf*
**11**, 142 (2025)](https://doi.org/10.1038/s41534-025-01096-9).

When performing a QPNN training simulation, we typically attempt to train network models in a set number of
optimization trials, each proceeding for a set number of training epochs. Each optimization trial
begins by  selecting random linear unitary transformations from the Haar measure, for each layer, and performing
Clements  decomposition (see [clements](clements.md)) to extract the initial phase shift parameters. This has been
shown in [S. Pai *et al*., "Matrix Optimization on Universal Unitary Photonic Devices", *Phys. Rev. Appl.* **11**,
064044 (2019)](https://doi.org/10.1103/PhysRevApplied.11.064044) to improve convergence speed. From the initial
parameters, the cost function can be evaluated, its gradients computed, and the results used to iteratively update
the parameters each epoch (see `update` method in each class). At the end, the results of each trial are saved to a
dictionary which is returned upon the completion of all trials (see `train` method in each class).

If you decide to use `quotonic` to perform research on QPNNs, feel free to develop a trainer class to go with your
QPNN model. Also, we'd be happy to add it if it fits the format appropriately, so please reach out!
"""

import jax.numpy as jnp
import numpy as np
import optax
from jax import value_and_grad
from jax.typing import DTypeLike

from quotonic.clements import Mesh
from quotonic.qpnn import IdealQPNN, ImperfectQPNN, TreeQPNN
from quotonic.types import jnp_ndarray
from quotonic.utils import genHaarUnitary


class Trainer:
    """Base class for a quantum photonic neural network (QPNN) trainer.

    This is effectively a template that prepares the fundamental attributes and methods for any QPNN trainer. Each
    trainer is designed to run a set number of optimization trials, each proceeding for a set number of epochs,
    with the option to print updates in a chosen interval of epochs. Also, each optimization trial will require a
    starting point, and the `initialize_params` method uses the provided `Mesh` instance upon initialization to help
    prepare these initial guesses easily.

    Attributes:
        num_trials (int): number of training trials to run
        num_epochs (int): number of training epochs to run
        print_every (int): specifies how often results should be printed, in terms of epochs
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer
            meshes) to be encoded and decoded, passed up from child class, otherwise defaults to a 4-mode mesh
    """

    def __init__(self, num_trials: int, num_epochs: int, print_every: int = 10, mesh: Mesh | None = None) -> None:
        """Initialization of a Trainer instance.

        Args:
            num_trials: number of training trials to run
            num_epochs: number of training epochs to run
            print_every: specifies how often results should be printed, in terms of epochs
            mesh: object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer
                meshes) to be encoded and decoded
        """

        # store the provided properties of the Trainer
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.mesh = mesh if mesh is not None else Mesh(4)

    def initialize_params(self, L: int) -> tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray]:
        """Initialize the phase shift parameters of a QPNN randomly.

        See the training description at the top of this module for more details. Here, $L$ is the number of layers in
        the QPNN and $m$ is the number of optical modes.

        Args:
            L: number of layers in the QPNN

        Returns:
            phi: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\phi}$, for all MZIs in each of the $L$
                meshes in the QPNN
            theta: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\theta}$, for all MZIs in each of the $L$
                meshes in the QPNN
            delta: $L\\times m$ array of phase shifts, $\\boldsymbol{\\delta}$, applied in each mode at the output of
                each of the $L$ meshes in the QPNN
        """

        # generate a random unitary from the Haar measure for each layer, and perform Clements decomposition
        # to extract the corresponding random phases
        m = self.mesh.m
        phi, theta, delta = (
            np.zeros((L, m * (m - 1) // 2), dtype=float),
            np.zeros((L, m * (m - 1) // 2), dtype=float),
            np.zeros((L, m), dtype=float),
        )
        for i in range(L):
            U = genHaarUnitary(m)
            phi[i], theta[i], delta[i] = self.mesh.decode(U)

        return jnp.asarray(phi), jnp.asarray(theta), jnp.asarray(delta)


class IdealTrainer(Trainer):
    """Class for training idealized QPNNs based on single-site Kerr-like nonlinearities.

    Attributes:
        num_trials (int): number of training trials to run
        num_epochs (int): number of training epochs to run
        print_every (int): specifies how often results should be printed, in terms of epochs
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer
            meshes) to be encoded and decoded, taken from IdealQPNN instance
        qpnn (IdealQPNN): object containing methods to construct the transfer function enacted by a QPNN, and compute
            the network fidelity
        sched (optax.Schedule): the exponential decay scheduler used during optimization
        opt (optax.GradientTransformation): the adam optimizer used during optimization
    """

    def __init__(
        self,
        qpnn: IdealQPNN,
        num_trials: int,
        num_epochs: int,
        print_every: int = 10,
        sched0: float = 0.025,
        sched_rate: float = 0.1,
    ) -> None:
        """Initialization of an Ideal Trainer instance.

        The exponential decay scheduler and optimizer are initialized here, so desired settings should be passed upon
        initialization if they differ from the default options.

        Args:
            qpnn: object containing methods to construct the transfer function enacted by a QPNN, and compute the
                network cost function
            num_trials: number of training trials to run
            num_epochs: number of training epochs to run
            print_every: specifies how often results should be printed, in terms of epochs
            sched0: initial value of the exponential decay scheduler used during optimization
            sched_rate: decay rate of the exponential decay scheduler used during optimization
        """

        super().__init__(num_trials, num_epochs, print_every=print_every, mesh=qpnn.mesh)

        # store the provided properties of the Ideal Trainer
        self.qpnn = qpnn

        # create the scheduler and optimizer for training
        self.sched = optax.exponential_decay(init_value=sched0, transition_steps=self.num_epochs, decay_rate=sched_rate)
        self.opt = optax.adam(self.sched)

    def cost(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> DTypeLike:
        """Evaluate the cost function that is minimized during training.

        See `[qpnn](qpnn.md)` for more details on the cost function.

        Args:
            phi: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\phi}$, for all MZIs in each of the $L$
                meshes in the QPNN
            theta: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\theta}$, for all MZIs in each of the $L$
                meshes in the QPNN
            delta: $L\\times m$ array of phase shifts, $\\boldsymbol{\\delta}$, applied in each mode at the output of
                each of the $L$ meshes in the QPNN

        Returns:
            C: cost (i.e. network error) of the QPNN
        """
        F = self.qpnn.calc_fidelity(phi, theta, delta)
        return 1 - F  # type: ignore

    def update(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray, optstate: optax.OptState
    ) -> tuple[DTypeLike, tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray], optax.OptState]:
        """Adjust the variational parameters to minimize the cost function.

        This method wraps around the `cost` function to evaluate it and its gradients with respect to the variational
        phase shift parameters. The updates to the parameters are computed using the attribute that stores the
        optimizer, then applied to the parameters which are subsequently returned alongside the value of the cost
        function and the state of the optimizer.

        Args:
            phi: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\phi}$, for all MZIs in each of the $L$
                meshes in the QPNN
            theta: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\theta}$, for all MZIs in each of the $L$
                meshes in the QPNN
            delta: $L\\times m$ array of phase shifts, $\\boldsymbol{\\delta}$, applied in each mode at the output of
                each of the $L$ meshes in the QPNN
            optstate: current state of the optimizer

        Returns:
            C: cost (i.e. network error) of the QPNN
            Theta: tuple `(phi, theta, delta)` including arrays that store the updated phase shift parameters
                $(\\boldsymbol{\\phi}, \\boldsymbol{\\theta}, \\boldsymbol{\\delta})$
            optstate: updated state of the optimizer
        """

        # calculate cost function and its gradient with respect to the 0th, 1st, 2nd parameters,
        # which are phi, theta, delta respectively
        C, grads = value_and_grad(self.cost, argnums=(0, 1, 2))(phi, theta, delta)

        # calculate updates to the parameters and the state of the optimizer from the gradients, then apply them
        updates, optstate = self.opt.update(grads, optstate)
        phi, theta, delta = optax.apply_updates((phi, theta, delta), updates)

        return C, (phi, theta, delta), optstate

    def train(self) -> dict:
        """Train the QPNN in a number of trials.

        A dictionary of results is first defined and initialized, then it is iteratively filled by training the given
        QPNN model in the given number of trials. Each trial, the phase shift parameters are initialized alongside
        the state of the optimizer, then the parameters are updated iteratively over the given number of epochs.
        Updates may be printed during training at set intervals of epochs, and at the end of each trial a summary
        statement is printed as well.

        Returns:
            Dictionary that contains the relevant results of the training simulation

                - **"F"** (`np_ndarray`): array of network fidelities at each trial and epoch, with shape `(num_trials,
                    num_epochs)`
                - **"phi"** (`np_ndarray`): array of optimized phase shifts $\\boldsymbol{\\phi}$ for all training
                    trials, with shape `(num_trials, L, m * (m - 1) // 2)`
                - **"theta"** (`np_ndarray`) array of optimized phase shifts $\\boldsymbol{\\theta}$ for all training
                    trials, with shape `(num_trials, L, m * (m - 1) // 2)`
                - **"delta"** (`np_ndarray`) array of optimized phase shifts $\\boldsymbol{\\delta}$ for all training
                    trials, with shape `(num_trials, L, m)`
        """

        # prepare the results dictionary
        results = {
            "F": np.empty((self.num_trials, self.num_epochs), dtype=float),
            "phi": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
            "theta": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
            "delta": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m), dtype=float),
        }

        for trial in range(self.num_trials):
            print(f"Trial: {trial + 1:d}")

            # prepare the initial parameters and initial state of the optimizer
            Theta0 = self.initialize_params(self.qpnn.L)
            initial_optstate = self.opt.init(Theta0)

            # iterate through the epochs, optimizing the parameters at each iteration
            Theta = Theta0
            optstate = initial_optstate
            F = np.zeros(self.num_epochs, dtype=float)
            C = 1.0
            for epoch in range(self.num_epochs):
                C, Theta, optstate = self.update(*Theta, optstate)
                F[epoch] = 1 - C  # type: ignore

                if epoch % self.print_every == 0:
                    print(f"Epoch: {epoch:d} \t Cost: {C:.4e} \t Fidelity: {F[epoch]:.4g}")
            print(f"COMPLETE! \t Cost: {C:.4e} \t Fidelity: {F[-1]:.4g}")

            # store the results from this trial
            results["F"][trial] = F
            results["phi"][trial], results["theta"][trial], results["delta"][trial] = [
                np.asarray(Theta[i]) for i in range(3)
            ]

            print("")

        return results


class ImperfectTrainer(Trainer):
    """Class for training imperfect QPNNs based on single-site Kerr-like nonlinearities.

    Attributes:
        num_trials (int): number of training trials to run
        num_epochs (int): number of training epochs to run
        print_every (int): specifies how often results should be printed, in terms of epochs
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer
            meshes) to be encoded and decoded, taken from ImperfectQPNN instance
        qpnn (ImperfectQPNN): object containing methods to construct the transfer function enacted by a QPNN, and
            compute the network performance measures
        sched (optax.Schedule): the exponential decay scheduler used during optimization
        opt (optax.GradientTransformation): the adam optimizer used during optimization
    """

    def __init__(
        self,
        qpnn: ImperfectQPNN,
        num_trials: int,
        num_epochs: int,
        print_every: int = 10,
        sched0: float = 0.025,
        sched_rate: float = 0.1,
    ) -> None:
        """Initialization of an Imperfect Trainer instance.

        The exponential decay scheduler and optimizer are initialized here, so desired settings should be passed upon
        initialization if they differ from the default options.

        Args:
            qpnn: object containing methods to construct the transfer function enacted by a QPNN, and compute the
                network performance measures
            num_trials: number of training trials to run
            num_epochs: number of training epochs to run
            print_every: specifies how often results should be printed, in terms of epochs
            sched0: initial value of the exponential decay scheduler used during optimization
            sched_rate: decay rate of the exponential decay scheduler used during optimization
        """

        super().__init__(num_trials, num_epochs, print_every=print_every, mesh=qpnn.meshes[0])

        # store the provided properties of the Imperfect Trainer
        self.qpnn = qpnn

        # create the scheduler and optimizer for training
        self.sched = optax.exponential_decay(init_value=sched0, transition_steps=self.num_epochs, decay_rate=sched_rate)
        self.opt = optax.adam(self.sched)

    def cost(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> DTypeLike:
        """Evaluate the cost function that is minimized during training.

        See `[qpnn](qpnn.md)` for more details on the cost function.

        Args:
            phi: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\phi}$, for all MZIs in each of the $L$
                meshes in the QPNN
            theta: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\theta}$, for all MZIs in each of the $L$
                meshes in the QPNN
            delta: $L\\times m$ array of phase shifts, $\\boldsymbol{\\delta}$, applied in each mode at the output of
                each of the $L$ meshes in the QPNN

        Returns:
            C: cost (i.e. network error) of the QPNN
        """
        Fu = self.qpnn.calc_unc_fidelity(phi, theta, delta)
        return 1 - Fu  # type: ignore

    def update(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray, optstate: optax.OptState
    ) -> tuple[DTypeLike, tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray], optax.OptState]:
        """Adjust the variational parameters to minimize the cost function.

        This method wraps around the `cost` function to evaluate it and its gradients with respect to the variational
        phase shift parameters. The updates to the parameters are computed using the attribute that stores the
        optimizer, then applied to the parameters which are subsequently returned alongside the value of the cost
        function and the state of the optimizer.

        Args:
            phi: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\phi}$, for all MZIs in each of the $L$
                meshes in the QPNN
            theta: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\theta}$, for all MZIs in each of the $L$
                meshes in the QPNN
            delta: $L\\times m$ array of phase shifts, $\\boldsymbol{\\delta}$, applied in each mode at the output of
                each of the $L$ meshes in the QPNN
            optstate: current state of the optimizer

        Returns:
            C: cost (i.e. network error) of the QPNN
            Theta: tuple `(phi, theta, delta)` including arrays that store the updated phase shift parameters
                $(\\boldsymbol{\\phi}, \\boldsymbol{\\theta}, \\boldsymbol{\\delta})$
            optstate: updated state of the optimizer
        """

        # calculate cost function and its gradient with respect to the 0th, 1st, 2nd parameters,
        # which are phi, theta,delta respectively
        C, grads = value_and_grad(self.cost, argnums=(0, 1, 2))(phi, theta, delta)

        # calculate updates to the parameters and the state of the optimizer from the gradients, then apply them
        updates, optstate = self.opt.update(grads, optstate)
        phi, theta, delta = optax.apply_updates((phi, theta, delta), updates)

        return C, (phi, theta, delta), optstate

    def train(self) -> dict:
        """Train the QPNN in a number of trials.

        A dictionary of results is first defined and initialized, then it is iteratively filled by training the given
        QPNN model in the given number of trials. Each trial, the phase shift parameters are initialized alongside
        the state of the optimizer, then the parameters are updated iteratively over the given number of epochs.
        Updates may be printed during training at set intervals of epochs, and at the end of each trial a summary
        statement is printed as well.

        Returns:
            Dictionary that contains the relevant results of the training simulation

                - **"Fu"** (`np_ndarray`): array of network unconditional fidelities at each trial and epoch,
                    with shape `(num_trials, num_epochs)`
                - **"Fc"** (`np_ndarray`): array of network conditional fidelities measured at the end of each trial,
                    with shape `(num_trials,)`
                - **"rate"** (`np_ndarray`): array of network logical rates measured at the end of each trial,
                    with shape `(num_trials,)`
                - **"phi"** (`np_ndarray`): array of optimized phase shifts $\\boldsymbol{\\phi}$ for all training
                    trials, with shape `(num_trials, L, m * (m - 1) // 2)`
                - **"theta"** (`np_ndarray`) array of optimized phase shifts $\\boldsymbol{\\theta}$ for all training
                    trials, with shape `(num_trials, L, m * (m - 1) // 2)`
                - **"delta"** (`np_ndarray`) array of optimized phase shifts $\\boldsymbol{\\delta}$ for all training
                    trials, with shape `(num_trials, L, m)`
                - **"ell_mzi"** (`np_ndarray`) array of fractional losses per arm of each MZI in each QPNN trained in
                    each trial, with shape `(num_trials, L, m, m)`
                - **"ell_ps"** (`np_ndarray`) array of fractional losses per mesh output phase shifter in each QPNN
                    trained in each trial, with shape `(num_trials, L, m)`
                - **"t_dc"** (`np_ndarray`) array of directional coupler splitting ratios (T:R) throughout each QPNN
                    trained in each trial, with shape `(num_trials, L, 2, m * (m - 1) // 2)`
        """

        # prepare the results dictionary
        results = {
            "Fu": np.empty((self.num_trials, self.num_epochs), dtype=float),
            "Fc": np.empty((self.num_trials,), dtype=float),
            "rate": np.empty((self.num_trials,), dtype=float),
            "phi": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
            "theta": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
            "delta": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m), dtype=float),
            "ell_mzi": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m, self.qpnn.m), dtype=float),
            "ell_ps": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m), dtype=float),
            "t_dc": np.empty((self.num_trials, self.qpnn.L, 2, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
        }

        for trial in range(self.num_trials):
            print(f"Trial: {trial + 1:d}")

            # refresh the imperfection model for the qpnn if not the first trial
            if trial > 0:
                self.qpnn = ImperfectQPNN(
                    self.qpnn.n,
                    self.qpnn.m,
                    self.qpnn.L,
                    varphi=self.qpnn.varphi,
                    ell_mzi=self.qpnn.ell_mzi,
                    ell_ps=self.qpnn.ell_ps,
                    t_dc=self.qpnn.t_dc,
                    training_set=self.qpnn.training_set,
                )

            # prepare the initial parameters and initial state of the optimizer
            Theta0 = self.initialize_params(self.qpnn.L)
            initial_optstate = self.opt.init(Theta0)

            # iterate through the epochs, optimizing the parameters at each iteration
            Theta = Theta0
            optstate = initial_optstate
            Fu = np.zeros(self.num_epochs, dtype=float)
            C = 1.0
            for epoch in range(self.num_epochs):
                C, Theta, optstate = self.update(*Theta, optstate)
                Fu[epoch] = 1 - C  # type: ignore

                if epoch % self.print_every == 0:
                    print(f"Epoch: {epoch:d} \t Cost: {C:.4e} \t Unconditional Fidelity: {Fu[epoch]:.4g}")

            # compute performance measures of the trained QPNN
            _, Fc, rate = self.qpnn.calc_performance_measures(*Theta)
            print(
                f"COMPLETE! \t Cost: {C:.4e} \t Unconditional Fidelity: {Fu[-1]:.4g} \t Conditional Fidelity: {Fc:.4g} \t Rate: {rate:.4g}"
            )

            # store the results from this trial
            results["Fu"][trial] = Fu
            results["Fc"][trial] = Fc
            results["rate"][trial] = rate
            results["phi"][trial], results["theta"][trial], results["delta"][trial] = [
                np.asarray(Theta[i]) for i in range(3)
            ]
            results["ell_mzi"][trial], results["ell_ps"][trial], results["t_dc"][trial] = self.qpnn.imperfections

            print("")

        return results


class TreeTrainer(Trainer):
    """Class for training imperfect QPNNs based on three-level system photon subtraction/addition nonlinearities that
    power a tree-type photonic cluster state generation protocol.

    Attributes:
        num_trials (int): number of training trials to run
        num_epochs (int): number of training epochs to run
        print_every (int): specifies how often results should be printed, in terms of epochs
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer
            meshes) to be encoded and decoded, taken from TreeQPNN instance
        qpnn (TreeQPNNExtended): object containing methods to construct the transfer function enacted by a QPNN, and
            compute the network performance measures
        sched (optax.Schedule): the exponential decay scheduler used during optimization
        opt (optax.GradientTransformation): the adam optimizer used during optimization
    """

    def __init__(
        self,
        qpnn: TreeQPNN,
        num_trials: int,
        num_epochs: int,
        print_every: int = 10,
        sched0: float = 0.025,
        sched_rate: float = 0.1,
    ) -> None:
        """Initialization of a Tree Trainer instance.

        The exponential decay scheduler and optimizer are initialized here, so desired settings should be passed upon
        initialization if they differ from the default options.

        Args:
            qpnn: object containing methods to construct the transfer function enacted by a QPNN, and compute the
                network performance measures
            num_trials: number of training trials to run
            num_epochs: number of training epochs to run
            print_every: specifies how often results should be printed, in terms of epochs
            sched0: initial value of the exponential decay scheduler used during optimization
            sched_rate: decay rate of the exponential decay scheduler used during optimization
        """

        super().__init__(num_trials, num_epochs, print_every=print_every, mesh=qpnn.meshes[0])

        # store the provided properties of the Tree Trainer
        self.qpnn = qpnn

        # create the scheduler and optimizer for training
        self.sched = optax.exponential_decay(init_value=sched0, transition_steps=self.num_epochs, decay_rate=sched_rate)
        self.opt = optax.adam(self.sched)

    def cost(self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray) -> DTypeLike:
        """Evaluate the cost function that is minimized during training.

        See `[qpnn](qpnn.md)` for more details on the cost function.

        Args:
            phi: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\phi}$, for all MZIs in each of the $L$
                meshes in the QPNN
            theta: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\theta}$, for all MZIs in each of the $L$
                meshes in the QPNN
            delta: $L\\times m$ array of phase shifts, $\\boldsymbol{\\delta}$, applied in each mode at the output of
                each of the $L$ meshes in the QPNN

        Returns:
            C: cost (i.e. network error) of the QPNN
        """
        cost = self.qpnn.calc_cost(phi, theta, delta)
        return cost  # type: ignore

    def update(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray, optstate: optax.OptState
    ) -> tuple[DTypeLike, tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray], optax.OptState]:
        """Adjust the variational parameters to minimize the cost function.

        This method wraps around the `cost` function to evaluate it and its gradients with respect to the variational
        phase shift parameters. The updates to the parameters are computed using the attribute that stores the
        optimizer, then applied to the parameters which are subsequently returned alongside the value of the cost
        function and the state of the optimizer.

        Args:
            phi: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\phi}$, for all MZIs in each of the $L$
                meshes in the QPNN
            theta: $L\\times m(m-1)/2$ array of phase shifts, $\\boldsymbol{\\theta}$, for all MZIs in each of the $L$
                meshes in the QPNN
            delta: $L\\times m$ array of phase shifts, $\\boldsymbol{\\delta}$, applied in each mode at the output of
                each of the $L$ meshes in the QPNN
            optstate: current state of the optimizer

        Returns:
            C: cost (i.e. network error) of the QPNN
            Theta: tuple `(phi, theta, delta)` including arrays that store the updated phase shift parameters
                $(\\boldsymbol{\\phi}, \\boldsymbol{\\theta}, \\boldsymbol{\\delta})$
            optstate: updated state of the optimizer
        """

        # calculate cost function and its gradient with respect to the 0th, 1st, 2nd parameters,
        # which are phi, theta, delta respectively
        C, grads = value_and_grad(self.cost, argnums=(0, 1, 2))(phi, theta, delta)

        # calculate updates to the parameters and the state of the optimizer from the gradients, then apply them
        updates, optstate = self.opt.update(grads, optstate)
        phi, theta, delta = optax.apply_updates((phi, theta, delta), updates)

        return C, (phi, theta, delta), optstate

    def train(self) -> dict:
        """Train the QPNN in a number of trials.

        A dictionary of results is first defined and initialized, then it is iteratively filled by training the given
        QPNN model in the given number of trials. Each trial, the phase shift parameters are initialized alongside
        the state of the optimizer, then the parameters are updated iteratively over the given number of epochs.
        Updates may be printed during training at set intervals of epochs, and at the end of each trial a summary
        statement is printed as well.

        Returns:
            Dictionary that contains the relevant results of the training simulation

                - **"cost"** (`np_ndarray`): array of network cost function values at each trial and epoch,
                    with shape `(num_trials, num_epochs)`
                - **"fid"** (`np_ndarray`): array of network fidelities measured at the end of each trial,
                    with shape `(num_trials,)`
                - **"succ_rate"** (`np_ndarray`): array of network success rates measured at the end of each trial,
                    with shape `(num_trials,)`
                - **"logi_rate"** (`np_ndarray`): array of network logical rates measured at the end of each trial,
                    with shape `(num_trials,)`
                - **"phi"** (`np_ndarray`): array of optimized phase shifts $\\boldsymbol{\\phi}$ for all training
                    trials, with shape `(num_trials, L, m * (m - 1) // 2)`
                - **"theta"** (`np_ndarray`) array of optimized phase shifts $\\boldsymbol{\\theta}$ for all training
                    trials, with shape `(num_trials, L, m * (m - 1) // 2)`
                - **"delta"** (`np_ndarray`) array of optimized phase shifts $\\boldsymbol{\\delta}$ for all training
                    trials, with shape `(num_trials, L, m)`
                - **"ell_mzi"** (`np_ndarray`) array of fractional losses per arm of each MZI in each QPNN trained in
                    each trial, with shape `(num_trials, L, m, m)`
                - **"ell_ps"** (`np_ndarray`) array of fractional losses per mesh output phase shifter in each QPNN
                    trained in each trial, with shape `(num_trials, L, m)`
                - **"t_dc"** (`np_ndarray`) array of directional coupler splitting ratios (T:R) throughout each QPNN
                    trained in each trial, with shape `(num_trials, L, 2, m * (m - 1) // 2)`
        """

        # prepare the results dictionary
        results = {
            "cost": np.empty((self.num_trials, self.num_epochs), dtype=float),
            "fid": np.empty((self.num_trials,), dtype=float),
            "succ_rate": np.empty((self.num_trials,), dtype=float),
            "logi_rate": np.empty((self.num_trials,), dtype=float),
            "phi": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
            "theta": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
            "delta": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m), dtype=float),
            "ell_mzi": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m, self.qpnn.m), dtype=float),
            "ell_ps": np.empty((self.num_trials, self.qpnn.L, self.qpnn.m), dtype=float),
            "t_dc": np.empty((self.num_trials, self.qpnn.L, 2, self.qpnn.m * (self.qpnn.m - 1) // 2), dtype=float),
        }

        for trial in range(self.num_trials):
            print(f"Trial: {trial + 1:d}")

            # refresh the imperfection model for the qpnn if not the first trial
            if trial > 0:
                self.qpnn = TreeQPNN(
                    self.qpnn.b,
                    self.qpnn.L,
                    varphi=self.qpnn.varphi,
                    ell_mzi=self.qpnn.ell_mzi,
                    ell_ps=self.qpnn.ell_ps,
                    t_dc=self.qpnn.t_dc,
                    training_set=self.qpnn.training_set,
                )

            # prepare the initial parameters and initial state of the optimizer
            Theta0 = self.initialize_params(self.qpnn.L)
            initial_optstate = self.opt.init(Theta0)

            # iterate through the epochs, optimizing the parameters at each iteration
            Theta = Theta0
            optstate = initial_optstate
            cost = np.zeros(self.num_epochs, dtype=float)
            C = 1.0
            for epoch in range(self.num_epochs):
                C, Theta, optstate = self.update(*Theta, optstate)
                cost[epoch] = C  # type: ignore

                if epoch % self.print_every == 0:
                    print(f"Epoch: {epoch:d} \t Cost: {C:.4e} \t Success Rate: {1 - cost[epoch]:.4g}")

            # compute performance measures of the trained QPNN
            fid, succ_rate, logi_rate = self.qpnn.calc_overall_performance_measures(*Theta)
            print(f"COMPLETE! \t Cost: {C:.4e} \t Fid: {fid:.4g} \t Succ: {succ_rate:.4g} \t Logi: {logi_rate:.4g}")

            # store the results from this trial
            results["cost"][trial] = cost
            results["fid"][trial] = fid
            results["succ_rate"][trial] = succ_rate
            results["logi_rate"][trial] = logi_rate
            results["phi"][trial], results["theta"][trial], results["delta"][trial] = [
                np.asarray(Theta[i]) for i in range(3)
            ]
            results["ell_mzi"][trial], results["ell_ps"][trial], results["t_dc"][trial] = self.qpnn.imperfections

            print("")

        return results
