"""
The `quotonic.trainer` module includes ...
give a very general description here of how the training works, can copy from the tree manuscript
if we describe everything here, then the remaining documentation can just focus on additions/extensions to the model
"""

from typing import Optional, Tuple

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

    ADD DOCUMENTATION HERE

    Attributes:
        num_trials (int): number of training trials to run
        num_epochs (int): number of training epochs to run
        print_every (int): specifies how often results should be printed, in terms of epochs
        mesh (Mesh): object containing methods that allow linear layers (i.e. rectangular Mach-Zehnder interferometer
            meshes) to be encoded and decoded, passed up from child class, otherwise defaults to a 4-mode mesh
    """

    def __init__(self, num_trials: int, num_epochs: int, print_every: int = 10, mesh: Optional[Mesh] = None) -> None:
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

    def initialize_params(self, L: int) -> Tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray]:
        """Initialize the phase shift parameters of a QPNN randomly.

        Args:
            L: number of layers in the QPNN

        Returns:
            A tuple containing arrays of the $\\phi$ ($L\\times m(m-1) / 2$), $\\theta$ ($L\\times m(m-1) / 2$), and
                $\\delta$ ($L\\times m$) phase shifts for each component of the QPNN
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

    ADD DOCUMENTATION HERE

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

        ADD DOCUMENTATION HERE

        Args:
            qpnn: object containing methods to construct the transfer function enacted by a QPNN, and compute the
                network fidelity
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

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer of the QPNN being trained

        Returns:
            Cost (i.e. network error) of the QPNN
        """
        F = self.qpnn.calc_fidelity(phi, theta, delta)
        return 1 - F  # type: ignore

    def update(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray, optstate: optax.OptState
    ) -> Tuple[DTypeLike, Tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray], optax.OptState]:
        """Adjust the variational parameters to minimize the cost function.

        ADD DOCUMENTATION HERE

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer of the QPNN being trained
            optstate: current state of the optimizer

        Returns:
            A tuple including the current value of the cost function, another tuple of the current parameters ($\\phi$,
                $\\theta$, $\\delta$), and the current state of the optimizer
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

        ADD DOCUMENTATION HERE

        Returns:
            Dictionary that contains the relevant results of the training simulation (needs more documentation)
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

    ADD DOCUMENTATION HERE

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

        ADD DOCUMENTATION HERE

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

        ADD DOCUMENTATION HERE

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the ith
                layer of the QPNN being trained
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer of the QPNN being trained

        Returns:
            Cost (i.e. network error) of the QPNN
        """
        Fu = self.qpnn.calc_unc_fidelity(phi, theta, delta)
        return 1 - Fu  # type: ignore

    def update(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray, optstate: optax.OptState
    ) -> Tuple[DTypeLike, Tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray], optax.OptState]:
        """Adjust the variational parameters to minimize the cost function.

        ADD DOCUMENTATION HERE

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer of the QPNN being trained
            optstate: current state of the optimizer

        Returns:
            A tuple including the current value of the cost function, another tuple of the current parameters ($\\phi$,
                $\\theta$, $\\delta$), and the current state of the optimizer
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

        ADD DOCUMENTATION HERE

        Returns:
            Dictionary that contains the relevant results of the training simulation (needs more documentation)
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
            for i in range(self.qpnn.L):
                results["ell_mzi"][trial][i] = self.qpnn.meshes[i].ell_mzi
                results["ell_ps"][trial][i] = self.qpnn.meshes[i].ell_ps
                results["t_dc"][trial][i] = self.qpnn.meshes[i].t_dc

            print("")

        return results


class TreeTrainer(Trainer):
    """Class for training imperfect QPNNs based on three-level system photon subtraction/addition nonlinearities that
    power a tree-type photonic cluster state generation protocol.

    ADD DOCUMENTATION HERE

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

        ADD DOCUMENTATION HERE

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

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the output
                of the mesh in the ith layer of the QPNN being trained

        Returns:
            Cost (i.e. network error) of the QPNN
        """
        Fu_full = self.qpnn.calc_unc_fidelity(phi, theta, delta)
        return 1 - Fu_full  # type: ignore

    def update(
        self, phi: jnp_ndarray, theta: jnp_ndarray, delta: jnp_ndarray, optstate: optax.OptState
    ) -> Tuple[DTypeLike, Tuple[jnp_ndarray, jnp_ndarray, jnp_ndarray], optax.OptState]:
        """Adjust the variational parameters to minimize the cost function.

        ADD DOCUMENTATION HERE

        Args:
            phi: $L\\times m(m-1)/2$ phase shifts, $\\phi$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            theta: $L\\times m(m-1)/2$ phase shifts, $\\theta$, where the ith row contains those for each MZI in the
                ith layer of the QPNN being trained
            delta: $L\\times m$ phase shifts, $\\delta$, where the ith row contains those for each mode at the
                output of the mesh in the ith layer of the QPNN being trained
            optstate: current state of the optimizer

        Returns:
            A tuple including the current value of the cost function, another tuple of the current parameters ($\\phi$,
                $\\theta$, $\\delta$), and the current state of the optimizer
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

        ADD DOCUMENTATION HERE

        Returns:
            Dictionary that contains the relevant results of the training simulation (needs more documentation)
        """

        # prepare the results dictionary
        results = {
            "Fu_full": np.empty((self.num_trials, self.num_epochs), dtype=float),
            "Fu": np.empty((self.num_trials,), dtype=float),
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
            Fu_full = np.zeros(self.num_epochs, dtype=float)
            C = 1.0
            for epoch in range(self.num_epochs):
                C, Theta, optstate = self.update(*Theta, optstate)
                Fu_full[epoch] = 1 - C  # type: ignore

                if epoch % self.print_every == 0:
                    print(f"Epoch: {epoch:d} \t Cost: {C:.4e} \t Full Unconditional Fidelity: {Fu_full[epoch]:.4g}")

            # compute performance measures of the trained QPNN
            _, Fu, Fc, rate = self.qpnn.calc_performance_measures(*Theta)
            print(f"COMPLETE! \t Cost: {C:.4e} \t Fu: {Fu:.4g} \t Fc: {Fc:.4g} \t Rate: {rate:.4g}")

            # store the results from this trial
            results["Fu_full"][trial] = Fu_full
            results["Fu"][trial] = Fu
            results["Fc"][trial] = Fc
            results["rate"][trial] = rate
            results["phi"][trial], results["theta"][trial], results["delta"][trial] = [
                np.asarray(Theta[i]) for i in range(3)
            ]
            for i in range(self.qpnn.L):
                results["ell_mzi"][trial][i] = self.qpnn.meshes[i].ell_mzi
                results["ell_ps"][trial][i] = self.qpnn.meshes[i].ell_ps
                results["t_dc"][trial][i] = self.qpnn.meshes[i].t_dc

            print("")

        return results
