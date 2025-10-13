"""
The `quotonic.types` module includes type definitions for `numpy` and `jax.numpy` arrays respectively. These labels are
used throughout the package such that the different array types are clearly labelled and differentiated.
"""

from typing import TypeAlias
import numpy as np
import jax.numpy as jnp

np_ndarray: TypeAlias = np.ndarray
"""type alias for a NumPy ndarray (from `numpy`)"""

jnp_ndarray: TypeAlias = jnp.ndarray
"""type alias for a JAX ndarray (from `jax.numpy`)"""
