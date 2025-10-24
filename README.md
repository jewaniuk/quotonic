# quotonic

[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-ff69b4)]()
[![license](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/jewaniuk/quotonic/blob/main/LICENSE)
[![build](https://github.com/jewaniuk/quotonic/actions/workflows/quality.yml/badge.svg)](https://github.com/jewaniuk/quotonic/actions/workflows/quality.yml)
[![tests](https://github.com/jewaniuk/quotonic/actions/workflows/tests.yml/badge.svg)](https://github.com/jewaniuk/quotonic/actions/workflows/tests.yml)
[![docs](https://github.com/jewaniuk/quotonic/actions/workflows/docs.yml/badge.svg)](https://github.com/jewaniuk/quotonic/actions/workflows/docs.yml)
[![wheels](https://github.com/jewaniuk/quotonic/actions/workflows/wheels.yml/badge.svg)](https://github.com/jewaniuk/quotonic/actions/workflows/wheels.yml)
[![commits](https://img.shields.io/github/commit-activity/m/jewaniuk/quotonic)](https://img.shields.io/github/commit-activity/m/jewaniuk/quotonic)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![coverage](./badges/coverage.svg)](https://pytest-cov.readthedocs.io/en/latest/)
[![flake8](./badges/flake8.svg)](https://flake8.pycqa.org/en/latest/)

<p>
<img src="docs/img/light/qpnn-visualization.png" alt="qpnn visualization">
</p>

`quotonic` is a package created for studying nonlinear quantum photonic circuits, including yet not limited to,
quantum photonic neural networks (QPNNs). It is designed to accommodate new circuit models that explore unknown
capabilities, teaching all of us what can be accomplished when a handful of photons are combined with strong few-photon
optical nonlinearities. We hope that you will use this package as a platform to begin answering pertinent research
questions around nonlinear quantum photonic circuits like QPNNs. If you are able to do so, and would like to make
additions here, please let us know! We would  love for this package to grow, including many different models that can
be explored in tandem.

When it comes to simulating quantum dynamics using classical computational resources, there is often a need to closely
consider performance. Here, we write circuit models to be compatible with [`jax`](https://github.com/jax-ml/jax) and
thus owe a massive thank you to the developers at [Google DeepMind](https://deepmind.google/). It is also necessary to
mention and thank similar packages, each of which have inspired some of the code within `quotonic`.
- [Bosonic: A Quantum Optics Library](https://github.com/steinbrecher/bosonic)
- [Cascaded Optical Systems Approach to Neural Networks (CasOptAx)](https://github.com/JasvithBasani/CasOptAx)
- [Piquasso](https://github.com/Budapest-Quantum-Computing-Group/piquasso)
- [The Walrus](https://github.com/XanaduAI/thewalrus)

The documentation for `quotonic` is live at [jewaniuk.github.io/quotonic](https://jewaniuk.github.io/quotonic/). It was
prepared using [`mkdocstrings`](https://mkdocstrings.github.io/) with
[`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/).

## Installation
You can install the latest release of `quotonic` from PyPI as
```shell
pip install quotonic
```
or install the latest development version from GitHub
```shell
pip install git+https://github.com/jewaniuk/quotonic.git
```

## Getting Started
`quotonic.qpnn` contains a variety of models of QPNNs, the simplest of which is `IdealQPNN`, which follows their
original  proposal. Each model is accompanied by a trainer from `quotonic.trainer`, and can be trained to perform some
task defined by a training set from `quotonic.training_sets`. In this example, we will train a two-layer, four-mode
QPNN to act as a deterministic two-photon CNOT gate.

```python
>>> from quotonic.qpnn import IdealQPNN
>>> from quotonic.trainer import IdealTrainer
>>> from quotonic.training_sets import CNOT

>>> n = 2  # number of photons
>>> m = 4  # number of optical modes
>>> L = 2  # number of network layers
```
We'll choose to perform one optimization trial that proceeds through 100 epochs.
```python

>>> num_trials = 1  # number of optimization trials to perform
>>> num_epochs = 100  # number of epochs to train for per trial
```
Now, we simply prepare the training set, instantiate a QPNN followed by a trainer, then train!
```python
>>> training_set = CNOT()
>>> qpnn = IdealQPNN(n, m, L, training_set=training_set)
>>> trainer = IdealTrainer(qpnn, num_trials, num_epochs)

>>> results = trainer.train()
Trial: 1
Epoch: 0 	 Cost: 9.3885e-01 	 Fidelity: 0.06115
Epoch: 10 	 Cost: 7.1853e-01 	 Fidelity: 0.2815
Epoch: 20 	 Cost: 6.6739e-01 	 Fidelity: 0.3326
Epoch: 30 	 Cost: 5.4786e-01 	 Fidelity: 0.4521
Epoch: 40 	 Cost: 4.2395e-01 	 Fidelity: 0.576
Epoch: 50 	 Cost: 3.0212e-01 	 Fidelity: 0.6979
Epoch: 60 	 Cost: 1.6486e-01 	 Fidelity: 0.8351
Epoch: 70 	 Cost: 7.8349e-02 	 Fidelity: 0.9217
Epoch: 80 	 Cost: 3.2369e-02 	 Fidelity: 0.9676
Epoch: 90 	 Cost: 1.2706e-02 	 Fidelity: 0.9873
Epoch: 100 	 Cost: 5.2117e-03 	 Fidelity: 0.9948
Epoch: 110 	 Cost: 2.0834e-03 	 Fidelity: 0.9979
Epoch: 120 	 Cost: 8.6629e-04 	 Fidelity: 0.9991
Epoch: 130 	 Cost: 3.8522e-04 	 Fidelity: 0.9996
Epoch: 140 	 Cost: 1.8960e-04 	 Fidelity: 0.9998
COMPLETE! 	 Cost: 1.1343e-04 	 Fidelity: 0.9999
```
This particular trial was able to tune the network parameters to achieve a fidelity of ~99.99%. The optimized parameters
are passed back in `results`, so we can calculate the fidelity directly using them to check.
```python
>>> qpnn.calc_fidelity(results["phi"][0], results["theta"][0], results["delta"][0])
Array(0.99989176, dtype=float32)
```

## Example Usage

Here, we provide example scripts that illustrate the methodology used in our previous research on QPNNs.

### [Imperfect Quantum Photonic Neural Networks](https://doi.org/10.1002/qute.202200125)
```python
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from quotonic.qpnn import ImperfectQPNN
from quotonic.trainer import ImperfectTrainer
from quotonic.training_sets import BSA

n = 2
m = 4
L = 2
varphi = np.pi / 2

ell_mzi = (0.00861, 0.00057)  # (0.00861 +/- 0.00057) dB loss per MZI, sota model
ell_ps = (0.0015, 0.0001)  # (0.0015 +/- 0.0001) dB loss per phase shifter, sota model
t_dc = (0.5000, 0.0508)  # (50.00 +/- 5.08) % T:R directional coupler splitting ratio

num_trials = 200
num_epochs = 1000
print_every = 100

tset = BSA()
qpnn = ImperfectQPNN(n, m, L, varphi=varphi, ell_mzi=ell_mzi, ell_ps=ell_ps, t_dc=t_dc, training_set=tset)
trainer = ImperfectTrainer(qpnn, num_trials, num_epochs, print_every=print_every)

results = trainer.train()
```

### [Large-Scale Tree-Type Photonic Cluster State Generation with Recurrent Quantum Photonic Neural Networks](https://doi.org/10.48550/arXiv.2505.14628)
```python
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from quotonic.qpnn import TreeQPNN
from quotonic.trainer import TreeTrainer
from quotonic.training_sets import Tree

b = 2
n = b + 1
m = 2 * n
L = 2
varphi = (0.0, np.pi)

ell_mzi = (0.0210, 0.0016)  # (0.0210 +/- 0.0016) dB loss per MZI, multi model
ell_ps = (0.0100, 0.0006)  # (0.0100 +/- 0.0006) dB loss per phase shifter, multi model
t_dc = (0.50, 0.005)  # (50 +/- 0.5) % T:R directional coupler splitting ratio

num_trials = 200
num_epochs = 1000
print_every = 100

tset = Tree(b)
qpnn = TreeQPNN(b, L, varphi=varphi, ell_mzi=ell_mzi, ell_ps=ell_ps, t_dc=t_dc, training_set=tset)
trainer = TreeTrainer(qpnn, num_trials, num_epochs, print_every=print_every)

results = trainer.train()
```

## Citing
Rather than citing the package directly, please cite the following works that it was developed for:
```
@article{Ewaniuk:23,
title   = {{Imperfect Quantum Photonic Neural Networks}},
author  = {Jacob Ewaniuk and Jacques Carolan and Bhavin J. Shastri and Nir Rotenberg},
journal = {Advanced Quantum Technologies},
volume  = {6},
pages   = {2200125},
year    = {2023},
doi     = {https://doi.org/10.1002/qute.202200125},
}
```
```
@misc{Ewaniuk:25,
title   = {{Large-Scale Tree-Type Photonic Cluster State Generation with Recurrent Quantum Photonic Neural Networks}},
author  = {Jacob Ewaniuk and Bhavin J. Shastri and Nir Rotenberg},
year    = {2025},
howpublished = {Preprint at https://arxiv.org/abs/2505.14628},
}
```

## Authors
`quotonic` was initially created by [Jacob Ewaniuk](https://www.linkedin.com/in/jacobewaniuk/) as part of his doctoral
studies at [Queen's University](https://www.queensu.ca/), working with the
[Quantum Nanophotonics Lab](https://www.queensu.ca/physics/qnl/) and
[Shastri Lab](https://www.queensu.ca/physics/shastrilab/). Currently, it is in active use as a research tool by a
number of graduate students in each of these research groups. As further contributions are made, additional authors
will be listed here.