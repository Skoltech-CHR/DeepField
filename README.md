[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)
[![Pytorch](https://img.shields.io/badge/PyTorch-orange.svg)](https://pytorch.org)
![](https://github.com/Skoltech-CHR/DeepField/workflows/pylint-check/badge.svg)


# DeepField

Machine learning framework for reservoir simulation.

![img](static/3d_basic.PNG)

## Features

* reservoir representation with Grid, Rock, States, Wells, Aquifer and PVT-tables components
* interactive 3D visualization with some advanced options
* common reservoir preprocessing tools
* working with arbitrary large datasets of field simulations
* constructor of neural network models
* generative models for field augmentation
* various model training scenarious for arbitrary long simulation periods
* detailed [documentation](https://Skoltech-CHR.github.io/DeepField) and step-by-step [tutorials](/tutorials)
* complete [pipelines](/pipelines) of the reservoir simulation steps


![img](static/framework.PNG)

## Installation

Clone the repository:

    git clone https://github.com/Skoltech-CHR/DeepField.git

Working with a remote server, it is recommended to install
VNC for remote rendering of 3D graphics (follow this [instruction](./vnc/README.md))

Another option is to build the docker image with DeepField inside.
Instructions and dockerfile are provided in the [docker](./docker) directory.

```
Note: the project is in developement. We welcome contributions and collaborations.
```

## Quick start

Load a reservoir model from `.DATA` file:

```python

  from deepfield import Field

  model = Field('model.data').load()
```

See the [tutorials](./tutorials) to explore the framework step-by-step
and the [documentation](https://Skoltech-CHR.github.io/DeepField) for more details.


## Model formats

Initial reservoir model can be given in a mixture of ECLIPSE, MORE, PETREL, tNavigator formats.
However, there is no guarantee that any mixture will be understood.
Main file should be in .DATA file. Dependencies can be text and binary files including common formats:

* .GRDECL
* .INC
* .RSM
* .UNRST
* .RSSPEC
* .UNSMRY
* .SMSPEC
* .EGRID
* .INIT

Within the `DeepField` framework it is recommended to use the HDF5 format
to speed up data load and dump in Python-friendly data formats. In this
case all data are contained in a single .HDF5 file. At any point the model
can be exported back into .DATA text and binary files to ensure a compatibility
with conventional software.

## Citing

Plain text
```
E. Illarionov, P. Temirchev, D. Voloskov, R. Kostoev, M. Simonov, D. Pissarenko, D. Orlov, D. Koroteev, 2022. End-to-end neural network approach to 3D reservoir simulation and adaptation. J. Pet. Sci. Eng. 208, 109332. https://doi.org/10.1016/j.petrol.2021.109332
```

BibTex
```
@article{ILLARIONOV2022109332,
author = {E. Illarionov and P. Temirchev and D. Voloskov and R. Kostoev and M. Simonov and D. Pissarenko and D. Orlov and D. Koroteev},
title = {End-to-end neural network approach to 3D reservoir simulation and adaptation},
journal = {Journal of Petroleum Science and Engineering},
volume = {208},
pages = {109332},
year = {2022},
issn = {0920-4105},
doi = {https://doi.org/10.1016/j.petrol.2021.109332},
url = {https://www.sciencedirect.com/science/article/pii/S0920410521009827}
}
```
