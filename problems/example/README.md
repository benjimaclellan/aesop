# Kirby, v3.0
Originally named ASOPE (short for _Automated Search of Optical Processing Experiments_), the updated version has the name changed to Kirby.
Kirby is a Python package for the inverse design of optical systems, and includes custom simulation software.

## New in v.3.0
Complete overhaul of the simulation library - which has added layers of abstraction and new extensibility, so that new simulators for other optical degrees-of-freedom
can be added later. The following parts of the README have not been updated for v3.0.

## Getting Started

This package aims to design an optical system which accomplishes an user-defined goal. 
A system is described by a set of components, connected in a specific way and with specific control variables.




### Prerequisites

All packages needed for this software are open-source, standard Python packages. The main packages and their uses are:

`deap` - Distributed Evolutionary Algorithms in Python, used for the Genetic Algorithm

`networkx` - Graph and network manipulation package, used for representing an experimental setup

`multiprocess` - Multiprocessing in Python, used to improve computation time with the Genetic Algorithm

`peakutils` - Peak detection for various evaluation, such as measuring the repetition rate of a pulse train

`numpy` - Standard scientific/numerical package in Python

`matplotlib` - Plotting and visualization

See [`requirements.txt`](../requirments.txt) for full list of all dependencies.

### Installing

Using Anaconda, a scientific computing environment for Python, is recommended. First, install the Anaconda environment for your system, [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/). Spyder is a useful Python IDE, but the command-line interface for installing Python packages is the main use.

To install the ASOPE package, clone the repository from [Github](https://github.com/) at [https://github.com/benjimaclellan/ASOPE.git](https://github.com/benjimaclellan/ASOPE.git). 

All the prerequisite packages can be install from the `requirements.txt` file. 

You can install all packages via pip.

## Running the tests

There are four main scripts which use different levels of the package.

* [`asope/simulation.py`](../asope/simulation.py) 
Simulates a defined optical source through a defined optical system. 
Control parameters can be defined or randomly chosen within bounds. 
No optimization occurs.
Sensitivity analysis can be applied at the output.

* [`asope/testing.py`](../asope/testing.py) 
Scratch document for testing various things (constantly changing)

* [`asope/main_paramopt.py`](../asope/main_paramopt.py) 
Optimizes the control parameters for a defined topology, objective function, and optical source.

* [`asope/main_asope.py`](../asope/main_asope.py) 
Full ASOPE algorithm, optimizing both topology and control parameters for a given optical source and objective function.

## Configuration

For each optimization run, a corresponding configuration file must be used to specify various hyperparameters.


## Known Bugs
* Gradient descent via autograd does not currently work due to choice of step size.
Consider a more sophisticated gradient method such as ADAM. 

## Contributing
Submit a push request to the Github repository.

## Authors
* **Benjamin MacLellan** - [Email](benjamin.maclellan@emt.inrs.ca)

## License
Please see [LICENSE.md](../LICENSE.md)

## Acknowledgments

 
