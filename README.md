# Automated Search of Optical Processing Experiments
ASOPE is a Python package for the inverse design of optical systems in the time-energy degree-of-freedom.
This README is a ever-changing document - please contact the authors for current status.

## New in v.3.0
Complete overhaul of the simulation library - which has added layers of abstraction and new extensibility.
New noise analysis, new components in the library, improved topology optimization, more methods of parameter optimization. 
## Getting Started
This package aims to design optical systems which accomplishes an user-defined goal. 
A system is described by a set of components, connected in a specific way and with specific control variables.

### Prerequisites

See [`requirements.txt`](../requirments.txt) for full, more up-to-date list of all dependencies.
Major packages used are:

`networkx` - Graph and network manipulation package, used for representing an experimental setup

`multiprocess` - Multiprocessing in Python, used to improve computation time with the Genetic Algorithm

`numpy` - Standard scientific/numerical package in Python

`scipy` - Scientific computing functions, including minimization

`matplotlib` - Plotting and visualization

`autograd` - Automatic differentiation of Python functions

### Installing

To install the ASOPE package, clone the repository from [Github](https://github.com/) at [https://github.com/benjimaclellan/ASOPE.git](https://github.com/benjimaclellan/ASOPE.git). 
All the prerequisite packages can be install from the `requirements.txt` file. 
You can install all packages via pip, or using `pip install requirements.txt`. 
A virtual environment is recommended.

## Running the tests
Not implemented.

## Authors
* **Benjamin MacLellan** - [Email](benjamin.maclellan@emt.inrs.ca)
* **Piotr Roztocki**
* **Julie Belleville**
* **Kaleb Ruscitti**

## License
Please see [LICENSE.md](../LICENSE.md)


 
