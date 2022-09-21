# aesop
AESOP (Automatic Exploration of Systems for Photonics & Optics) is an inverse design framework 
for automaticallty designing photonic signal processing systems. 
The software provides:
* differentiable photonic circuit solver, with library of fiber components
* circuit analysis techniques, including noise simulation and gradient-based sensitivity analysis
* design of both system topology and parameters, using topology search methods and gradient-based parameter optimization

## Getting Started
AESOP provides a framework for the inverse design of photonics and optics systems. 
Searches the design space for both system topology and parameter optimization.


## Installing

To install the ASEOP package, clone the repository from [Github](https://github.com/) at [https://github.com/benjimaclellan/ASOPE.git](https://github.com/benjimaclellan/ASOPE.git). 
Install dependencies via  `pip install -r requirements.txt` file. 

### Dependencies

`networkx` - Graph and network manipulation package, used for representing an experimental setup

`multiprocess` - Multiprocessing in Python, used to improve computation time with the Genetic Algorithm

`numpy` - Standard scientific/numerical package in Python

`scipy` - Scientific computing functions, including minimization

`matplotlib` - Plotting and visualization

`autograd` - Automatic differentiation of Python functions

## New in v.3.0
Overhaul of the circuit simulator, 
including noise analysis, 
new photonics components in the library, 
improved topology optimization, 
and more methods of parameter optimization. 

## Citing
* **Benjamin MacLellan**, benjamin.maclellan [AT] inrs.ca
* **Piotr Roztocki**
* **Julie Belleville**
* **Kaleb Ruscitti**

## License
Please see [LICENSE.md](../LICENSE.md)


 
