# ASOPE: Automated Search of Optical Processing Experiments
ASOPE, or Automated Search of Optical Processing Experiments, simulates and optimizes optical processing experiments. 

## Getting Started
Using Anaconda, a scientific computing environment for Python, is recommended. 

### Prerequisites

All packages needed for this software are open-source, standard Python packages. The main packages and their uses are:
`deap` - Distributed Evolutionary Algorithms in Python, used for the Genetic Algorithm
`networkx` - Graph and network manipulation package, used for representing an experimental setup
`multiprocess` - Multiprocessing in Python, used to improve computation time with the Genetic Algorithm
`peakutils` - Peak detection for various evaluation, such as measuring the repetition rate of a pulse train
`numpy` - Standard scientific/numerical package in Python
`matplotlib` - Plotting and visualization

### Installing

First, install the Anaconda environment for your system, [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/). Spyder is a useful Python IDE, but the command-line interface for installing Python packages is the main use.

To install the ASOPE package, clone the repository from [Github](https://github.com/) at [https://github.com/benjimaclellan/ASOPE.git](https://github.com/benjimaclellan/ASOPE.git). 

All the prerequisite packages can be install from the `setup.py` file. First install the `setuptools` package using the command

```
conda install setuptools
```

Once the installation is successful, navigate to the ASOPE directory and install the dependencies with

```
python setup.py install
```

If no error messages are thrown, you should be good to go.


## Running the tests

First, to check that the basics of simulating an experiment work, run [`single_test.py`](../asope/single_test.py). You should see a graph structure outlining the experimental setup, and a plot of the pulse output. Next, you can run [`asope_main.py`](../asope/asope_main.py) to have the Genetic Algorithm optimize the parameters for a single setup.


## Contributing
Submit a push request to the Github repository.


## Authors
* **Benjamin MacLellan** - [Email](benjamin.maclellan@emt.inrs.ca)

## License
Please see [LICENSE.md](../LICENSE.md)

## Acknowledgments
Thank you to all the members of the Nonlinear Photonics Group at INRS-EMT, as well as all the authors of the packages on which this project is founded

 
