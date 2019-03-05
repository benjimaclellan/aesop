# ASOPE: Automated Search of Optical Processing Experiments
ASOPE, or Automated Search of Optical Processing Experiments, simulates and optimizes optical processing experiments. 

## Getting Started

This software will simulate a photonic experiment setup, and can optimize the parameters on each component for a desired output. A brief outline of how it works:

An experiment is described by a directed graph (implemented with `networkx`). For now, you define the setup by listing the components as graph nodes, and the connections between them as directed edges. Each component has a set type of transformation in either in the temporal or frequency domain, with a certain number of paramters. These can be changed, if need be, but it is important to ensure that how the parameters are unpacked is consistent. Both the experiment and each component are defined as custom classes.

The pulse information is also represented as a custom class, and also stored the fitness function that will be opimized for in the genetic algorithm. 


### Prerequisites

All packages needed for this software are open-source, standard Python packages. The main packages and their uses are:

`deap` - Distributed Evolutionary Algorithms in Python, used for the Genetic Algorithm

`networkx` - Graph and network manipulation package, used for representing an experimental setup

`multiprocess` - Multiprocessing in Python, used to improve computation time with the Genetic Algorithm

`peakutils` - Peak detection for various evaluation, such as measuring the repetition rate of a pulse train

`numpy` - Standard scientific/numerical package in Python

`matplotlib` - Plotting and visualization

### Installing

Using Anaconda, a scientific computing environment for Python, is recommended. First, install the Anaconda environment for your system, [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/). Spyder is a useful Python IDE, but the command-line interface for installing Python packages is the main use.

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


## Directory Structure
```
asope/
    README.md
    LICENSE.md
    setup.py
    asope/
        various_experiments.py
        single_test.py 
        asope_main.py
        results/
        assets/
            __init__.py
            functions.py
            environment.py
            components.py
            classes.py
        optimization/
            __init__.py
            geneticalgorithminner.py
            gradientdescent.py
```



## Contributing
Submit a push request to the Github repository.


## Authors
* **Benjamin MacLellan** - [Email](benjamin.maclellan@emt.inrs.ca)

## License
Please see [LICENSE.md](../LICENSE.md)

## Acknowledgments
Thank you to all the members of the Nonlinear Photonics Group at INRS-EMT, as well as all the authors of the packages on which this project is founded

 
