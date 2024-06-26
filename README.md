<h1 align="center">
 aesop
</h1>

<div align="center">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
[![LPR Paper](https://img.shields.io/badge/doi-10.1002/lpor.202300500-red)](https://onlinelibrary.wiley.com/doi/10.1002/lpor.202300500)

</div>

<p align="center" style="font-size:20px">
    AESOP (Autonomous Exploration of Systems for Optics and Photonics) 
    is an inverse-design framework for photonic systems and circuits. 
</p>

## Features
* differentiable photonic circuit solver, with a library of standard, commercially available fiber components
* circuit analysis techniques, including noise simulation and gradient-based sensitivity analysis
* design of both system topology and parameters, using topology search methods and gradient-based parameter optimization

## Installation
See the paper in Laser & Photonics Reviews [here](https://onlinelibrary.wiley.com/doi/10.1002/lpor.202300500)!

To install AESOP, clone the [repository](https://github.com/benjimaclellan/aesop) 
and install dependencies via `pip install -r requirements.txt` file.


## Example 

```py
from lib.graph import Graph
from simulator import *

propagator = Propagator(window_t=10/12e9, n_samples=2**14, central_wl=1.55e-6)
evaluator = RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_waveform='saw')

nodes = {'source': TerminalSource(),
         0: VariablePowerSplitter(),
         1: VariablePowerSplitter(),
         2: VariablePowerSplitter(),
         3: VariablePowerSplitter(),
         'sink': TerminalSink()}

edges = {('source', 0): ContinuousWaveLaser(),
         (0, 1): PhaseModulator(),
         (1, 2): WaveShaper(),
         (2, 3): OpticalAmplifier(),
         (3, 'sink'): Photodiode(),
         }

graph = Graph.init_graph(nodes, edges)
graph.update_graph()
graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

```

## Citing
```
@article{MacLellan2024,
  title = {Inverse Design of Photonic Systems},
  volume = {18},
  ISSN = {1863-8899},
  url = {http://dx.doi.org/10.1002/lpor.202300500},
  DOI = {10.1002/lpor.202300500},
  number = {5},
  journal = {Laser &amp; Photonics Reviews},
  publisher = {Wiley},
  author = {MacLellan,  Benjamin and Roztocki,  Piotr and Belleville,  Julie and Romero Cortés,  Luis and Ruscitti,  Kaleb and Fischer,  Bennet and Azaña,  José and Morandotti,  Roberto},
  year = {2024},
  month = feb 
}
```




 
