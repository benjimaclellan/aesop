from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter
import numpy as np


POTENTIALS = {'splitters':[PowerSplitter, FrequencySplitter],
              'nonsplitters':[Fiber, AWG, PhaseModulator, WaveShaper]}
#POTENTIALS = {'splitters':[],
#              'nonsplitters':[Fiber, PhaseModulator, WaveShaper]}

#goal = 'Talbot'
goal = 'PA-AWG'

if goal == 'Talbot':
    FITNESS_VARS = {'profile':'gauss', 'p':3, 'q':1}
elif goal == 'PA-AWG':
    func = lambda t: np.sin(2*np.pi*1e9*t)
    FITNESS_VARS = {'profile':'cw', 'func':func}