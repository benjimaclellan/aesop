from assets.components import Fiber, AWG, PhaseModulator, WaveShaper, PowerSplitter, FrequencySplitter, AmplitudeModulator
import numpy as np
"""
Things that need to be fixed
- all simulation parameters are based on pulsed excitation. make more general
- AWG was based on pulsed source - fix this (it has been removed for now)
- consider the carrier frequency, as this affects all components (cannot have unlimited )


ideas:
- instead of looping through many experimental setups, we constrain it to ones that make sense (start cw, create side bands via modulation, filter with pulse shaper, add dispersion). This will greatly limit the complexity of the search space
- Friday April 16th, try to make a better model of a waveshaper so that we can use a phasemodulator and waveshaper to make sine wave (two side bands, filter, beat)
"""

POTENTIALS = {'splitters':[PowerSplitter, FrequencySplitter],
              'nonsplitters':[Fiber, PhaseModulator, WaveShaper, AmplitudeModulator]}
#POTENTIALS = {'splitters':[],
#              'nonsplitters':[Fiber, PhaseModulator, WaveShaper]}

#goal = 'Talbot'
goal = 'PA-AWG'

if goal == 'Talbot':
    FITNESS_VARS = {'profile':'gauss', 'p':3, 'q':1}
elif goal == 'PA-AWG':
    func = lambda t: np.sin(2*np.pi*1e9*t)
    FITNESS_VARS = {'profile':'cw', 'func':func}