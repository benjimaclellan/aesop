# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# various imports
import matplotlib.pyplot as plt
import psutil
from scipy.signal import square
import numpy as np

from simulator.fiber.node_types_subclasses import *

from config import config as configuration
from lib.functions import InputOutput

from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evolver import HessianProbabilityEvolver

from simulator.fiber.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from lib.functions import parse_command_line_args

from scripts.setups.setup_rfawg import run_experiment

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    propagator = Propagator(window_t=1e-9, n_samples=2**14, central_wl=1.55e-6)

    def pattern1(t, amplitude, frequency):
        sig=amplitude * ((square(2 * np.pi * t * frequency, 0.5) + 1) / 2 +
                         (square(2 * np.pi * t * frequency + 0.5 * np.pi, 0.5) + 1) / 2) / 2
        return sig

    evaluators = [RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_amplitude=0.02,
                                                   target_waveform='square',),
                  RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_amplitude=0.02,
                                                   target_waveform='saw', ),
                  RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_amplitude=0.02,
                                                   target_waveform='tri',),
                  RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_amplitude=0.02,
                                                   target_waveform='other-saw',),
                  ]

    for evaluator in evaluators:
        io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)

        ga_opts = {'n_generations': 12,
                   'n_population': 12,
                   'n_hof': 8,
                   'verbose': options_cl.verbose,
                   'num_cpus': psutil.cpu_count()-1}

        evolver = HessianProbabilityEvolver(verbose=False)

        run_experiment(evaluator, propagator, io, evolver, ga_opts)
