# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform
import copy

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# various imports
import matplotlib.pyplot as plt
import psutil

import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evolver import ProbabilityLookupEvolver

from problems.example.node_types import TerminalSource, TerminalSink

from problems.example.evaluator_subclasses.evaluator_rfawg import RadioFrequencyWaveformGeneration

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

from lib.functions import parse_command_line_args

from scripts.setups.setup_rfawg import run_experiment

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    propagator = Propagator(window_t=1e-9, n_samples=2**14, central_wl=1.55e-6)

    evaluators = [RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_amplitude=0.02,
                                                   target_waveform='square',),
                  RadioFrequencyWaveformGeneration(propagator, target_harmonic=12e9, target_amplitude=0.02,
                                                   target_waveform='saw', ),
                  RadioFrequencyWaveformGeneration(propagator, target_harmonic=24e9, target_amplitude=0.02,
                                                   target_waveform='square', ),
                  RadioFrequencyWaveformGeneration(propagator, target_harmonic=36e9, target_amplitude=0.02,
                                                   target_waveform='saw', ),
                  RadioFrequencyWaveformGeneration(propagator, target_harmonic=10e9, target_amplitude=0.04,
                                                   target_waveform='saw', )
                  ]

    for evaluator in evaluators:
        io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)

        ga_opts = {'n_generations': 16,
                   'n_population': 16,
                   'n_hof': 6,
                   'verbose': options_cl.verbose,
                   'num_cpus': psutil.cpu_count()-1}

        evolver = ProbabilityLookupEvolver(verbose=False)

        run_experiment(evaluator, propagator, io, evolver, ga_opts)
