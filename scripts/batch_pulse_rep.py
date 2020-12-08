"""
Test of topology optimization routines

TODO: HoF should be taken prior to speciation!!!!!
"""

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
import numpy as np
import config.config as config

from lib.functions import InputOutput

from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator
from problems.example.assets.functions import psd_, power_, fft_, ifft_

from problems.example.evolver import ProbabilityLookupEvolver

from problems.example.node_types import TerminalSource, TerminalSink

from problems.example.evaluator_subclasses.evaluator_pulserep import PulseRepetition

from problems.example.node_types_subclasses.inputs import PulsedLaser, ContinuousWaveLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import DispersiveFiber, PhaseModulator, WaveShaper, DelayLine
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from algorithms.topology_optimization import topology_optimization, plot_hof, save_hof

from lib.functions import parse_command_line_args

from scripts.setups.setup_pulse_rep import run_experiment

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    run_settings = [
        dict(pq=(1, 2), pulse_width=0.3e-9, rep_t=20.0e-9, peak_power=1.0),
        dict(pq=(2, 1), pulse_width=0.3e-9, rep_t=20.0e-9, peak_power=1.0),
        dict(pq=(3, 1), pulse_width=0.5e-9, rep_t=20.0e-9, peak_power=1.0),
        dict(pq=(1, 3), pulse_width=0.1e-9, rep_t=20.0e-9, peak_power=1.0),
    ]

    ga_opts = {'n_generations': 16,
               'n_population': 16,
               'n_hof': 6,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=300e-9, n_samples=2**14, central_wl=1.55e-6)
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)

    for i, settings in enumerate(run_settings):

        pulse_width, rep_t, peak_power = (settings['pulse_width'], settings['rep_t'], settings['peak_power'])
        p, q = settings['pq']

        input_laser = PulsedLaser(parameters_from_name={'pulse_width': pulse_width,
                                                        'peak_power': peak_power,
                                                        't_rep': rep_t,
                                                        'pulse_shape': 'gaussian',
                                                        'central_wl': 1.55e-6,
                                                        'train': True})
        input_laser.node_lock = True
        input_laser.protected = True

        input = input_laser.get_pulse_train(propagator.t,
                                            pulse_width=pulse_width,
                                            rep_t=rep_t,
                                            peak_power=peak_power)
        target = input_laser.get_pulse_train(propagator.t,
                                             pulse_width=pulse_width * (p / q),
                                             rep_t=rep_t * (p / q),
                                             peak_power=peak_power * (p / q))
        evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

        evolver = ProbabilityLookupEvolver(verbose=False)

        run_experiment(evaluator, propagator, io, evolver, ga_opts, input_laser, param_opt='L-BFGS+GA')
