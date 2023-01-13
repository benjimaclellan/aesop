"""
Test of topology optimization routines
"""

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

from lib.functions import InputOutput
from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.evolver import HessianProbabilityEvolver
from simulator.fiber.node_types_subclasses import *
from simulator.fiber.evaluator_subclasses.evaluator_pulserep import PulseRepetition
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from lib.functions import parse_command_line_args
from scripts.setups.setup_pulse_rep import run_experiment

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    run_settings = [
        dict(pq=(4, 3), pulse_width=9e-12, rep_t=126.666e-12, peak_power=1.0),
        # dict(pq=(1, 2), pulse_width=3e-12, rep_t=1/10.0e9, peak_power=1.0),
        # dict(pq=(2, 1), pulse_width=3e-12, rep_t=1/10.0e9, peak_power=1.0),
        # dict(pq=(1, 3), pulse_width=3e-12, rep_t=1/10.0e9, peak_power=1.0),
    ]

    ga_opts = {'n_generations': 18,
               'n_population': 18,
               'n_hof': 20,
               'verbose': options_cl.verbose,
               'num_cpus': 18
               }

    propagator = Propagator(window_t=126.666e-12 * 12, n_samples=2**15, central_wl=1.55e-6)
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
                                             pulse_width=pulse_width,
                                             rep_t=rep_t * (p / q),
                                             peak_power=peak_power * (p / q))
        evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

        evolver = HessianProbabilityEvolver(verbose=False)
        try:
            run_experiment(evaluator, propagator, io, evolver, ga_opts, input_laser, param_opt='L-BFGS+GA')
        except RuntimeError as e:
            print(f"Error caught, moving to next: {e}")