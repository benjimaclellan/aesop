import sys
import pathlib
sys.path.append('..')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config.config as config

from lib.functions import InputOutput

plt.close('all')
if __name__ == '__main__':

    directory = None
    # directory = input()
    if not directory:
        directory = r'C:\Users\benjamin\Documents\INRS - Projects\asope_data\testing\20200920_3D9f_run'
    io = InputOutput(directory=directory, verbose=True)
    io.init_load_dir(sub_path=None)

    ga_opts = io.load_json('ga_opts.json')
    propagator = io.load_object('propagator.pkl')
    evaluator = io.load_object('evaluator.pkl')
    evolver = io.load_object('evolver.pkl')

    for i in range(ga_opts['n_hof']):
        graph = io.load_object(f'graph_hof{i}.pkl')
        parameters, *_ = graph.extract_parameters_to_list()
        state = graph.measure_propagator(-1)

        fig, ax = plt.subplots(2,1, figsize=[4, 3])
        ax[0].set_title(graph.score)
        graph.draw(ax=ax[0], legend=True)
        ax[1].plot(propagator.t, evaluator.target, label='Target')
        ax[1].plot(propagator.t, np.power(np.abs(state),2), label='Solution')

    plt.show()
