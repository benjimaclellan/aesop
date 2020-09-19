import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config.config as config

from lib.functions import InputOutput

plt.close('all')
if __name__ == '__main__':
    io = InputOutput(directory='20200915_ff42_batch_test', verbose=True)
    io.init_load_dir(sub_path=None)
    n_gens, n_pop = 2, 16

    for gen in range(n_gens):
        for ind in range(n_pop):
            graph = io.load_graph(f'gen{gen}_{ind}.pkl')
            parameters, *_ = graph.extract_parameters_to_list()
            state = graph.measure_propagator(-1)

            fig, ax = plt.subplots(2,1, figsize=[4, 3])
            graph.draw(ax=ax[0], legend=True)
            ax[1].plot(np.power(np.abs(state),2))
    plt.show()
