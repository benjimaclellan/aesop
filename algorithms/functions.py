"""


"""
import seaborn as sns
import autograd.numpy as np
import pandas as pd
import time

# for plotting
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lib.autodiff_helpers import unwrap_arraybox_val


def logbook_update(generation, population, log, log_metrics, time=-1, best=None, verbose=False):
    """ updates the statistics log book. Runtime -1 means that the runtime of a generation was not calculated """
    scores = [score for score, _ in population]
    tmp_log = {'generation':generation, 'time':time, 'best':best}

    for metric, func in log_metrics.items():
        tmp_log[metric] = func(scores)
    log.loc[generation] = tmp_log
    if verbose: print(tmp_log)


def logbook_initialize():
    log_metrics = {'n_population': len, 'mean': np.mean, 'variance': np.var, 'maximum': np.max, 'minimum': np.min}
    log = pd.DataFrame(columns=['generation', 'time', 'best'] + list(log_metrics.keys()))
    return log, log_metrics


class ParameterOptimizationLogger():
    """
    Logger for parameter optimizations
    """
    def __init__(self, log_metrics=None, valid_algorithms=None, algorithm_colours=None):
        """

        :param log_metrics: dictionary with (key=metric name (string), value=function to extract metric)
                            Note that the functions to extract the metric don't take in the population! They take in a single new score
        :param valid algorithms: list or set of strings describing valid algorithms
        :param algorithm_colours: a dictionary with key=algorithm name, value=colour for the algorithm when plotted
        """
        self.start_time = None
        self.valid_algorithms = list(['L-BFGS', 'PSO', 'CMA', 'ADAM', 'GA'])
        self.current_algorithm = None
        self.current_population = -1
        self.log_number = 0 # number of times we've logged. Equivalent to number of iterations for gradient descent algs, to num of individuals updated for other algs
        self.row_offset = 0
    
        self.scores_in_this_generation = np.array([])

        if log_metrics is None:
            self.log_metrics = {'algorithm': lambda score: self.current_algorithm,\
                                'process runtime (s)': lambda score: time.process_time() - self.start_time, \
                                'gen_or_iter': lambda score: self.log_number // self.current_population, \
                                'n_population': lambda score: self.current_population, \
                                'minimum': self._update_min, \
                                'mean': self._update_mean, \
                                'maximum': self._update_max, \
                                'variance': self._update_variance
                                }
        else:
            self.log_metrics = log_metrics
        
        if algorithm_colours is None:
            self.colour_map = ListedColormap(sns.color_palette(palette='Accent'))
            self.algorithm_colours = {}
            for (i, alg) in enumerate(self.valid_algorithms):
                self.algorithm_colours[alg] = self.colour_map(i / len(self.valid_algorithms))
        else:
            self.algorithm_colours = algorithm_colours
        
        self.dataframe = pd.DataFrame(columns=list(self.log_metrics.keys()))
    
    def get_log(self):
        return self.dataframe
    
    def start_logger_time(self):
        self.start_time = time.process_time()
    
    def set_optimization_algorithm(self, alg_str, pop_size=1):
        """

        :param alg_str: string corresponding to the algorithm which we're loggin for
        :param pop_size: MUST BE SPECIFIED FOR ANY ALGORITHM WITH POPULATIONs
        """
        self.row_offset = self.log_number // self.current_population
        self.log_number = 0
        if alg_str not in self.valid_algorithms:
            raise ValueError(f'Choice of algorithm {alg_str} not in the list of valid logged algorithms: {self.valid_algorithms}')

        self.current_algorithm = alg_str
        self.current_population = pop_size
        self.scores_in_this_generation = np.zeros(pop_size)
    
    def log_score(self, score):
        score = unwrap_arraybox_val(score)

        num_in_gen = self.log_number % self.current_population
        self.scores_in_this_generation[num_in_gen] = score
        
        for metric in self.log_metrics.keys():
            self.dataframe.loc[self._last_row_num, metric] = self.log_metrics[metric](score)
        
        self.log_number += 1

        if self.log_number % self.current_population == 0:
            self.scores_in_this_generation = np.zeros(self.current_population)
    
    def display_log(self, fig=None, ax=None, show=True, xaxis='process runtime (s)', yaxis='minimum'):
        if ax is None:
            fig, ax = plt.subplots()
        
        algs_in_use = self.dataframe.algorithm.unique()

        for alg in algs_in_use:
            filtered_df = self.dataframe.loc[self.dataframe['algorithm'] == alg]
            ax.scatter(filtered_df[xaxis], filtered_df[yaxis], color=self.algorithm_colours[alg], label=alg)

        ax.legend()
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        if show:
            plt.show()
        return fig, ax

    @staticmethod
    def display_log_comparison(logger_dict, title='', fig=None, ax=None, show=True, xaxis='process runtime (s)', yaxis='minimum'):
        """

        :param logger_dict: ("data label", logger obj)
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        print(f'logger dict items: {logger_dict.items()}')
        print(f'enumerate logger dict items: {enumerate(logger_dict.items())}')
        for i, (label, logger) in enumerate(logger_dict.items()):
            data = logger.get_log()
            ax.plot(data[xaxis], data[yaxis], label=label, color=logger.colour_map(i / len(logger_dict)))
        
        ax.legend()
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        plt.title(title)
        if show:
            plt.show()
        
        return fig, ax

    @property
    def _last_row_num(self):
        return self.row_offset + self.log_number // self.current_population

    def _update_mean(self, score):
        return np.mean(self.scores_in_this_generation)
    
    def _update_variance(self, score):
        return np.var(self.scores_in_this_generation) 
    
    def _update_min(self, score):
        """
        Doesn't use np.min on self.scores_in_this_generation,
        because scores could be negative, and this doesn't work with our initialization as array of zeros
        """
        if self.log_number % self.current_population == 0:
            return score
        
        return min(score, self.dataframe.loc[self._last_row_num, 'minimum'])

    def _update_max(self, score):
        """
        Doesn't use np.min on self.scores_in_this_generation,
        because scores could be negative, and this doesn't work with our initialization as array of zeros
        """
        if self.log_number % self.current_population == 0:
            return score
        
        return max(score, self.dataframe.loc[self._last_row_num, 'maximum'])
