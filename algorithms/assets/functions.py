"""


"""

import numpy as np
import pandas as pd

def logbook_update(generation, population, log, log_metrics, runtime=-1, verbose=False):
    """ updates the statistics log book. Runtime -1 means that the runtime of a generation was not calculated """
    scores = [score for score, _ in population]
    tmp_log = {'gen':generation, 'runtime':runtime}
    for metric, func in log_metrics.items():
        tmp_log[metric] = func(scores)
    log.loc[generation] = tmp_log
    if verbose:
        print(tmp_log)


def logbook_initialize():
    log_metrics = {'n_pop': len, 'avg': np.mean, 'var': np.var, 'max': np.max, 'min': np.min}
    log = pd.DataFrame(columns=['gen'] + list(log_metrics.keys()) + ['runtime'])
    return log, log_metrics