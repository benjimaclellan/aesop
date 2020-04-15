"""


"""

import numpy as np
import pandas as pd

def logbook_update(generation, population, log, log_metrics, verbose=False):
    """ updates the statistics log book """
    scores = [score for score, _ in population]
    tmp_log = {'gen':generation}
    for metric, func in log_metrics.items():
        tmp_log[metric] = func(scores)
    log.loc[generation] = tmp_log
    if verbose:
        print(tmp_log)


def logbook_initialize():
    log_metrics = {'n_pop': len, 'avg': np.mean, 'var': np.var, 'max': np.max, 'min': np.min}
    log = pd.DataFrame(columns=['gen'] + list(log_metrics.keys()))
    return log, log_metrics