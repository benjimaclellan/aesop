"""


"""

import numpy as np
import pandas as pd

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