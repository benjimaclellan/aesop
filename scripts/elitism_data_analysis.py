"""
Test of topology optimization routines
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pickle

def display_log_data(directory, xaxis='generation', yaxis=['mean', 'minimum'], categories=[(8, 0), (8, 0.1), (8, 0.3)], show=True, save=True):
    """
    """
    if xaxis not in ['generation', 'time']:
        raise ValueError('Generation (generation) and runtime (time) are the only acceptable xaxis variables')
    
    colour_map = ListedColormap(sns.color_palette(palette='Paired'))
    fig, ax = plt.subplots()
    for i, (pop_size, elitism_ratio) in enumerate(categories):
        # setup averaging
        sum_df = None

        # setup an ace aesthetic
        colour = colour_map(i / len(categories))
        
        # setup directories
        if type(elitism_ratio) is int:
            subdirectory = f'popSize{pop_size}_elitismRatio{elitism_ratio}'
        else:
            subdirectory = f"popSize{pop_size}_elitismRatio{int(elitism_ratio)}_{str(elitism_ratio).split('.', 1)[1]}"
        
        sub_subdirectory = os.listdir(f'{directory}/{subdirectory}')
        
        for dir in sub_subdirectory:
            with open(f'{directory}/{subdirectory}/{dir}/log.pkl', 'rb') as handle:
                    log = pickle.load(handle)

                    if sum_df is None:
                        sum_df = log
                    else:
                        sum_df = sum_df.add(log, fill_value=0)
        sum_df = sum_df / len(sub_subdirectory)
        if 'mean' in yaxis:
            ax.plot(sum_df[xaxis], sum_df['mean'], ls='-', color=colour, label=f'avg mean, pop: {pop_size}, elitism_ratio: {elitism_ratio}')
        if 'minimum' in yaxis:
            ax.plot(sum_df[xaxis], sum_df['minimum'], ls=':', color=colour, label=f'avg min, pop: {pop_size}, elitism_ratio: {elitism_ratio}')
        if 'maximum' in yaxis:
            ax.plot(sum_df[xaxis], sum_df['maximum'], ls='--', color=colour, label=f'avg max, pop: {pop_size}, elitism_ratio: {elitism_ratio}')

    ax.legend()
    ax.set_xlabel(xaxis)
    ax.set_ylabel('Error function')
    plt.title(f'Comparing fitness across elitism values (across {len(sub_subdirectory)} runs)')

    if show:
        plt.show()

if __name__ == "__main__":
    display_log_data('scripts/elitism_test_data')
    

