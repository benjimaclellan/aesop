"""
Copyright Benjamin MacLellan

The inner optimization process for the Automated Search for Optical Processing Experiments (ASOPE). This uses a genetic algorithm (GA) to optimize the parameters (attributes) on the components (nodes) in the experiment (graph).

"""

#%% this allows proper multiprocessing (overrides internal multiprocessing settings)
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

#%% import public modules
import time
import matplotlib.pyplot as plt
#import multiprocess as mp

import copy
import autograd.numpy as np
import scipy.io as sio
import scipy.signal as sig

#%% import custom modules
from assets.functions import extractlogbook, save_class
from assets.functions import FFT, IFFT, P, PSD

from classes.environment import OpticalField_PPLN
from classes.components import DelayLine, Test
# from classes.experiment import Experiment

import random
from deap import base, creator, tools, algorithms
import seaborn
import pandas as pd

plt.close("all")


def fsr_value(psd, peak_inds, peak_widths):
    # fitness = np.mean(np.diff(peak_inds)) / (np.std(np.diff(peak_inds) + 1)) #* np.mean(psd)
    fitness = np.std(psd)
    return fitness


def q_factor(psd, peak_inds, peak_widths):
    # fitness = 1/np.mean(peak_widths)
    fitness = np.mean(peak_widths)
    return fitness

def get_peaks(psd):
    # psd = PSD(field, env.dt, env.df)
    peak_array = np.squeeze(psd / max(psd))
    peak_inds, prop = sig.find_peaks(peak_array,
                                     height=None, threshold=None, distance=None, prominence=0.5, width=None,
                                     wlen=None,
                                     rel_height=None, plateau_size=None)
    (peak_widths, width_heights, *_) = sig.peak_widths(peak_array, peak_inds, rel_height=0.5,
                                                       prominence_data=None,
                                                       wlen=None)
    return peak_inds, peak_widths

#%%

def create_individual(delayline):
    at = delayline.newattribute()
    return at

def fitness(at, env, delayline):
    # Fonseca–Fleming function
    f1 = (1 - np.exp(-np.sum( (np.array(at) - 1/np.sqrt(len(at)))**2 ) ))
    f2 = (1 - np.exp(-np.sum( (np.array(at) + 1/np.sqrt(len(at)))**2 ) ))

    # print(at, f1, f2)
    # delayline.at = at
    # print(at)
    # field = delayline.simulate(env, env.field0)
    #
    # psd = PSD(field, env.dt, env.df)
    #
    # peak_inds, peak_widths = get_peaks(psd)
    #
    # fsr = fsr_value(psd, peak_inds, peak_widths)
    # q = q_factor(psd, peak_inds, peak_widths)
    # return fsr, q
    return f1, f2
#%%
def run_ea(toolbox, stats=None, verbose=False):
    pop = toolbox.population(n=toolbox.pop_size)
    pop = toolbox.select(pop, len(pop))
    return algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.pop_size,
                                     lambda_=toolbox.pop_size,
                                     cxpb=1-toolbox.mut_prob,
                                     mutpb=toolbox.mut_prob,
                                     stats=stats,
                                     ngen=toolbox.max_gen,
                                     verbose=verbose)

#%% initialize our input pulse, with the fitness function too
env = OpticalField_PPLN(n_samples=2**16, window_t=1e-9, lambda0=1.55e-6, bandwidth=[1.54e-6, 1.56e-6])
delayline = Test() #DelayLine()
# at = delayline.newattribute()
# delayline.at = at
# field = delayline.simulate(env, env.field0)

##
creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Attribute generator
toolbox = base.Toolbox()
toolbox.register("attribute", delayline.newattribute)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=delayline.LOWER, up=delayline.UPPER, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=delayline.LOWER, up=delayline.UPPER, eta=20.0, indpb=1.0/2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", fitness, env=env, delayline=delayline)

toolbox.pop_size = 500
toolbox.max_gen = 25
toolbox.mut_prob = 0.2

if __name__ == '__main__':
    res, _ = run_ea(toolbox, stats=None, verbose=True)
    fronts = tools.emo.sortLogNondominated(res, len(res), first_front_only=False)

    plot_colors = seaborn.color_palette("Set1", n_colors=10)
    fig, ax = plt.subplots(1, figsize=(4, 4))
    for i, inds in enumerate(fronts):
        par = [toolbox.evaluate(ind) for ind in inds]
        df = pd.DataFrame(par)
        df.plot(ax=ax, kind='scatter', label='Front ' + str(i + 1),
                x=df.columns[0], y=df.columns[1],
                color=plot_colors[i])
    inds = fronts[0]
    df = pd.DataFrame([toolbox.evaluate(ind) for ind in inds])
    df.sort_values(by=[0], inplace=True)
    ax.plot(df[0], df[1], color='black')

    plt.xlabel('$f_1(\mathbf{x})$')
    plt.ylabel('$f_2(\mathbf{x})$')