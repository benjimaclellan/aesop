"""
Copyright Benjamin MacLellan & Mario Chemnitz
Preliminary optimization process for delay-line shaping of PPLN source
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
import matplotlib as mpl
import multiprocess as mp
import copy
import autograd.numpy as np
from scipy import signal
import seaborn

#%% import custom modules
from assets.functions import extractlogbook, save_class, load_class, splitindices, reload_experiment
from assets.callbacks import save_experiment_and_plot
from classes.environment import OpticalField_PPLN
from classes.components import DelayLine
from classes.experiment import Experiment
from classes.geneticalgorithmparameters import GeneticAlgorithmParameters
from optimization.wrappers import optimize_experiment
plt.close("all")


from math import factorial
import random

from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools


#%% initialize our input pulse, with the fitness function too
env = OpticalField_PPLN(n_samples=2**14, window_t=10e-9, peak_power=1)
# env.init_fitness()

components = {0: DelayLine() }
adj = []

exp = Experiment()
exp.buildexperiment(components, adj)
exp.make_path()
exp.inject_optical_field(env.field)



# Problem definition
def fitness_function(individual, exp):
    print(individual)
    fit = np.sum(individual)
    return (fit,)

PROBLEM = "dtlz2"
NOBJ = 1
K = 2
NDIM = NOBJ + K - 1
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0

# Algorithm parameters
MU = 20     # population size
NGEN = 100  # number of generations
CXPB = 1.0  # cross-over probabilit
MUTPB = 1.0
##

# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)

# Create classes
creator.create("FitnessMin", base.Fitness, weights=(1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)
##


# Toolbox initialization
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function, exp=exp)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
##


def main(seed=None):
    random.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook


if __name__ == "__main__":
    pop, stats = main()
    pop_fit = np.array([ind.fitness.values for ind in pop])

    # pf = problem.pareto_front(ref_points)
    # print(igd(pop_fit, pf))

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as Axes3d

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    p = np.array([ind.fitness.values for ind in pop])
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", s=24, label="Final Population")

    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], marker="x", c="k", s=32, label="Ideal Pareto Front")

    ref_points = tools.uniform_reference_points(NOBJ, P)

    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker="o", s=24, label="Reference Points")

    ax.view_init(elev=11, azim=-25)
    ax.autoscale(tight=True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("nsga3.png")