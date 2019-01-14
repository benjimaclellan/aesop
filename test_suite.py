import time
import matplotlib.pyplot as plt
import numpy as np
from components.component_parameters import component_parameters

from classes import GeneticAlgorithmParameters
from functions.helperfunctions import extract_bounds, experiment_description, buildexperiment, extractlogbook

from environments.environment_pulse import PulseEnvironment
from simulators.simulator_classical import ClassicalSimulator

from geneticalgorithms.ga_functions_inner import CREATE_Inner

plt.close("all")

if __name__ == '__main__':    
    
    """
    This test suite is to ensure that we are getting proper environment and simulation results, with actually using any optimization. An experiment is chosen by the user, defined by the component numbers - and component values can either be randomized or set by the user.
    """
    

    fitnessfunction = 'SubIntegerImage'
    sim_kwargs = {'domain':'spectral', 'p':1, 'q':2}
    
    env = PulseEnvironment()
    sim = ClassicalSimulator(fitnessfunction, **sim_kwargs)

    ## Define an experiment by the component numbers - or leave empty and random experiment is created
    experiment_nums = [4]
    component_parameters = component_parameters(env, sim)
    experiment = buildexperiment(component_parameters, experiment_nums)


    (N_ATTRIBUTES, BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS) = extract_bounds(experiment)
        
    individual = CREATE_Inner(BOUNDSLOWER, BOUNDSUPPER, DTYPES, DSCRTVALS)
#    individual = [0,0.3*np.pi]
#    zT = 2 * (1/env.f_rep)**2 / (2*np.pi * np.abs( experiment[0].beta ))


    for tmp in range(1):

        fig, ax = plt.subplots(3,1)
        ax[0].plot(env.t, env.P(env.At0), label='initial')
        ax[1].plot(env.f, env.PSD(env.Af0, env.df))
        
        sim.simulate_experiment(individual, experiment, env, verbose=True)
        fitness = sim.fitness(env)
        print(fitness)
        
        ax[0].plot(env.t, env.P(env.At), label='output')
        ax[1].plot(env.f, env.PSD(env.Af, env.df))
        ax[1].set_xlim([-1e9,1e9])
        ax[0].legend()
        
        rfspectrum = env.RFSpectrum(env.At, env.dt) 
        ax[2].plot(env.f[-len(rfspectrum):], rfspectrum)
        plt.show()
