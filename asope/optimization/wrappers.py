import time
import copy

from assets.functions import extractlogbook
from optimization.geneticalgorithminner import inner_geneticalgorithm
from optimization.gradientdescent import gradient_descent, finetune_individual

def optimize_experiment(experiment, env, gap, verbose=False):
	if verbose:
		print('Number of cores: {}, number of generations: {}, size of population: {}'.format(gap.NCORES, gap.N_GEN,
		                                                                                      gap.N_POPULATION))

	# run (and time) the genetic algorithm
	tstart = time.time()
	hof, population, logbook = inner_geneticalgorithm(gap, env, experiment)
	tstop = time.time()

	if verbose:
		print('\nElapsed time = {}'.format(tstop - tstart))

	# % convert DEAP logbook to easier datatype
	log = extractlogbook(logbook)

	# %
	hof_fine = []
	for j in range(gap.N_HOF):
		individual = copy.deepcopy(hof[j])
		# hof_fine.append(individual)

		# % Now fine tune the best individual using gradient descent
		if gap.FINE_TUNE:
			if verbose:
				print('Fine-tuning the most fit individual using quasi-Newton method')
			if gap.GRADIENT_DESCENT == 'numerical':
				individual_fine = finetune_individual(individual, env, experiment)

			elif gap.GRADIENT_DESCENT == 'analytical':
				individual_fine = gradient_descent(individual, env, experiment,  gap.ALPHA, gap.MAX_STEPS)
		else:
			individual_fine = individual
		hof_fine.append(individual_fine)

	return experiment, hof, hof_fine, log
