import autograd.numpy as np

"""
Module for speciation! Generally follows the model given by NEAT:
1. Each species has a representative member
2. You add each new element to the first species where dist(species_representative, element) < threshold
3. Threshold adjusts proportionally to try and achieve the optimal # of species (PID also possible but that's fine for now...)
4. The new representative member each generation is picked to have the least distance from the previous member

Different distance functions are considered:
1. Simple subpopulation Scheme
2. vectorDIFF
3. photoNEAT
4. shapeDIFF (graph shaped based speciation) but not sure how that works yet


Resources:
(1) Topology optimization - Readings and Musings (internal doc): https://docs.google.com/document/d/1B7fZL53t9ir5lV4CrzrWsbXoHXnKwaBLPRoSpcrKZ3s/edit?usp=sharing
(2) All the papers linked in (1)
(3) NEAT-python, useful reference: https://github.com/CodeReclaimers/neat-python/tree/master/neat
"""

THRESH_MAX_ADJUST = 0.3
THRESH_ADJUST_PROP_CONSTANT = 0.1

class Speciation():
    """
    """
    def __init__(self, target_species_num=None, d_thresh=0.4, protection_half_life=None, distance_func=None):
        """
        :param target_species_num: target number of species (note: actual number will fluctuate)
        :param d_thresh: initial (start) delta threshold for 2 items to be considered the same species
        :param protection_half_life: species are "protected" by fitness sharing, such that a single species can't take everyone over
                                     this implementation allows the protection to decay over time, such that
                                     the individual's fitness is less and less affective (i.e. more greedy behaviour)
        :param distance_funct: distance function determining whether or not 2 
        """
        self.target_species_num = target_species_num

        if (d_thresh <= 0 or d_thresh >= 1):
            raise ValueError(f'Delta threshold (d_thresh) for speciation must be in (0, 1), not {d_thresh}')
        self.d_thresh = d_thresh

        self.species = {} # key: representative, value: num of elements in species
        self.individual_species_map = {} # maps most recent population individuals to their species representative
        self.distance_func = distance_func
        self.protection_half_life = protection_half_life

    def speciate(self, population):
        """
        Splits population into species

        :param population: the population of (score/None, graph) elements. Elements are grouped based on graph similarity
        """
        self.individual_species_map = {} # clear previous map

        # 1. Update all the representative individuals. The representative individual is always the one with the LEAST distance to the previous representative individual
        # if there is no suitable representative individual, a species may die out
        # also set # of element per species to zero, we'll update in part 2
        graph_set = set([graph for (_, graph) in population])
        for rep in self.species.items():
            if rep in graph_set:
                graph_set.remove(rep) # this element is no longer available to represent another species
                self.species[rep] = 0
                continue
            best_rep = None
            best_rep_score = 5 # all distance functions have range 0 to 1, so this is fine as a max
            for graph in graph_set:
                if (graph not in self.species): # if it is, it's also out of bounds bc another species needs it as its representative
                    score = self.distance_func(rep, graph)
                    if score < best_rep_score and score < self.d_thresh:
                        best_rep = graph
                        best_rep_score = score
            
            self.species.pop(rep, None)
            if best_rep is not None: # The rep is dead, long live the rep! Is best_rep is None, the species dies out
                self.species[best_rep] = 0
                graph_set.remove(best_rep) # this graph is no longer available to represent a species
                

        # 2. Slot every other graph
        for _, graph in population:
            if graph in self.species:
                self.individual_species_map[graph] = graph # graph is its own representative, yay!
                self.species[graph] += 1
                continue
            for rep in self.species:
                if self.distance_func(rep, graph) < self.d_thresh:
                    self.individual_species_map[graph] = rep
                    self.species[rep] += 1
                    break
    
            # if you don't manage to slot it, make a new species
            if graph not in self.individual_species_map: # alas, even after checking all the species, no dice. Make a new species with graph being the representative
                self.individual_species_map[graph] = graph 
                self.species[graph] = 1
        
        # 3. Update self.d_thresh with my awesome P(ID) method
        self.d_thresh = min((self.target_species_num - len(self.species)) * THRESH_ADJUST_PROP_CONSTANT, THRESH_MAX_ADJUST) + self.d_thresh
        self.d_thresh = max(0, min(self.d_thresh, 1)) # can't go past 0 or 1, since distance is always in that range (by def)
    
    def execute_fitness_sharing(self, population, generation_num):
        """
        Changes population scores to reflect effects of fitness sharing
        This assumes that each item in the population if already scored

        :param population: the population of (score, graph) elements
        :param generation_num: the current generation (may affect sharing, if self.protection_half_life is not None)
        :modifies population: updates the scores to reflect fitness sharing. population becomes a list of (adjusted_score, graph) elements
        """
        if self.protection_half_life is None:
            coeff = 0
        else:
            coeff = self.protection_half_life / np.log(2)

        for i in range(len(population)):
            graph = population[i][1]
            species_size = self.species[self.individual_species_map[graph]]
            denominator = np.max(1, np.exp(-coeff * generation_num) * species_size)

            population[i][0] /= denominator # adjust fitness score


class DistanceEvaluatorInterface():
    def __init__(self):
        pass
    
    def distance(self, graph0, graph1):
        pass


class SimpleSubpopulationSchemeDist(DistanceEvaluatorInterface):      
    def __init__(self, num_species=10):
        self.num_species = num_species

    def distance(self, graph0, graph1):
        """
        Based on W. M. Spear's Simple Subpopulation Scheme
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.54.8319&rep=rep1&type=pdf

        Each graph has a label (which all of its children inherit) which indicates its species. Distance is 0 between species,
        1 otherwise. Really simple
        """
        if graph0.speciation_descriptor == graph1.speciation_descriptor:
            return 1

        return 0

