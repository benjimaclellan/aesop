import autograd.numpy as np

import config.config as config

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

    historical_marker = 1 # not always needed (only necessary for photoNEAT), but REALLY quite easy to maintain
                        # starts at one such that next returned marker is 2 (1st of all population two nodes all have the same markers)

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

    def speciate(self, population, debug=True):
        """
        Splits population into species

        :param population: the population of (score/None, graph) elements. Elements are grouped based on graph similarity
        """
        self.individual_species_map = {} # clear previous map

        # 1. Update all the representative individuals. The representative individual is always the one with the LEAST distance to the previous representative individual
        # if there is no suitable representative individual, a species may die out
        # also set # of element per species to zero, we'll update in part 2
        graph_set = set([graph for (_, graph) in population])
        rep_set = set(self.species.keys()) # make shallow copy such that we can iterate without fear

        for rep in rep_set:
            if rep in graph_set:
                graph_set.remove(rep) # this element is no longer available to represent another species
                self.species[rep] = 0
                continue
            best_rep = None
            best_rep_score = 5 # all distance functions have range 0 to 1, so this is fine as a max
            for graph in graph_set:
                if (graph not in self.species.keys()): # if it is, it's also out of bounds bc another species needs it as its representative
                    score = self.distance_func(rep, graph)
                    if score < best_rep_score and score < self.d_thresh:
                        best_rep = graph
                        best_rep_score = score
            
            self.species.pop(rep, None)
            if best_rep is not None: # The rep is dead, long live the rep! Is best_rep is None, the species dies out
                self.species[best_rep] = 0
                graph_set.remove(best_rep) # this graph is no longer available to represent a species
        
        # 2. Eliminate redundant species (i.e. species the representatives of which are excessively similar to one another)
        # TODO: consider whether there's a more efficient way of checking this (spoiler, there def is...)
        redundant_set = set()
        for species0 in self.species.keys():
            for species1 in self.species.keys():
                if species0 != species1 and species0 not in redundant_set and species1 not in redundant_set and self.distance_func(species0, species1) < self.d_thresh:
                    redundant_set.add(species1)
        
        for redundant in redundant_set:
            self.species.pop(redundant)

        if debug:
            print(f'PRIOR TO SPECIATION')
            print(f'species: {self.species}')
            print(f'map (individual->species): {self.individual_species_map}')

        # 2. Slot every other graph
        if debug:
            print(f'population size to speciate: {len(population)}')
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

        if debug:
            print(f'POST SPECIATION')
            print(f'species: {self.species}')

        # 3. Update self.d_thresh with my awesome P(ID) method
        if debug:
            print(f'current d_thresh: {self.d_thresh}')
        delta = np.clip(-1 * (self.target_species_num - len(self.species)) * THRESH_ADJUST_PROP_CONSTANT, -1 * THRESH_MAX_ADJUST, THRESH_MAX_ADJUST)
        self.d_thresh = delta + self.d_thresh
        self.d_thresh = max(0.05, min(self.d_thresh, 0.95)) # can't go past 0 or 1, since distance is always in that range (by def)
        if debug:
            print(f'updated d_thresh: {self.d_thresh}')
        
        print(f'post speciation, species pop size: {len(self.individual_species_map)}')
    
    def execute_fitness_sharing(self, population, generation_num):
        """
        Changes population scores to reflect effects of fitness sharing
        This assumes that each item in the population if already scored

        :param population: the population of (score, graph) elements
        :param generation_num: the current generation (may affect sharing, if self.protection_half_life is not None)
        :modifies population: updates the scores to reflect fitness sharing. population becomes a list of (score, graph, adjusted_score) elements
        """
        if self.protection_half_life is None:
            coeff = 0
        else:
            coeff = self.protection_half_life / np.log(2)

        for i in range(len(population)):
            graph = population[i][1]
            species_size = self.species[self.individual_species_map[graph]]
            denominator = max(1, np.exp(-coeff * generation_num) * species_size)

            population[i] = (population[i][0] / denominator, graph) # adjust fitness score

    def reverse_fitness_sharing(self, population, generation_num):
        if self.protection_half_life is None:
            coeff = 0
        else:
            coeff = self.protection_half_life / np.log(2)

        for i in range(len(population)):
            graph = population[i][1]
            species_size = self.species[self.individual_species_map[graph]]
            denominator = max(1, np.exp(-coeff * generation_num) * species_size)

            population[i] = (population[i][0] * denominator, graph)
    
    def get_crossover_candidates(self, graph, population):
        """
        Returns a list of crossover candidates for graph (i.e. individuals in the same species as graph)

        :param graph: graph which we want to crossover with another element
        :param population: pool of potentially compatible crossover candidates

        :pre-condition: speciate HAS BEEN CALLED on population already (or has been called on a superset of population)
                        Essentially requires all elements of population to exist inside our individual to species map

        :returns: the compatible (i.e. same species) crossover candidates
        """
        try:
            crossover_candidates = []
            for score, candidate in population:
                if self.individual_species_map[candidate] is self.individual_species_map[graph] and candidate is not graph:
                    crossover_candidates.append((score, candidate))
        
            crossover_candidates.sort(reverse = False, key=lambda x: x[0])  # we sort ascending, and take first (this is the minimum, as we minimizing)
            return crossover_candidates
        except KeyError as e:
            print(f'Warning: {e}. Population must be speciated prior to getting crossover candidates')
            return []
    
    @staticmethod
    def next_historical_marker():
        Speciation.historical_marker += 1 
        return Speciation.historical_marker


class NoSpeciation(Speciation):
    def __init__(self):
        pass

    def speciate(self, population):
        pass
    
    def execute_fitness_sharing(self, population, generation_num):
        pass

    def reverse_fitness_sharing(self, population, generation_num):
        pass
    
    def get_crossover_candidates(self, graph, population):
        if not isinstance(population, list):
            return list(population)
        return population


class DistanceEvaluatorInterface():
    def __init__(self):
        pass
    
    def distance(self, graph0, graph1):
        """
        Must be reflective and commutative, but NOT (necessarily) transitive
        """
        pass

class SimpleSubpopulationSchemeDist(DistanceEvaluatorInterface):      
    def __init__(self):
        pass 

    def distance(self, graph0, graph1):
        """
        Based on W. M. Spear's Simple Subpopulation Scheme
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.54.8319&rep=rep1&type=pdf

        Each graph has a label (which all of its children inherit) which indicates its species. Distance is 0 between species,
        1 otherwise. Really simple
        """
        if (graph0.speciation_descriptor['name'] != 'simple subpopulation scheme' or graph1.speciation_descriptor['name'] != 'simple subpopulation scheme'):
            raise ValueError(f'speciation descriptor {graph0.speciation_descriptor}, {graph1.speciation_descriptor} does not match simple subpopulation scheme!')
        if graph0.speciation_descriptor['label'] == graph1.speciation_descriptor['label']:
            return 0

        return 1


class vectorDIFF(DistanceEvaluatorInterface):
    """
    Comparison in this class is based solely on the number of same-class models in a graph. It does not rely on graph shape
    Each model forms a "basis" for the similarity vector. 
    E.g. if CWL->PM->PD were <1, 1, 0, …, 1>, then CWL->PM->PM->PD would be <1, 2, 0, …, 1>
    """
    def __init__(self):
        self.vector_basis = {} # maps model class to index number in the vector comparison
        index = 0
        for _, node_sub_classes in config.NODE_TYPES_ALL.items():
            for _, model_class in node_sub_classes.items():
                self.vector_basis[model_class] = index
                index += 1
        
        self.dimension = index
        
    def distance(self, graph0, graph1):
        if (graph0.speciation_descriptor['name'] != 'vectorDIFF' or graph1.speciation_descriptor['name'] != 'vectorDIFF'):
            raise ValueError(f'speciation descriptor {graph0.speciation_descriptor}, {graph1.speciation_descriptor} does not match vectorDIFF!')
        
        vector0 = self._get_vector(graph0)
        vector1 = self._get_vector(graph1)

        # normalization: suppose they both have totally different nodes, max possible norm is sqrt(size0**2 + size1**2)
        # actually it's realistically less bc you can't have all the same node type, but this will do as a normalization
        return np.linalg.norm(vector0 - vector1) / np.sqrt(np.power(len(graph0.nodes), 2) + np.power(len(graph1.nodes), 2))

    def _get_vector(self, graph):
        vector = np.zeros(self.dimension)
        for node in graph.nodes:
            model = type(graph.nodes[node]['model'])
            vector[self.vector_basis[model]] += 1
        return vector


class photoNEAT(DistanceEvaluatorInterface):
    """
    Basically NEAT speciation, but only nodes get a historical marker, and distance function also weighs heterogeneity of nodes
    
    TODO: add disjoint genes (not just excess) if we get crossovers going. Not useful rn
    """
    def __init__(self, a1=0.7, a2=0.3):
        weights_sum = a1 + a2
        self.weights = (a1 / weights_sum, a2 / weights_sum)
    
    def distance(self, graph0, graph1):
        if (graph0.speciation_descriptor['name'] != 'photoNEAT' or graph1.speciation_descriptor['name'] != 'photoNEAT'):
            raise ValueError(f'speciation descriptor {graph0.speciation_descriptor}, {graph1.speciation_descriptor} does not match photoNEAT!')
        
        markers0_set = set(graph0.speciation_descriptor['marker to node'].keys())
        markers1_set = set(graph1.speciation_descriptor['marker to node'].keys())
        markers_intersection = markers0_set.intersection(markers1_set)

        print(f'markers0: {markers0_set}')
        print(f'markers1: {markers1_set}')
        print(f'markers intersection: {markers_intersection}')

        structural_diff = (len(markers0_set) + len(markers1_set) - 2 * len(markers_intersection)) / (len(markers0_set) + len(markers1_set))
        print(f'structural_diff: {structural_diff}')
        # list[marker] contains 0 (False) if the nodes at the same marker are of different classes, 1 otherwise
        compositional_diff_list = [type(graph0.speciation_descriptor['marker to node'][marker]) != type(graph1.speciation_descriptor['marker to node'][marker]) for marker in markers_intersection]
        compositional_diff = np.sum(np.array(compositional_diff_list, dtype=int)) / len(markers_intersection)
        print(f'compositional_diff: {compositional_diff}')

        return self.weights[0] * structural_diff + self.weights[1] * compositional_diff
        
