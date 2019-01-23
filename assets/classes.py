import networkx as nx
import matplotlib.pyplot as plt
## ***************************************************************88    
    
class Environment(object):
    """   
    A class that stores all our environment parameters etc.
    """
    def __init__(self):
        return
    
    def reset(self):
        return


## ***************************************************************88    

class Experiment(nx.DiGraph):
    def buildexperiment(self, components):
        self.n_components = len(components)
        self.clear()
        
        self.add_node(-1, title = 'input', info = None)
        for i in range(0, self.n_components):
            self.add_edges_from([(i-1, i)])
            self.nodes[i]['title'] = components[i].name
            self.nodes[i]['info'] = components[i]
            
    def simulate(self, env, visualize=False):
        for i in range(0, self.n_components):
            self.nodes[i]['info'].simulate(env, visualize=visualize)
        return 
    
    def visualize(self, env):
        self.simulate(env, visualize=True)
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
        for i in range(0, self.n_components):
            for line in self.nodes[i]['info'].lines:
                if line is not None:
                    if line[0] == 't':
                        ax[0].plot(env.t, line[1], label=self.nodes[i]['info'].name)
                    elif line[0] == 'f':
                        ax[1].plot(env.f, line[1], label=self.nodes[i]['info'].name)
                    else:
                        raise ValueError('Invalid x variable')                    
        ax[0].legend()
        ax[1].legend()
        return
            
    def newattributes(self):
        ind = []
        for i in range(0, self.n_components):
            ind.append( self.nodes[i]['info'].newattribute() )
        return ind
        
    def setattributes(self, attributes):
        for i in range(0, self.n_components):
            self.nodes[i]['info'].at = attributes[i]
        return 
    
    def printinfo(self):
        for i in range(0, self.n_components):
            c = self.nodes[i]['info']
            print('Name: {}, ID: {}, Type: {}'.format(c.name, c.id, c.type))



## ***************************************************************88    

class GeneticAlgorithmParameters(object):
    """
    A simple class that stores all the common parameters for a genetic algorithm run.
    """
    def __init__(self):
        self.N_POPULATION = 100
        self.N_GEN = 100
        self.MUT_PRB = 0.1
        self.CRX_PRB = 0.1
        self.N_HOF = 1
        self.VERBOSE = 0
        return

    
    
    
    
    
    
    
    