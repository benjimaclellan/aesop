import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from assets.functions import recurse

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
    
    def make_path(self):
        in_edges = {}
        for node in self.nodes(): 
            in_edges[node] = 0
        
        startpoints = []
        for node in self.nodes():
            if len(self.pre(node)) == 0:
                startpoints.append(node)
        
        (out_edges, path) = ([], [])
        
        node = startpoints[0]
        startpoints.pop(0)
        
        (self, node, out_edges, path, startpoints, in_edges) = recurse(self, node, out_edges, path, startpoints, in_edges)
        
        self.path = path
        return
    
    def check_path(self):
        assert hasattr(self, 'path')
        try:
            path_flatten = [item for sublist in self.path for item in sublist]
            for node in self.nodes():
                assert path_flatten.count(node)
            assert len(path_flatten) == len(self.nodes)
            print('All seems well to run experiments on this graph')
        except:
            raise ValueError('There seems to be a problem with how this graph is transversed to perform the experiment')
        return 

    def suc(self, node):
        return list( self.successors(node) )
    
    def pre(self, node):
        return list( self.predecessors(node) )
        
    def buildexperiment(self, components, adj):        
        self.add_nodes_from(list(components.keys()))
        self.add_edges_from(adj)

        for comp_key, comp in components.items():
            self.nodes[comp_key]['title'] = comp.name
            self.nodes[comp_key]['info'] = comp 
        
    def draw(self, titles = 'names'):
        with_labels = True
        labeldict = {}
        for i in self.nodes():
            if titles == 'titles':
                labeldict[i] = self.nodes[i]['title']
            elif titles == 'keys':
                labeldict[i] = i
            elif titles == 'both':
                labeldict[i] = '{}, {}'.format(i, self.nodes[i]['title'])
            else:
                with_labels = False
        nx.draw_shell(self, labels = labeldict, with_labels=with_labels)    
        
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
        attributes = []
        for node in self.nodes():
            if self.nodes[node]['info'].N_PARAMETERS > 0:
                attributes.append( self.nodes[node]['info'].newattribute() )
        return attributes
        
    def setattributes(self, attributes):
#        cnt = 0
        for node in self.nodes():
            if self.nodes[node]['info'].N_PARAMETERS > 0:
#                self.nodes[node]['info'].at = attributes[cnt]
                self.nodes[node]['info'].at = attributes[node]
#                cnt += 1

    def printinfo(self):
        for i in range(0, self.n_components):
            c = self.nodes[i]['info']
            print('Name: {}, ID: {}, Type: {}'.format(c.name, c.id, c.type))

#    def plot_env(self, env, title=None):
#        fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
#        ax[0].set_title(title)
#        alpha = 0.4
#        ax[0].plot(env.t, env.P(env.At0), ls='--', label='Input', alpha=alpha)
#        ax[0].plot(env.t, env.P(env.At), label='Output')    
#        ax[0].legend()
#        
#        ax[1].plot(env.f, env.P(env.Af0),ls='--', label='Input', alpha=alpha)
#        ax[1].plot(env.f, env.P(env.Af), label='Output')
#        ax[1].legend()
        
        
        
    def checkexperiment(self):
        ## check experiment
        mat = nx.adjacency_matrix(self).todense()
        isuptri = np.allclose(mat, np.triu(mat)) # check if upper triangular
        assert isuptri
        
        ## ensure that any node with more than one predecessor/successor is a splitter
        for node in self.nodes():
            if (len(self.suc(node)) + len(self.pre(node)) == 0):
                raise ValueError('There is an unconnected component.')
            
            if (len(self.suc(node)) > 1 or len(self.pre(node)) > 1) and not self.nodes[node]['info'].splitter:
                raise ValueError("There is a component which splits the paths, but is not a 'splitter' type")



    def simulate(self, env, visualize=False):
        raise ValueError
#        for i in range(0,self.n_components):
#            
#            env_out = self.nodes[i]['info'].simulate(env_in)
#            for jj in range(self.out_degree()[i]):
#                s = suc[jj]
#        
#                self[i][s]['edge_name'] = 'Edge-{}-{}'.format(i,s)
#                self[i][s]['env'] = env_out[jj]    
#            
#        p = list(self.predecessors(self.terminal))[0]
#        env_out = self[p][self.terminal]['env']
            
        return 
    
        
        
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

    
    
    
    
    
    
    
    