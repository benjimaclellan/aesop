import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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
    
#    def buildexperiment(self, components):
#        self.n_components = len(components)
#        self.clear()
#        
#        self.add_node(-1, title = 'input', info = None)
#        for i in range(0, self.n_components):
#            self.add_edges_from([(i-1, i)])
#            self.nodes[i]['title'] = components[i].name
#            self.nodes[i]['info'] = components[i]
            
    def buildexperiment(self, components, adj):
        nodes = [i for i in range(0,len(components))]
        
        self.add_nodes_from(nodes)
        self.add_edges_from(adj)
        self.terminal_nodes = []
        
        self.n_components = self.number_of_nodes()
        for i in range(self.number_of_nodes()):
            self.nodes[i]['title'] = components[i].name
            self.nodes[i]['info'] = components[i]    
    
    def terminate_path(self, node_number):
        self.add_node('terminal{}'.format(node_number), title = 'terminal{}'.format(node_number))
        self.add_edge(node_number, 'terminal{}'.format(node_number))
        self.terminal_nodes.append('terminal{}'.format(node_number))
        
        return
            
#    def simulate(self, env, visualize=False):
##        for i in range(0, self.n_components):
##            self.nodes[i]['info'].simulate(env, visualize=visualize)
##        return 
#        
#        for i in range(0, self.n_components):
#            self.nodes[i]['info'].simulate(env, visualize=visualize)
#            
#        return
        
    def draw(self):
        labeldict = {}
        for i in self.nodes():
            labeldict[i] = self.nodes[i]['title']
        nx.draw_shell(self, labels = labeldict, with_labels=True)    
        
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

    def plot_env(self, env, title=None):
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
        ax[0].set_title(title)
        alpha = 0.4
        ax[0].plot(env.t, env.P(env.At0), ls='--', label='Input', alpha=alpha)
        ax[0].plot(env.t, env.P(env.At), label='Output')    
        ax[0].legend()
        
        ax[1].plot(env.f, env.P(env.Af0),ls='--', label='Input', alpha=alpha)
        ax[1].plot(env.f, env.P(env.Af), label='Output')
        ax[1].legend()
        
        

    def checkexperiment(self):
        ## check experiment
        mat = nx.adjacency_matrix(self).todense()
        isuptri = np.allclose(mat, np.triu(mat)) # check if upper triangular
        assert isuptri
        
        for i in range(self.number_of_nodes()):
            if self.in_degree()[i] == 0 :
                assert i == 0
                
            elif self.in_degree()[i] > 1:
                assert self.nodes[i]['info'].type == ('powersplitter' or 'frequencysplitter')
                
            elif self.in_degree()[i] == 1:
                pass
            else:
                raise ValueError()
                
            if self.out_degree()[i] == 0:
                self.terminate_path(i)
            
            elif self.out_degree()[i] > 1:
    #            assert self.nodes[i]['info'].type == 'powersplitter'
                assert self.in_degree()[i] == 1 or self.in_degree()[i] == 0
                
            elif self.out_degree()[i] == 1:
                pass
            else:
                raise ValueError()
        return 
        

    def simulate(self, env, visualize=False):
        for i in range(0,self.n_components):
        
            pre = list(self.predecessors(i))
            suc = list(self.successors(i)) 
            
            if len(pre) > 0:
                env_in = []
                for p in pre:
                    env_in.append( self[p][i]['env'])
        #            self.plot_env(self[p][i]['env'], title='Edge{}-{}, {}-{}'.format(p,i, self.nodes[p]['title'], self.nodes[i]['title']))
            else:
                env_in = [env]
                
            
            
            env_out = self.nodes[i]['info'].simulate(env_in)
            for jj in range(self.out_degree()[i]):
                s = suc[jj]
        
                self[i][s]['edge_name'] = 'Edge-{}-{}'.format(i,s)
                self[i][s]['env'] = env_out[jj]    
            
        p = list(self.predecessors(self.terminal))[0]
        env_out = self[p][self.terminal]['env']
            
        return env_out
    
        
        
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

    
    
    
    
    
    
    
    