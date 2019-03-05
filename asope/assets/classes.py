import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from assets.functions import recurse
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum

"""
ASOPE
|- classes.py

Contains two custom classes, Experiment and GeneticAlgorithmParameters.

Experiment() defines an experiment as a directed graph (based on a DirectedGraph() from the networkx package), and stores all the class functions/variables to simulate a pulse through the setup. 

GeneticAlgorithmParameters simply contains variables used in the genetic algorithm, but no class functions. 

"""
    


## ***************************************************************88    

class Experiment(nx.DiGraph):
    
    def simulate(self, env):
        """
            Simulates an experiment once everything is setup, by traversing through the path to each component and applying the transformation(s)
        """
        
        # loop through each subpath, such that we simulate components in the right order
        for path_i, subpath in enumerate(self.path):
            
            # for each component in the current subpath
            for ii, node in enumerate(subpath):  
                    
                # if component (node) is a splitter, collect all incoming pulses
                if self.nodes[node]['info'].splitter:
                    
                    # if this is an input node (no predeccessors), get the prescribed input
                    if len(self.pre(node)) == 0:
                        At = self.nodes[node]['input']
                    
                    # if not an input node (somewhere in the middle of the setup)
                    else:
                        At = np.zeros([env.N, len(self.pre(node))]).astype('complex')
                        for jj in range(len(self.pre(node))):
                            At[:, jj] = self[self.pre(node)[jj]][node]['At']
                                  
                        # simulate the effect of a splitter
                        At = self.nodes[node]['info'].simulate(env, At, max(1,len(self.suc(node))))
                        
                        # store the split/coupled pulses to the successors
                        for jj in range(len(self.suc(node))):
                            self[node][self.suc(node)[jj]]['At'] = At[:,jj]
                
                # if component is not a splitter
                else:    
                    # if this is the first component in the subpath
                    if ii == 0:
                        # if this is an input node (no predeccessors), get the prescribed input
                        if len(self.pre(node)) == 0:
                            At = self.nodes[node]['input']
                          
                        # if this in the middle, get the incoming pulses
                        else:
                            At = self[self.pre(node)[ii]][node]['At']
                    
                    # now the pulse is stored in memory properly
                    At = self.nodes[node]['info'].simulate(env, At) 
                    
                    # if this is the last node in subpath, we simulate and save the pulse for future extraction
                    if ii == len(subpath)-1 and len(self.suc(node)) > 0: 
                        At = self.nodes[node]['info'].simulate(env, At)
                        self[node][self.suc(node)[0]]['At'] = At
                
                # if we're at a measurement node, save the pulse for easy checking later
                if node in self.measurement_nodes:
                    self.nodes[node]['output'] = At

        return
    
    
    
    
    def buildexperiment(self, components, adj, measurement_nodes):
        """
            With a list of components (nodes), connections (adjacency pairs) and measurement nodes, save everything to the class instance
        """
        
        # build the corresponding experiment graph
        self.add_nodes_from(list(components.keys()))
        self.add_edges_from(adj)
        
        # save the measurement nodes
        self.measurement_nodes = measurement_nodes
        
        # save the info of each component to the corresponding graph node
        for comp_key, comp in components.items():
            self.nodes[comp_key]['title'] = comp.name
            self.nodes[comp_key]['info'] = comp  # this save most of the information
    

    
    def checkexperiment(self):
        """
            A few sanity checks that this experiment setup is valid
        """
        
        # ensure the adjacency matrix is upper triangular
        mat = nx.adjacency_matrix(self).todense()
        isuptri = np.allclose(mat, np.triu(mat))
        assert isuptri
        
        ## ensure that any node with more than one predecessor/successor is a splitter
        for node in self.nodes():
            if (len(self.suc(node)) + len(self.pre(node)) == 0) and len(self.nodes()) > 1:
                raise ValueError('There is an unconnected component.')
            
            if (len(self.suc(node)) > 1 or len(self.pre(node)) > 1) and not self.nodes[node]['info'].splitter:
                raise ValueError("There is a component which splits the paths, but is not a 'splitter' type")
    
    
    
    def make_path(self):
        """
            Defines how to traverse an experiment's graph, such that each component is simulated/applied before the following ones. This is very critical when dealing with splitters, so that each incoming arm is simulated first. The function is recursive and outputs a list of lists (list of subpaths) and are the master instructions on the order to simulate the components (nodes)
        """
        
        # the number of times we've come *into* a node
        in_edges = {}
        for node in self.nodes(): 
            in_edges[node] = 0
        
        # all starting points of the setup
        startpoints = []
        for node in self.nodes():
            if len(self.pre(node)) == 0:
                startpoints.append(node)
        
        # empty lists for all the out_edges
        (out_edges, path) = ([], [])
        
        # initialize at the first startpoint, and remove it from our list for the future
        node = startpoints[0]
        startpoints.pop(0)
        
        # recursive function loops through all branches of graph, and finds the best way to traverse
        (self, node, out_edges, path, startpoints, in_edges) = recurse(self, node, out_edges, path, startpoints, in_edges)
        
        # store for future use
        self.path = path
        return
    
    
    
    def check_path(self):
        """
            Checks the path that has been created with .make_path(), performing some sanity checks to avoid errors or incorrect results later on
        """
        
        # ensure a path has been made already
        assert hasattr(self, 'path')
        
        # check that each node is travelled once, and the right number of nodes are there
        try:
            path_flatten = [item for sublist in self.path for item in sublist]
            for node in self.nodes():
                assert path_flatten.count(node)
            assert len(path_flatten) == len(self.nodes)
            print('All seems well to run experiments on this graph')
        except:
            raise ValueError('There seems to be a problem with how this graph is transversed to perform the experiment')
        return 


    def print_path(self):
        """
            Prints the path that will be traversed (list of subpaths)
        """
        assert hasattr(self, 'path')
        print('This graph will be transversed as follows: {}'.format(self.path))
        return


    def newattributes(self):
        """
            Creates the set of attributes (parameters) for all the nodes (components) in the setup and returns them
        """
        attributes = {}#[]
        for node in self.nodes():
            if self.nodes[node]['info'].N_PARAMETERS > 0:
                attributes[node] = self.nodes[node]['info'].newattribute()                
        return attributes
        
    def setattributes(self, attributes):
        """
            Saves a set of attributes (parameters) to the corresponding nodes (components)
        """
        for node in self.nodes():
            if self.nodes[node]['info'].N_PARAMETERS > 0:
                self.nodes[node]['info'].at = attributes[node]


    def suc(self, node):
        """
            Return the successors of a node (nodes which follow the current one) as a list
        """
        return list( self.successors(node) )
    

    def pre(self, node):
        """
            Return the predeccessors of a node (nodes which lead to the current one) as a list
        """
        return list( self.predecessors(node) )
        

    def measure(self, env, measurement_node):
        """
            Plots the pulse at a given measurement node
        """
        
        # collect the pulse as was simulated
        At = self.nodes[measurement_node]['output'].reshape(env.N)
    
        # plot both temporal and frequency domains, of input and output
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
        ax[0].set_title('Measurement node {}: {}'.format(measurement_node, self.nodes[measurement_node]['title']))
        alpha = 0.4
        ax[0].plot(env.t, P(env.At0), lw = 4, label='Input', alpha=alpha)
        ax[0].plot(env.t, P(At), ls='--', label='Output')    
        ax[0].legend()
        
        Af = FFT(At, env.dt)
        ax[1].plot(env.f, PSD(env.Af0, env.df), lw = 4, label='Input', alpha=alpha)
        ax[1].plot(env.f, PSD(Af, env.df), ls='-', label='Output')
        ax[1].legend()
        
        return

        
    def draw(self, titles = 'names'):
        """
            Plot the graph structure of the experiment, with either the nases of node key, or both
        """
        
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
        plt.figure()
        nx.draw_shell(self, labels = labeldict, with_labels=with_labels)    
        
        
    def printinfo(self):
        """
            Prints all the information about the components in the experiment
        """
        for node in self.nodes():
            c = self.nodes[node]['info']
            print('Name: {}, ID: {}, Type: {}'.format(c.name, c.id, c.type))
        
        
        
        
        
        
    def visualize(self, env):
        """
            Broken function - please ignore. It will potentially be fixed in future updates. Was meant for simple and clean visualization of the pulse as it progressed, but has not been updated since using a graph structure to represent the experiment(s)
        """
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
            

        
        
## ***************************************************************88    

class GeneticAlgorithmParameters(object):
    """
        A simple class that stores all the common parameters for running the genetic algorithm.
    """
    def __init__(self):
        self.N_POPULATION = 100
        self.N_GEN = 100
        self.MUT_PRB = 0.1
        self.CRX_PRB = 0.1
        self.N_HOF = 1
        self.VERBOSE = 0
        return

    
    
    
    
    
    
    
    