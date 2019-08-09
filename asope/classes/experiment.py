import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as np
from assets.functions import FFT, IFFT, P, PSD, RFSpectrum
import copy

from analysis.wrappers import mc_analysis_wrapper, udr_analysis_wrapper, lha_analysis_wrapper, autograd_hessian

"""

Experiment() defines an experiment as a directed graph (based on a DirectedGraph() from the networkx package), and stores all the class functions/variables to simulate a pulse through the setup. 

"""

#%%
class Experiment(nx.DiGraph):
    
    def simulate(self, env, visualize=False):
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
                        At = np.zeros([env.n_samples, len(self.pre(node))]).astype('complex')
                        for jj in range(len(self.pre(node))):
                            At[:, [jj]] = self[self.pre(node)[jj]][node]['At']
                                  
                    # simulate the effect of a splitter
                    At = self.nodes[node]['info'].simulate(env, At, max(1,len(self.suc(node))), visualize)
                    # store the split/coupled pulses to the successors
#                    if len(self.suc(node)) > 0:
                    for jj in range(len(self.suc(node))):
                        self[node][self.suc(node)[jj]]['At'] = At[:,[jj]]

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
                    At = self.nodes[node]['info'].simulate(env, At, visualize) 
                    # if this is the last node in subpath, we save the pulse for future extraction
                    if ii == len(subpath)-1 and len(self.suc(node)) > 0: 
                        self[node][self.suc(node)[0]]['At'] = At
                
                # if we're at a measurement node, save the pulse for easy checking later
                if node in self.measurement_nodes:
                    self.nodes[node]['output'] = At

        return
    
    
    
    
    def buildexperiment(self, components, adj, measurement_nodes=None):
        """
            With a list of components (nodes), connections (adjacency pairs) and measurement nodes, save everything to the class instance
        """
        
        # build the corresponding experiment graph
        self.add_nodes_from(list(components.keys()))
        self.add_edges_from(adj)
        
        # save the measurement nodes
        if measurement_nodes == None:
            self.measurement_nodes = self.find_measurement_nodes()
        
        # save the info of each component to the corresponding graph node
        for comp_key, comp in components.items():
            self.nodes[comp_key]['title'] = comp.name
            self.nodes[comp_key]['info'] = comp  # this save most of the information
        return
    
    
    def cleanexperiment(self):
        """
            Removes redundancies in a graph
        """
        nodes_to_remove = set()
        for node in self.nodes():
            if not self.nodes[node]['info'].splitter:
                for pre in self.pre(node):
                    if self.nodes[node]['info'].type == self.nodes[pre]['info'].type:
                        nodes_to_remove.add(node)
                        if len(self.suc(node)) > 0:
                            self.add_edge(pre, self.suc(node)[0])
                            
                            
            
            if self.nodes[node]['info'].splitter and len(self.pre(node)) == 1 and len(self.suc(node)) == 1:
                self.add_edge(self.pre(node)[0], self.suc(node)[0])
                nodes_to_remove.add(node)
            
            if self.nodes[node]['info'].splitter and len(self.pre(node)) == 0 and len(self.suc(node)) == 1:
                nodes_to_remove.add(node)
            
            if self.nodes[node]['info'].splitter and len(self.pre(node)) == 1 and len(self.suc(node)) == 0:
                nodes_to_remove.add(node)
            
            
            if self.nodes[node]['info'].splitter and len(self.pre(node)) < 1 and len(self.suc(node)) < 1:
                nodes_to_remove.add(node)
        
        for node in nodes_to_remove:
#            print('removing node ', node)
            self.remove_node(node)

        return
    
    def checkexperiment(self):
        """
            A few sanity checks that this experiment setup is valid
        """
        
        # ensure the adjacency matrix does not have undirected edges (ie two-way)
        mat = nx.adjacency_matrix(self).todense()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i,j] > 1 and mat[j,i] > 1:
                    raise ValueError('There seems to be a two-way edge')
        # ensure there are no loops in the graph
        if len(list(nx.simple_cycles(self))) > 0:
            raise ValueError('There are loops in the graph')
        
        ## ensure that any node with more than one predecessor/successor is a splitter
        for node in self.nodes():
            if self.nodes[node]['info'].splitter and 'splitter' not in self.nodes[node]['info'].name:
                print('Splitter, but not really', node )
            
            if (len(self.suc(node)) + len(self.pre(node)) == 0) and len(self.nodes()) > 1:
                raise ValueError('There is an unconnected component.')
            
            if (len(self.suc(node)) > 1 or len(self.pre(node)) > 1) and not self.nodes[node]['info'].splitter:
                print('not a splitter', node)
                raise ValueError("There is a component which splits the paths, but is not a 'splitter' type")
        return
    
    def find_measurement_nodes(self):
        """
            returns the nodes with no successors
        """
        measurement_nodes = []
        for node in self.nodes():
            if not self.suc(node):
                measurement_nodes.append(node)
        return measurement_nodes
    
    
    def inject_optical_field(self, At):
        """
            inject light
        """
        for node in self.nodes():
            if not self.pre(node):
                self.nodes[node]['input'] = At
        return
    
    
    
    def make_path(self):
        """
            Defines how to traverse an experiment's graph, such that each component is simulated/applied before the following ones. This is very critical when dealing with splitters, so that each incoming arm is simulated first. The function is recursive and outputs a list of lists (list of path) and are the master instructions on the order to simulate the components (nodes)
        """
        
        node_list = set(self.nodes())
        path = []
        
        # find out splitter locations
        for node in self.nodes():
            if self.nodes[node]['info'].splitter:
                path.append([node])
        node_list -= set([item for sublist in path for item in sublist])
        
        
        while len(node_list) > 0:
            base_node = next(iter(node_list))
            curr_path = [base_node]
            node_list -= set([base_node])    
            node = base_node
            while True:
                if len(self.pre(node)) > 0:
                    if not self.nodes[self.pre(node)[0]]['info'].splitter:
                        node = self.pre(node)[0]
                        node_list -= set([node])
                        curr_path = [node] + curr_path
                    else:
                        break
                else:
                    break
                
            node = base_node
            while True:
                if len(self.suc(node)) > 0:
                    if not self.nodes[self.suc(node)[0]]['info'].splitter:
                        node = self.suc(node)[0]
                        node_list -= set([node])
                        curr_path = curr_path + [node]
                    else:
                        break
                else:
                    break
            
            path.append(curr_path)
        
        
        pathscopy = copy.deepcopy(path)
        for k in range(len(path)):
            for i in range(0,len(path)):
                subpath_i = pathscopy[i]
                node = subpath_i[0]
                pres = []
                for pre in self.pre(node):
                    for idx, subpath in enumerate(path):
                        if subpath[-1] == pre:
                            pres.append(idx)
                
                if not pres:
                    continue
                
                loc = max(pres)
                curr_loc = path.index(subpath_i)
                if loc > curr_loc:
                    path.insert(loc, path.pop(curr_loc))
        
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
#            print('All seems well to run experiments on this graph')
        except:
            raise ValueError('There seems to be a problem with how this graph is transversed to perform the experiment')
        return 


    def print_path(self):
        """
            Prints the path that will be traversed (list of path)
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
        return

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
        
    def remove_component(self, node):
        """
        Removes given node, if possible, and stiches graph back together
        """
        if (len(self.pre(node)) > 1) or (len(self.suc(node)) > 1):
            print('Cannot remove splitter node in this way')
        
        else:
            pre = self.pre(node)
            suc = self.suc(node)
            
            if len(pre) == 1:
                self.remove_edge(pre[0], node)
            if len(suc) == 1:
                self.remove_edge(node, suc[0])
            if len(pre) == 1 and len(suc) == 1:
                self.add_edge(pre[0], suc[0])    
            self.remove_node(node)
        return

    def get_nonsplitters(self):
        valid_nodes = []
        for node in self.nodes():
            if not self.nodes[node]['info'].splitter:
                valid_nodes.append(node)
        return valid_nodes

    
    def measure(self, env, measurement_node, check_power = False, fig = None):
        """
            Plots the pulse at a given measurement node
        """
        
        # collect the pulse as was simulated
        At = self.nodes[measurement_node]['output'].reshape(env.n_samples)
        
        PAt, PAt0 = P(At), P(env.At0)
        PAf, PAf0 = PSD(At, env.dt, env.df), PSD(env.At0, env.dt, env.df)
        if check_power:
            check = self.power_check_single(At, display=True)
            
        # plot both temporal and frequency domains, of input and output
        if fig == None:
            fig = plt.figure(dpi=80, figsize=(8, 10))
        else: 
            fig.clf()
        
        ax = []
        ax.append(fig.add_subplot(2,1,1))
        ax.append(fig.add_subplot(2,1,2))     

        alpha = 0.4
        ax[0].plot(env.t, PAt0, lw = 4, label='Input', alpha=alpha)
        ax[0].plot(env.t, PAt, ls='--', label='Output')  
        ax[0].set_ylim([0,2*max([max(PAt), max(PAt0)])])
        ax[0].legend()
        
        ax[1].plot(env.f, PAf0, lw = 4, label='Input', alpha=alpha)
        ax[1].plot(env.f, PAf, ls='--', label='Output')
        ax[1].set_ylim([0,2*max([max(PAf), max(PAf0)])])
        ax[1].legend()
        
        return At
        
        
        
    def draw(self, node_label = 'both', title=None, ax=None):
        """
            Plot the graph structure of the experiment, with either the nases of node key, or both
        """
        
        with_labels = True
        labeldict = {}
        for i in self.nodes():
            if node_label == 'titles':
                labeldict[i] = self.nodes[i]['title']
            elif node_label == 'keys':
                labeldict[i] = i
            elif node_label == 'disp_name':
                st = (self.nodes[i]['info'].disp_name).replace(' ', '\n') 
                labeldict[i] = "{}\n{}".format(i,st)
            elif node_label == 'both':
                labeldict[i] = '{}, {}'.format(i, self.nodes[i]['title'])
            else:
                with_labels = False
                

        nodePos = nx.drawing.nx_pydot.graphviz_layout(self, prog='circo')

        if ax == None:
            fig, ax = plt.subplots(1,1)

        nx.draw(self, ax= ax, pos = nodePos, labels = labeldict, with_labels=with_labels, arrowstyle='fancy', edge_color='burlywood', node_color='powderblue', node_shape='8', font_size=7)
        return ax
        
        
    def printinfo(self):
        """
            Prints all the information about the components in the experiment
        """
        for node in self.nodes():
            c = self.nodes[node]['info']
            print('Name: {}, ID: {}, Type: {}'.format(c.name, c.id, c.type))
        
   
    def visualize(self, env, at=None, measurement_node=None, ax1=None, ax2=None):
        """
        """
        
        self.simulate(env, visualize=True)
        
        if ax1 == None or ax2 == None:
            fig, ax = plt.subplots(2, 1, figsize=(8, 10), dpi=80)
            ax1, ax2 = ax[0], ax[1]
            
        if measurement_node is None:
            measurement_node = self.measurement_nodes[0]
            
        if measurement_node is not None:
            At = self.nodes[measurement_node]['output'].reshape(env.n_samples)
            PAt = P(At)
            PAf = PSD(At, env.dt, env.df)
            
            ax1.plot(env.t/1e-9, PAt/np.max(PAt), label='Power', alpha=0.7)
            ax2.plot(env.f/1e9, PAf/np.max(PAf), label='PSD', alpha=0.7)
            
        for node in self.nodes():
            if self.nodes[node]['info'].lines is not None:
                for line in self.nodes[node]['info'].lines:
                    if line[0] == 't':
                        ax1.plot(env.t/1e-9, line[1]/np.max(line[1]), label=line[2])
                    elif line[0] == 'f':
                        ax2.plot(env.f/1e9, line[1]/np.max(line[1]), label=line[2])
                    else:
                        raise ValueError('Invalid x variable')                    
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel(' ')
        
        ax2.set_xlabel('Frequency Offset from Carrier (GHz)')
        ax2.set_ylabel(' ')
        
        ax1.legend()
        ax2.legend()
        
#        fig.tight_layout(rect=[0,0,.7,1])
        at_str = ''
        for node, node_ats in at.items():
            at_str += '{}:\n'.format(node)
            for at_i in node_ats:
                at_str += '--{}\n'.format(at_i)
        if at is not None:
            ax1.text(1.2,1.2, at_str, size=8, ha="left", transform=ax2.transAxes)
        
        fig.tight_layout()
        return 
           
    def attributes_from_list(self, x_opt, node_lst, idx_lst):
        at_dict = {}
        for node in set(node_lst):
            indices = [i for i, x in enumerate(node_lst) if x == node]
            node_ats = list(indices)

            for index in indices:
                node_ats[idx_lst[index]] = x_opt[index]
            at_dict[node] = node_ats
        return at_dict



    def experiment_info_as_list(self, at):
        """
        Get the attributes and return as a list instead of a dictionary
        """
        at_lst, node_lst, idx_lst, sigma_lst, mu_lst, at_names = [], [], [], [], [], []
        for node in self.nodes():
            if node in at.keys():
                for i, allele in enumerate(at[node]):
                    if i not in self.nodes[node]['info'].FINETUNE_SKIP:
                        at_lst.append(allele)
                        node_lst.append(node)
                        idx_lst.append(i)
                        at_names.append(self.nodes[node]['info'].AT_NAME[i])
                        sigma_lst.append(self.nodes[node]['info'].SIGMA[i])
                        mu_lst.append(self.nodes[node]['info'].MU[i])
        return at_lst, node_lst, idx_lst, sigma_lst, mu_lst, at_names


    def power_check_single(self, At, display=False):
        """
            Simple sanity check for total power, that input power >= output power, for one output node
        """
        
        check = True # assume all is good, change to false if there is a problem
        
        totalpower_in = 0
        for node in self.nodes():
            if len(self.pre(node)) == 0:
                totalpower_in += np.sum(P(self.nodes[node]['input']))
        totalpower_out = np.sum(P(At))
        
        ratio = (totalpower_out - totalpower_in)/totalpower_in
        if ratio > 0.05:
            display = True
            check = False
            print('There seems to be an issue in energy conservation')
        if display:
            print('Input power: {}\nOutput power: {}'.format(totalpower_in, totalpower_out))
        return check






    def init_fitness_analysis(self, at, env, method='LHA', verbose=False, **kwargs):
        x_opt, node_lst, idx_lst, sigma_lst, mu_lst, at_names = self.experiment_info_as_list(at)

        self.analysis_verbose = verbose
        self.at_names = at_names
        self.node_lst = node_lst
        self.method = method

        if method == 'LHA':
            if verbose: print('LHA. Does need an initialize to get autograd function')
            def analysis_wrapper(x_opt, env=env, exp=self, node_lst=node_lst, idx_lst=idx_lst, mu_lst=mu_lst, sigma_lst=sigma_lst):
                x_opt = np.array(x_opt)
                exp.inject_optical_field(env.At)
                at = exp.attributes_from_list(x_opt, node_lst, idx_lst)
                exp.setattributes(at)
                exp.simulate(env)
                at = exp.nodes[exp.measurement_nodes[0]]['output']
                fit = env.fitness(at)
                return fit[0]

            Hf = autograd_hessian(analysis_wrapper)
            self.analysis_function = lambda x_opt: lha_analysis_wrapper(x_opt, analysis_wrapper, mu_lst,
                                                                        sigma_lst, Hf = Hf)

        elif method == 'UDR' or method == 'MC':
            def analysis_wrapper(x_opt, env=env, exp=self, node_lst=node_lst, idx_lst=idx_lst):
                exp.inject_optical_field(env.At)
                at = exp.attributes_from_list(x_opt, node_lst, idx_lst)
                exp.setattributes(at)
                exp.simulate(env)
                At = exp.nodes[exp.measurement_nodes[0]]['output']
                fit = env.fitness(At)
                return fit[0]

            if method == 'UDR':
                if verbose: print("UDR. Does need to initialize matrices as they can be computed once")
                self.analysis_function = lambda x_opt: udr_analysis_wrapper(x_opt, analysis_wrapper, mu_lst, sigma_lst)

            elif method == 'MC':
                if verbose: print("MC - doesn't need an initialization, just run")
                self.analysis_function = lambda x_opt: mc_analysis_wrapper(x_opt, analysis_wrapper, mu_lst, sigma_lst)

        else:
            print('Not a method')
        return



    def run_analysis(self, at, verbose=False):
        x_opt, *_ = self.experiment_info_as_list(at)
        parameter_stability, *tmp = self.analysis_function(x_opt)

        if verbose:
            comp_labels = []
            for node in self.node_lst:
                comp_labels.append(self.nodes[node]['title'])

            results = pd.DataFrame(comp_labels, columns=["Component"])
            results = results.assign(Attribute=self.at_names)
            results = results.assign(Output_Deviation=parameter_stability)
            print(results)
            print('Analysis with {} method is completed.\n'.format(self.method))

        return parameter_stability, tmp


    def run_sim(self, x_opt, node_lst, idx_lst, env):
        """ testing func to be removed """
        at = self.attributes_from_list(x_opt, node_lst, idx_lst)
        x_old, node_lst, idx_lst, sigma_lst, mu_lst, at_names = self.experiment_info_as_list(at)
        x_opt = (np.array(x_opt)*np.array(sigma_lst)) + np.array(x_old) # for the lha we need a coordinate transformation
        self.inject_optical_field(env.At)
        at = self.attributes_from_list(x_opt, node_lst, idx_lst)
        self.setattributes(at)
        self.simulate(env)
        at = self.nodes[self.measurement_nodes[0]]['output']
        fit = env.fitness(at)
        return fit[0]













    # # %% Redundancy checks
    # def remove_redundancies(exp, env, verbose=False):
    #     """
    #     Check over an experiment for redundant nodes
    #
    #     :param experiment:
    #     :param env:
    #     :param verbose:
    #     :return:
    #     """
    #     exp_redundancies = deepcopy(exp)
    #     exp_backup = deepcopy(exp)
    #
    #     # Compute the fitness of the unmodified experiment, for comparison
    #     At = exp_redundancies.nodes[exp_redundancies.measurement_nodes[0]]['output']
    #     fit = env.fitness(At)
    #     valid_nodes = exp_redundancies.get_nonsplitters()
    #
    #     for node in valid_nodes:
    #         exp_redundancies.remove_component(node)  # Drop a node from the experiment
    #         exp_redundancies.measurement_nodes = exp_redundancies.find_measurement_nodes()
    #         exp_redundancies.checkexperiment()
    #         exp_redundancies.make_path()
    #         exp_redundancies.check_path()
    #         exp_redundancies.inject_optical_field(env.At)
    #         # Simulate the optical table with dropped node
    #         exp_redundancies.simulate(env)
    #         At_mod = exp_redundancies.nodes[exp_redundancies.measurement_nodes[0]]['output']
    #         fit_mod = env.fitness(At_mod)
    #         if verbose:
    #             print(fit_mod, fit)
    #             print("Dropped node: {}. Fitness: {}".format(node, fit_mod))
    #
    #         # Compare the fitness to the original optical table
    #         if fit_mod[0] >= fit[0]:  # and fit_mod[1] >= fit[1]:
    #             print("Dropping node")
    #             # If dropping the node didn't hurt the fitness, then drop it permanently
    #             fit = fit_mod
    #         else:
    #             exp_redundancies = deepcopy(exp_backup)
    #
    #     return exp_redundancies