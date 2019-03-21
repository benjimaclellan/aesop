def recurse(E, node, waiting_edges, path, startpoints, met_edges, level=0):
    path_i = []       
    
    # loop until conditions met (if for some reason there's an error, Python has fail-safes - but still be careful)
    
    print('node {}\nout_edges {}\npath {}\nstartpoints {}\nin_edges {}\nlevel {}\n\n'.format(node, waiting_edges, path, startpoints, met_edges, level))
    
    while True:
                    
        # add node (should happen only once)
        path_i.append(node)
        
        # if we are at a termination node (no successors), we can finish and check other, unfinished paths
        if not E.suc(node) or E.nodes[node]['info'].splitter:
            break    
        
        # if we're not coming up to a splitter, we can continue on the path no problem and add it to this sub_path
        if not E.nodes[E.suc(node)[0]]['info'].splitter:
            node = E.suc(node)[0]
        
        # if we ARE coming to a splitter, we need to start a new sub_path and recurse the function again
        else:
            break
    
#    # we've traversed through one subpath, let's add it to our list of paths
    path.append(path_i)
    
#    ## now we need to figure out where to start from for the next recursion

    for suc in E.suc(node):
        met_edges[suc] += 1

    if (E.nodes[node]['info'].splitter and met_edges[node] == len(E.pre(node))):
        print('is splitter, in_edge at node is less than num_pre')
        # were at a splitter but have not visited it enough
        if len(waiting_edges) > 0:
            node = waiting_edges[0]; waiting_edges.pop(0)
            (E, node, waiting_edges, path, startpoints, met_edges, level) = recurse(E, node, waiting_edges, path, startpoints, met_edges, level=level+1)
        
        if len(startpoints) != 0:
            node = startpoints[0]; startpoints.pop(0)
            (E, node, waiting_edges, path, startpoints, met_edges, level) = recurse(E, node, waiting_edges, path, startpoints, met_edges, level=level+1)
#        
#    
#    
#    
#        
#        
#    if not E.suc(node) and len(startpoints) == 0 and len(waiting_edges) == 0:
#        pass # all finished
#
##    if (E.nodes[node]['info'].splitter and met_edges[node] < len(E.pre(node))):
#        
#            
            
    
    
#    if not E.suc(node):# or (E.nodes[node]['info'].splitter and met_edges[node] != len(E.pre(node))):
#        
#        # if there's more unfinished paths, set one as our current node, remove it from the list and recurse
#        if len(waiting_edges) > 0: 
#            node = waiting_edges[0]; waiting_edges.pop(0)
#            (E, node, waiting_edges, path, startpoints, met_edges, level) = recurse(E, node, waiting_edges, path, startpoints, met_edges, level=level+1)
#            
#        # if there's NO MORE unfinished paths, but we have unchecked startpoint nodes (input nodes), go to one of those, set it as the current node, and recurse again
#        if len(startpoints) != 0:
#            node = startpoints[0]; startpoints.pop(0)
#            (E, node, waiting_edges, path, startpoints, met_edges, level) = recurse(E, node, waiting_edges, path, startpoints, met_edges, level=level+1)
#    
#    # if we're not at a termination point, append the current node to the current subpath
#    else:
#        node = E.suc(node)[0]
#        
#        # if this is the first time we encounter this node, we list it's output nodes and increment our count of how many times we've encountered this node
#        if met_edges[node] == 0:
#            for s in E.suc(node): waiting_edges.append(s)
#        met_edges[node] += 1 
#        
#        # if we have now we've run into this node the right number of times, we add it to the subpath as we can now simulate it
#        if met_edges[node] == len(E.pre(node)):
#            path.append([node])
#            
#            # we go back and check if there are more unfinished paths. If there is, select one and set our node there, remove if from the list, and recurse
#            if len(waiting_edges) > 0:
#                node = waiting_edges[0]; waiting_edges.pop(0)
#                (E, node, waiting_edges, path, startpoints, met_edges, level) = recurse(E, node, waiting_edges, path, startpoints, met_edges, level=level+1)
#            
#        # check if we still haven't run through all the incoming paths to the current node, we loop back again
#        elif met_edges[node] != len(E.pre(node)): ## finished loop?
#            if len(waiting_edges) > 0:
#                node = waiting_edges[0]; waiting_edges.pop(0)
#                (E, node, waiting_edges, path, startpoints, met_edges, level) = recurse(E, node, waiting_edges, path, startpoints, met_edges, level=level+1)
#            
#        # if we've made it this far, we've finished all the way through from one startpoint, and we choose another startpoint and recurse
#        if len(startpoints) != 0:
#            node = startpoints[0]
#            startpoints.pop(0)
#            (E, node, waiting_edges, path, startpoints, met_edges, level) = recurse(E, node, waiting_edges, path, startpoints, met_edges, level=level+1)
            
#     # once we've reached this far, we have traversed the whole graph and built instructions on how to simulate the graph (experiment)
    return (E, node, waiting_edges, path, startpoints, met_edges, level)




























def recurse(E, node, waiting_edges, path, startpoints, in_edges, out_edges, level=0):
    
    print('node {}\nout_edges {}\npath {}\nstartpoints {}\nin_edges {}\nlevel {}\n\n'.format(node, waiting_edges, path, startpoints, in_edges, out_edges, level))
    
    
    for suc in E.suc(node):
        in_edges[suc] -= set([node])
    
    if not E.nodes[node]['info'].splitter:#not a splitter
        path[-1].append(node)
#        for suc in E.suc(node): # check if next 
#            if E.nodes[suc]['info'].splitter:
#                node = suc
        node = E.suc(node)[0]
        (E, node, waiting_edges, path, startpoints, in_edges, out_edges, level) = recurse(E, node, waiting_edges, path, startpoints, in_edges, out_edges, level)
        
        
    if E.nodes[node]['info'].splitter:#splitter
        waiting_edges.append(out_edges[node])
        out_edges[suc] -= set([node])
        
        if not in_edges[node]: # haven't come from every input path
            node = waiting_edges[0]; waiting_edges.pop(0)
            (E, node, waiting_edges, path, startpoints, in_edges, out_edges, level) = recurse(E, node, waiting_edges, path, startpoints, in_edges, out_edges, level)
        else: 
            pass
            
    if not out_edges[node]: #term
        pass
    
        
    
#    ## now we need to figure out where to start from for the next recursion

    
    return (E, node, waiting_edges, path, startpoints, in_edges, out_edges, level)