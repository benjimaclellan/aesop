# BUGS, TODOS, and IDEAS

Here is will try to keep track of any bugs, todos and ideas I have for this project

## IDEAS
* Each time a new experiment is run (next individual in top-level GA), we use could use the best parameters for each component from last experiment as the starting values. Maybe one individual inherits the best parameters, a percentage get certain alleles from hof mixed with random others. Others are completely random
    
* Change the mutation, crossover, selection percentages as we go. I think that we should reduce mutation through time, increase selection and crossover. This somewhat mimics RL exploration vs exploitation

* Potentially (though this could be unnecessary with idea above) is to implement a gradient ascent optimization on the best individuals of each run to refine them. This is based on an unproven assumption that in the local area of an extrema, the gradient is well defined and can be followed to the exact extreme value, not just close to its

* Here's a good idea (though, this will take lots of time). Once we have ASOPE working for quantum simulations and experiments, can we implement a neural network afterwards to tease out new information. In reality, we don't even need the GA's, as they could influence the results. So, we can use Principled Component Analysis or K-means clustering to do something useful? 

## BUGS
* The inner mutation, 

## TODOS
* Implement SSFM for pulse propagation modelling
* Build structure of both genetic algorithms