# Example Problem
The problem folder is meant to hold various optical simulators.
Currently, this only includes the time-frequency degree-of-freedom (called example).

To create a new simulator, there is some flexibility in how this is done, but it would be best to simply adapt 
existing functionality.
The main things to change are the Propagator. This is the fundamental properties of what dof is of interest.
This information is stored in the propagator.state variable.

Each node then alters the Propagator.state variable through the .propagate() function.



 
