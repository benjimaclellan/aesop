# as a temporary measure, copy this into the top level directory to run

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from problems.example.evaluator_subclasses.case_evaluator import CaseEvaluator
from problems.example.assets.propagator import Propagator

"""
The goal of this script is to establish the runtimes of the fitness functions offered in case_evaluator.py, 
and see how said runtimes scale with the number of bits N in our input string.

Note that the number of samples per input does not really matter, since it simplies multiplies the input length N
by a constant (for a given problem) coefficient. For the purposes of this test, we shall set the bit width (in samples)
to 1.

This benchmarking IGNORES the runtime of the propagator through the graph itself (which will be the same for each fitness function)
"""

# set up testing parameters
BIT_WIDTH = 10 # 1 samples for 1 bit
EXPONENTS_TO_TEST = range(0, 18) # we shall test bit lengths which are powers of two between [8, 2^32)
# FUNCTIONS_TO_TEST_STR = ['l1', 'l2', 'max eye', 'BER pure alternate'] # aka the nice quick ones
FUNCTIONS_TO_TEST_STR = ['BER with mask'] # just for testing find points in path function


# set up result collection
results = {}
findpoint_runtime = np.zeros(len(EXPONENTS_TO_TEST))
for name in FUNCTIONS_TO_TEST_STR:
    results[name] = np.zeros(len(EXPONENTS_TO_TEST))


# run tests
for i in EXPONENTS_TO_TEST:
    print(i)
    # create arbitrary binary sequence of correct length
    # See: https://stackoverflow.com/questions/43528637/create-large-random-boolean-matrix-with-numpy
    bit_seq_length = 2**i
    bit_seq = np.random.choice(a=[False, True], size=bit_seq_length, p=[0.5, 0.5])
    mocked_graph = np.random.choice(a=[False, True], size=bit_seq_length * BIT_WIDTH, p=[0.5, 0.5]) 

    # setup graph, propagator, evaluator (window_t, central_wl do not really matter at all)
    propagator = Propagator(n_samples=BIT_WIDTH*bit_seq_length, window_t=1e-9, central_wl=1)
    evaluator = CaseEvaluator(propagator, bit_seq, BIT_WIDTH, findpoints_runtimes=True)

    # run each test
    for name in FUNCTIONS_TO_TEST_STR:
        evaluator.evaluate_graph(mocked_graph, name, mocking_graph=True)
        results[name][i] = evaluator.runtimes[name]
    
    findpoint_runtime[i] = evaluator.find_points_runtime


# display results
x_axis = np.array([2**i for i in EXPONENTS_TO_TEST])
fig, ax = plt.subplots()
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
linestyle = cycle(['-', '--', '-.', ':'])
for name in FUNCTIONS_TO_TEST_STR:
    line = {'color':next(colors), 'lw':2, 'ls':next(linestyle)}
    ax.plot(x_axis, results[name], label=name, **line)

ax.set_xlabel('length of bit sequence (bits)')
ax.set_ylabel('runtime (s)')
ax.legend()
plt.title('Runtimes of different fitness functions')
plt.show()

# display runtime ratio if that's a thing
fig, ax = plt.subplots()
ax.plot(x_axis[3:], findpoint_runtime[3:] / results['BER with mask'][3:])
plt.title('findpoint runtime over total runtime vs input bits')
plt.show()
