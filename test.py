
import unittest

import matplotlib.pyplot as plt
plt.style.use(r"config/plot-style.mplstyle")

import config.config as configuration

from problems.example.node_types import Input
from problems.example.evaluator import Evaluator
from problems.example.graph import Graph
from problems.example.assets.propagator import Propagator


class Test(unittest.TestCase):
    def test_all_available_nodes(self):
        propagator = Propagator(window_t=100e-12, n_samples=2 ** 14)
        for node_type, nodes in configuration.NODE_TYPES_ALL.items():
            print('Testing {} node-types'.format(node_type))
            for node_name, node in nodes.items():
                tmp = node().propagate([propagator.state], propagator)
                print('\tTesting {} node'.format(node_name))
            print('\n')
        return

if __name__ == "__main__":
    unittest.main()