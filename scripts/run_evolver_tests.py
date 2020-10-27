import matplotlib.pyplot as plt

# necessary for successful imports below -------------------
import sys
import pathlib
import os
import platform
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)
# ----------------------------------------------------------

from problems.example.graph import Graph

from problems.example.node_types_subclasses.inputs import ContinuousWaveLaser, PulsedLaser
from problems.example.node_types_subclasses.outputs import MeasurementDevice, Photodiode
from problems.example.node_types_subclasses.single_path import PhaseModulator, WaveShaper, EDFA, CorningFiber, VariableOpticalAttenuator
from problems.example.node_types_subclasses.multi_path import VariablePowerSplitter

from problems.example.evolution_operators.evolution_operators import SwapNode

"""
These tests are only really around for debugging: the implementation isn't terribly complicated
so we don't expect to have to repeatedly test it
"""

def test_evo_ops():
    nodes = {0:ContinuousWaveLaser(),
             1:PhaseModulator(),
             2:WaveShaper(),
            -1:MeasurementDevice()}
    edges = [(0, 1), (1, 2), (2, -1)]
    graph = Graph(nodes, edges, propagate_on_edges=False)
    graph.assert_number_of_edges()

    swap_op = SwapNode()
    swap_op.apply_evolution(graph)
    graph.draw()
    plt.show()


def test_equal_prob():
    # TODO: sample repeatedly, make a histogram (visual)
    pass


def test_offset_prob():
    # TODO: sample repeatedly, make a histogram
    pass


def test_time_dependent_prob():
    # TODO: histogram, where time dependence is a step function (easy to see)
    pass 

if __name__=="__main__":
    test_evo_ops()
    # test_equal_prob()
    # test_offset_prob()
    # test_time_dependent_prob()