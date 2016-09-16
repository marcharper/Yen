
"""
Yen Search Algorithm:
- From a given initial state
    - compute the yen to each neighboring state
    - move in the direction of max or min (depending on extrema desired)
    - stop if all positive (negative)
- Improvement: use Boltzmann distribution to break ties and escape local extrema
"""
#
# # Standard Libraries
# from collections import defaultdict
# from itertools import izip, islice
# from math import log
# import os
#
# import matplotlib
# from matplotlib import pyplot
# import matplotlib.gridspec as gridspec
# import pandas

import numpy as np

# # Github libraries from https://github.com/marcharper
# import mpsim
# import stationary
# from stationary.processes import incentive_process
# from stationary.processes.incentives import replicator, fermi, linear_fitness_landscape
# from stationary.utils.bomze import bomze_matrices
# import ternary


def yen_search(initial_state, neighbor_function, transition_function,
               beta=None, extrema="max"):
    state = initial_state
    while True:
        print(state)
        neighbors = neighbor_function(state)
        yens = dict()
        for neighbor in neighbors:
            out_transition = transition_function(state, neighbor)
            in_transition = transition_function(neighbor, state)
            yen = np.log(out_transition) - np.log(in_transition)
            yens[neighbor] = yen
        # Check for extremum
        if extrema.lower() == "max":
            if all(y > 0 for y in yens.values()):
                return state
        else:
            if all(y < 0 for y in yens.values()):
                return state
        # Move to next state
        next_state = min(yens.keys(), key=(lambda key: yens[key]))
        state = next_state

def two_type_test():
    """Test the yen search algorithm on the neutral fitness landscape for a
    two type Moran process with mutation. The algorithm should converge to (5, 5)
    for any starting position."""
    def neighbor_function(state):
        i, j = state
        neighbors = []
        if i > 0:
            neighbors.append((i-1, j+1))
        if j > 0:
            neighbors.append((i+1, j-1))
        return neighbors

    def transition_function(source, target, mu=0.05):
        i1, j1 = source
        i2, j2 = target
        if i1 == 0:
            return mu
        if j1 == 0:
            return mu
        N = float(i1 + j1)
        if j2 > j1:
            return (i1 + (j1 - i1)* mu) / N ** 2 * j1
        if j1 > j2:
            return (j1 + (i1 - j1)* mu) / N ** 2 * i1

    state = (1, 9)
    e = yen_search(state, neighbor_function, transition_function)
    print(e)

    state = (9, 1)
    e = yen_search(state, neighbor_function, transition_function)
    print(e)

if __name__ == "__main__":
    two_type_test()
