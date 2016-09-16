import copy
import random

import numpy as np

from stationary.processes.incentives import replicator, fermi, linear_fitness_landscape


def yen_search(initial_state, neighbor_function, transition_function,
               beta=None, extrema="max"):
    """
    Yen Search Algorithm:
    - From a given initial state
        - compute the yen to each neighboring state
        - move in the direction of max or min (depending on extrema desired)
        - stop if all positive (negative)
    - Improvement: use Boltzmann distribution to break ties and escape local extrema
    """

    visited_states = []

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
        # Could use the Boltzmann distribution here for some stochasticity
        visited_states.append(state)
        min_value = min(yens.values())
        min_keys = [k for k in yens.keys() if yens[k] == min_value]
        next_state = random.choice(min_keys)
        state = next_state
        if state in visited_states:
            return "Repeated State"

def two_type_test():
    """Test the yen search algorithm on the neutral fitness landscape for a
    two type Moran process with mutation. The algorithm should converge to (5, 5)
    for any starting position."""

    def neighbor_function(state):
        # Neighbors of (i, N-i) are (i+1, N-i-1) and (i-1, N-i+1)
        i, j = state
        neighbors = []
        if i > 0:
            neighbors.append((i-1, j+1))
        if j > 0:
            neighbors.append((i+1, j-1))
        return neighbors

    def transition_function(source, target, mu=0.05):
        # Neutral fitness landscape
        i1, j1 = source
        i2, j2 = target
        if i1 == 0:
            return mu
        if j1 == 0:
            return mu
        N = float(i1 + j1)
        if j2 > j1:
            return (i1 + (j1 - i1) * mu) / N ** 2 * j1
        if j1 > j2:
            return (j1 + (i1 - j1) * mu) / N ** 2 * i1

    state = (1, 9)
    e = yen_search(state, neighbor_function, transition_function)
    print(e)

    state = (9, 1)
    e = yen_search(state, neighbor_function, transition_function)
    print(e)


def graph_test():
    """Test yen search on large process -- a two type neutral Moran process on
    a cycle."""
    def neighbor_function(state):
        """States in this case are a list of ones and zeroes, i.e. a graph
        coloring."""
        neighbors = []
        for i, t in enumerate(state):
            neighbor = copy.copy(list(state))
            if t == 0:
                neighbor[i] = 1
            else:
                neighbor[i] = 0
            neighbors.append(tuple(neighbor))
        return neighbors

    # print(neighbor_function([0, 1, 0]))

    def transition_function(source, target, mu=0.05):
        # Find the state that differs
        for i, (s, t) in enumerate(zip(source, target)):
            if s != t:
                break
        # i is now the index of the state that differs
        # Look at neighboring states (within the cycle) to determine transition
        N = len(source)
        indices = [i-1, i+1]
        transition = 0.
        for index in indices:
            if source[index % N] == s:
                transition += 1 - mu
            else:
                transition += mu
        return transition

    state = [0, 1, 0, 1, 0, 0]
    e = yen_search(state, neighbor_function, transition_function)
    print(e)

    state = [0, 1, 0, 1, 1, 1]
    e = yen_search(state, neighbor_function, transition_function)
    print(e)

    # state = [0, 0, 0, 1, 1, 1]
    # e = yen_search(state, neighbor_function, transition_function)
    # print(e)

    # Non-neutral landscape test

    m = [[1,2], [2,1]] # Hawk-Dove
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)

    def transition_function2(source, target, mu=0.01):
        """Non-neutral landscape specified by a game matrix (above)"""
        # Find the state that differs
        for i, (s, t) in enumerate(zip(source, target)):
            if s != t:
                break
        # i is now the index of the state that differs
        s = sum(source)
        N = len(source)
        population_state = (N - s, s)
        inc = incentive(population_state)
        denom = float(sum(inc))
        indices = [i-1, i+1]
        transition = 0.
        for index in indices:
            rep_type = source[index % N]
            r = float(inc[rep_type]) / denom
            if rep_type == s:
                transition += r * (1 - mu)
            else:
                transition += r * mu
        return transition

    state = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    e = yen_search(state, neighbor_function, transition_function2)
    print(e)

    state = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1]
    e = yen_search(state, neighbor_function, transition_function2)
    print(e)

    state = [0, 1] * 8
    e = yen_search(state, neighbor_function, transition_function2)
    print(e)

    state = [0, 1] * 256
    e = yen_search(state, neighbor_function, transition_function2)
    print(e)

    state = [0] * 256 + [1] * 256
    e = yen_search(state, neighbor_function, transition_function2)
    print(e)


if __name__ == "__main__":
    # two_type_test()
    graph_test()