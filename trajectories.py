
# Standard Libraries
from itertools import islice
from math import log
import os

# Major Third Party libraries
import matplotlib
from matplotlib import pyplot
import matplotlib.gridspec as gridspec
#import pandas

# Github libraries from https://github.com/marcharper
import stationary
from stationary.processes import incentive_process
from stationary.processes.incentives import replicator, fermi, linear_fitness_landscape
import mpsim


# Helpers

def edges_to_dictionary(edges):
    """
    Converts a list of edges to a dictionary for easy direct access.
    """

    d = dict()
    for (a,b,v) in edges:
        d[(a,b)] = v
    return d

# Yen, self-info, and friends along actual trajectories

def generate_trajectories(edges, initial_state, iterations=10, max_iterations=100, per_run=None):
    """
    Generates sample trajectories for a Markov process specified by edges using
    mpsim.

    Parameters
    -----------
    edges: list of tuples 
        The transitions of the process, a list of (source, target,
        transition_probability) tuples
    initial_state: a state of the process
        The initial state of the simulated trajectory
    iterations: integer
        The number of trajectories to compute
    max_iterations: integer
        the maximum length of each trajectory
    per_run: integer
        The number of trajectories to compute at a time

    Yields
    ------
    trajectory, a list of states of the process
    """

    if not per_run:
        per_run = iterations // 8
    cache = mpsim.compile_edges(edges)
    initial_state_generator = mpsim.generators.constant_generator(initial_state)
    iters_gen = mpsim.generators.iterations_generator(iterations)
    param_gen = mpsim.generators.parameter_generator(cache, initial_state_generator,
                                                     max_iterations=max_iterations)
    # Batched simulations using multiple processing cores if possible.
    runs = mpsim.batched_simulations(param_gen, iters_gen, processes=4)
    for seed, length, trajectory in runs:
        yield trajectory

def compute_yen(trajectory, transition_dict):
    """
    Computes the (cumulative) yen along a trajectory.
    """

    yen = 0
    for i in range(len(trajectory) - 1):
        source = trajectory[i]
        target = trajectory[i+1]
        forward_transtion = transition_dict[(source, target)]
        reverse_transtion = transition_dict[(target, source)]
        yen += log(reverse_transtion / forward_transtion)
    return -yen

#def compute_fitness_flux(trajectory, fitness_landscape):
    #initial_state = trajectory[0]
    #N = sum(initial_state)
    #final_state = trajectory[-1]
    #initial_fitness = dot_product(fitness_landscape(initial_state), initial_state)
    #final_fitness = dot_product(fitness_landscape(final_state), final_state)
    #phi = N*(log(final_fitness) - log(initial_fitness))
    ##phi = log(final_fitness) - log(initial_fitness)
    #return phi

def compute_fitness_flux(trajectory, incentive):
    """
    Computes the fitness flux of a trajectory.
    """

    initial_state = trajectory[0]
    N = sum(initial_state)
    final_state = trajectory[-1]
    initial_fitness = sum(incentive(initial_state))
    final_fitness = sum(incentive(final_state))
    phi = N*(log(final_fitness) - log(initial_fitness))
    #phi = log(final_fitness) - log(initial_fitness)
    return phi

def invert_enumeration(cache, ranks):
    """Inverts an enumeration."""
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d

def transition_matrix_power(edges, initial_state=None, power=20, yield_all=False):
    """
    Computes successive powers of the transition matrix associated to the
    Markov process defined by edges. Needed to compute the self-information.
    """

    g = stationary.utils.graph.Graph()
    g.add_edges(edges)
    cache = stationary.Cache(g)

    num_states = len(g.vertices())
    initial_state_ = [0]*(num_states)
    initial_state_[cache.enum[initial_state]] = 1
    gen = stationary.stationary_generator(cache, initial_state=initial_state_)

    yield invert_enumeration(cache, initial_state_)
    for i, ranks in enumerate(gen):
        if not yield_all:
            if i == power:
                break
        else:
            yield invert_enumeration(cache, ranks)
    yield invert_enumeration(cache, ranks)

def compute_self_info(trajectory, edges):
    """
    Computes the self-information of a trajectory.
    """

    length = len(trajectory)
    initial_state = trajectory[0]
    final_state = trajectory[-1]
    matrix_powers = list(islice(transition_matrix_power(edges, initial_state, power=length), 0, length))
    initial_self_info = -log(matrix_powers[0][initial_state])
    final_self_info = -log(matrix_powers[-1][final_state])
    return final_self_info - initial_self_info

def histograms(yens, fluxes, self_infos, bins=30):
    """
    Makes histograms of yens, fitness fluxes, and self-informations.
    """

    num_values = len(yens)
    diffs = []
    for i in range(num_values):
        diffs.append(yens[i] - self_infos[i])

    # Make four plots in a 4x1 grid
    gs = gridspec.GridSpec(4, 1)
    ax1 = pyplot.subplot(gs[0, 0])
    ax1.set_title("Yen")
    ax2 = pyplot.subplot(gs[1, 0])
    ax2.set_title("Fitness Flux $\\Phi$")
    ax3 = pyplot.subplot(gs[2, 0])
    ax3.set_title("Self-info $\\Delta S$")
    ax4 = pyplot.subplot(gs[3, 0])
    ax4.set_title("Yen - $\\Delta S$")
    ax1.hist(yens, bins=bins)
    ax2.hist(fluxes, bins=bins)
    ax3.hist(self_infos, bins=bins)
    ax4.hist(diffs, bins=bins)

def compute_everything(N, m, mu, initial_state, num_trajectories=100, trajectory_length=100, incentive_func=replicator, beta=1.):
    # Get the edges of the Markov process
    edges = incentive_process.compute_edges(N=N, m=m, mu=mu, beta=beta,
                                            incentive_func=incentive_func)

    transition_dict = edges_to_dictionary(edges)
    # Generate some trajectories
    trajectories = list(generate_trajectories(edges, initial_state, iterations=num_trajectories, max_iterations=trajectory_length))

    # Compute yens along the trajectories
    yens = [compute_yen(trajectory, transition_dict) for trajectory in trajectories]

    # Compute the fitness flux
    fitness_landscape = linear_fitness_landscape(m)
    try:
        incentive = incentive_func(fitness_landscape, beta=beta)
    except TypeError :
        incentive = incentive_func(fitness_landscape)
    fluxes = [compute_fitness_flux(trajectory, incentive) for trajectory in trajectories]
    # Compute the self-informations
    self_infos = [compute_self_info(trajectory, edges) for trajectory in trajectories]

    return (yens, fluxes, self_infos)

if __name__ == "__main__":

    # Compute yen and friends on actual trajectories
    # Two types
    N = 20
    mu = 1./N
    m = [[1,1],[0,1]]
    i = N // 4
    initial_state = (i, N-i)
    (yens, fluxes, self_infos) = compute_everything(N, m, mu, initial_state, num_trajectories=1000, trajectory_length=10*N)
    histograms(yens, fluxes, self_infos)
    pyplot.show()

    # Three types
    N = 40
    mu = 1./N
    m = [[1,1,1], [0,1,1], [0,0,1]]
    i = N//4
    initial_state = (i, i, N-2*i)
    (yens, fluxes, self_infos) = compute_everything(N, m, mu, initial_state, incentive_func=fermi, num_trajectories=40, trajectory_length=10*N)
    histograms(yens, fluxes, self_infos)
    pyplot.show()

