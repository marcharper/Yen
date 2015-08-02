"""
Compute decompositions of yen into interesting components for the Moran process.
"""

# Standard Libraries
from collections import defaultdict
from math import exp, log
import os

# Major Third Party libraries
import matplotlib
from matplotlib import pyplot
import matplotlib.gridspec as gridspec
import pandas

# Github libraries from https://github.com/marcharper
import stationary
from stationary.processes import incentive_process
from stationary.processes.incentives import replicator, fermi, linear_fitness_landscape
from stationary.utils.bomze import bomze_matrices
import ternary


# Helpers

def dot_product(a, b):
    """Dot product of two lists."""
    c = 0
    for i in range(len(a)):
        c += a[i] * b[i]
    return c

def apply_beta(fitness_landscape, beta=1.):
    """Exponentiates a fitness landscape to avoid divisions by zero in
    fitness proportionate selection. This is similar to using a Fermi
    incentive without the normalization."""

    def func(pop):
        return map(lambda x: exp(beta*x), fitness_landscape(pop))
    return func

## Yen decomposition figures

# Two-type Populations

def yen_decompositions_2(N, fitness_landscape, mu):
    """
    Computes the individual components of yen for the Moran process for two
    types. Assumes small mutations \mu for valid Taylor expansion.

    Parameters
    ----------
    N, integer
        The population size.
    fitness_landscape, function
        The fitness landscape, a function on population states (a,b)
    mu, float
        The mutation rate mu

    Returns
    -------
    results, dictionary of lists of computed values
    """

    results = defaultdict(list)
    domain = list(range(1, N-1))
    results["domain"] = domain

    # Compute the components for each population state
    for a in domain:
        # We're looking at positive transitions
        pop = (a, N - a)
        pop2 = (a + 1, N - a - 1)

        # Drift term
        drift = - log(float(pop2[1] * pop2[0]) / (pop[0] * pop[1]))
        results["drift"].append(drift)

        # Adaptation / mean fitness term
        mean_1 = dot_product(fitness_landscape(pop), pop)
        mean_2 = dot_product(fitness_landscape(pop2), pop2)
        adaptation = log(float(mean_2) / mean_1)
        results["adaptation"].append(adaptation)

        # Relative fitness term
        relative_fitness = log(float(fitness_landscape(pop)[0]) / fitness_landscape(pop2)[1])
        results["relative_fitness"].append(relative_fitness)

        # Mutation term
        mutation = mu * (float(pop[1]) / pop[0] * fitness_landscape(pop)[1] / fitness_landscape(pop)[0] - pop2[0] / float(pop2[1]) * fitness_landscape(pop2)[0] / fitness_landscape(pop2)[1])
        results["mutation"].append(mutation)

        # Yen (sum of all terms)
        yen = drift + adaptation + relative_fitness + mutation
        results["yen"].append(yen)

    return results

def decomposition_bar_chart(N, m, mu=None):
    """
    Plots the Yen decomposition along with the stationary distribution for
    two type Moran processes with mutation. Note: valid for small mu only!
    (Taylor expansion)

    Parameters
    ----------
    N, integer
        The population size.
    m, 2x2 matrix
        The matrix used to compute a fitness landscape,
        a function on population states (a,b)
    mu, float
        The mutation rate mu, defaults to 1 / N
    """

    if not mu:
        mu = 1. / N

    fitness_landscape = linear_fitness_landscape(m)

    # Compute the stationary distribution
    edges = incentive_process.compute_edges(N=N, m=m, mu=mu,
                                            incentive_func=replicator)
    s = stationary.stationary_distribution(edges)

    # Plot figures on a 2 x 1 grid
    pyplot.figure()
    gs = gridspec.GridSpec(3,1)
    ax1 = pyplot.subplot(gs[0:2,0])
    ax2 = pyplot.subplot(gs[2,0])

    # Compute the decompositions
    data = yen_decompositions_2(N, fitness_landscape, mu)
    # Don't include the domain or yens in the stacked bars
    domain = data["domain"]
    del data["domain"] # Remove from the data dictionary
    yens = data["yen"]
    del data["yen"] # Remove from the data dictionary

    # Yen Figure
    # Plot the sum (yen) as a curve
    ax1.plot(yens, color="black", linewidth=2)
    ax1.set_title("Yen Decomposition")
    # Plot the yen components in a stacked bar chart
    df = pandas.DataFrame(data=data, index=domain)
    df.plot(kind='bar', stacked=True, ax=ax1)

    # Stationary distribution figure
    # Convert from dictionary to list
    l = [0] * (N+1)
    for state, value in s.items():
        a = state[0] # a of state (a, b)
        l[a] = value

    ax2.plot(range(len(l)), l)
    ax2.set_title("Stationary Distribution")
    ax2.set_xlabel("Population state (a, N-a)")

# Three-type Populations

def remove_boundary(d, N):
    """
    Removes the boundary of the 2-simplex for plotting purposes.
    """

    for (i,j) in d.keys():
        if i == 0 or j == 0 or (i+j) == N:
            del d[(i,j)]
    return d

def simplex_iterator(scale, boundary=True):
    """
    Systematically iterates through a lattice of points on the 2-simplex.

    Parameters
    ----------
    scale: Int
        The normalized scale of the simplex, i.e. N such that points (x,y,z)
        satisify x + y + z == N

    boundary: bool, True
        Include the boundary points (tuples where at least one
        coordinate is zero)

    Yields
    ------
    3-tuples, There are binom(n+2, 2) points (the triangular
    number for scale + 1, less 3*(scale+1) if boundary=False
    """

    start = 0
    if not boundary:
        start = 1
    for i in range(start, scale + (1 - start)):
        for j in range(start, scale + (1 - start) - i):
            k = scale - i - j
            yield (i, j, k)

def population_from_indices(types, indices):
    """
    Computes the new population state resulting from an increase in
    index[0] and a decrease in index[1]. This yields the two population
    states of a transition, e.g. (a, b, c) to (a+1, b-1, c)

    Parameters
    ----------
    types, a list of three integers
        Representing the number of each type e.g. the population state (a, b, c)
    indices, a list of three values 0,1,2 in some order
        index[0] is the type to increase, index[1] is the type to decrease
        index[2] is unchanged.

    Returns
    -------
    pop1, pop2, two population states (a, b, c)
        The source and target of a transition.
    """

    pop1 = tuple(types)
    pop2 = list(tuple(types))
    pop2[indices[0]] += 1
    pop2[indices[1]] -= 1
    return pop1, pop2

def yen_decompositions_3(N, fitness_landscape, mu, index_1, index_2):
    """
    Computes the individual components of yen for the Moran process for three
    types. Assumes small mutations \mu for valid Taylor expansion.

    Parameters
    ----------
    N, integer
        The population size.
    fitness_landscape, function
        The fitness landscape, a function on population states (a,b)
    mu, float
        The mutation rate mu
    index_1, integer in [0, 1, 2]
        The population type to increase
    index_2, integer in [0, 1, 2]
        The population type to decrease

    Returns
    -------
    results, dictionary of lists of computed values
    """

    # The yen calculation is for adjacent states, which involves increasing one
    # index (index_1) and decreasing another (index_2), e.g. (a, b, c) to
    # (a+1, b-1, c). For the calculations, we also need to know which index is
    # unchanged.
    indices = set([0,1,2])
    indices.remove(index_1)
    indices.remove(index_2)
    index_3 = list(indices)[0]
    indices = [index_1, index_2, index_3]

    # Collect the results into a dictionary
    results = defaultdict(dict)

    # Compute the yen decomposition components for transitions originating at
    # each population state
    domain = list(range(0, N+1))
    for types in simplex_iterator(N):
        try:
            # First obtain the source and target population states
            pop1, pop2 = population_from_indices(types, indices)

            # Compute yen components
            # Drift term
            drift = -log(float(pop2[index_2] * pop2[index_1]) / (pop1[index_2] * pop1[index_1]))
            results["drift"][pop1] = drift

            # Adaptation / mean fitness term
            mean_1 = dot_product(fitness_landscape(pop1), pop1)
            mean_2 = dot_product(fitness_landscape(pop2), pop2)
            adaptation = log(mean_2 / mean_1)
            results["adaptation"][pop1] = adaptation

            # Relative fitness term
            fitness_1 = [fitness_landscape(pop1)[i] for i in [0,1,2]]
            fitness_2 = [fitness_landscape(pop2)[i] for i in [0,1,2]]
            r = log(float(fitness_1[index_1]) / fitness_2[index_2])
            results["relative_fitness"][pop1] = r

            # Mutation term
            mutation = mu *( (pop1[index_2]/2. * fitness_1[index_2] + pop1[index_3]/2.* fitness_2[index_3]) / pop1[index_1]*fitness_1[index_1] - ((pop2[index_1])/2 * fitness_2[index_1] + pop2[index_3]/2.*fitness_2[index_3])/((pop2[index_2])*fitness_1[index_2]) )
            results["mutation"][pop1] = mutation

            # Yen (sum of all terms)
            yen = drift + adaptation + r + mutation
            results["yen"][pop1] = yen
        except:
            # Division by zero or ValueError from the boundary
            continue
    return results

def decomposition_heatmaps_3(N, m, mu=None, incentive_func=fermi, beta=0.1, index_1=0, index_2=1):
    """
    Plots the Yen decomposition along with the stationary distribution for
    three type Moran processes with mutation. Note: valid for small mu only!
    (Taylor expansion)
    """

    if not mu:
        mu = 1./N

    fitness_landscape = apply_beta(linear_fitness_landscape(m), beta=beta)

    # Compute the stationary distribution
    edges = incentive_process.compute_edges(N=N, m=m, mu=mu, num_types=3,
                                            incentive_func=incentive_func,
                                            beta=beta)
    s = stationary.stationary_distribution(edges)

    # Compute the yen decompositions
    results = yen_decompositions_3(N, fitness_landscape, mu, index_1, index_2)

    # Plot everything in a 3x3 grid
    figure = pyplot.figure(figsize=(30, 18))
    gs = gridspec.GridSpec(2, 3)

    plot_params = [(0, 0, "Adaptation", results["adaptation"]),
                   (1, 0, "Relative fitness", results["relative_fitness"]),
                   (0, 1, "Drift", results["drift"]),
                   (1, 1, "Mutation", results["mutation"]),
                   (0, 2, "Yen", results["yen"]),
                   (1, 2, "Stationary", s)]

    for i,j, title, data in plot_params:
        ax = pyplot.subplot(gs[i, j])
        ternary_ax = ternary.TernaryAxesSubplot(ax=ax, scale=N)
        ternary_ax.heatmap(data, style="hexagonal", scientific=True)
        ternary_ax.set_title(title)
    return figure


if __name__ == "__main__":
    ## Examples
    # Two types
    N = 40
    mu = 1. / N
    m = [[1, 3], [2, 1]]
    decomposition_bar_chart(N, m, mu)
    pyplot.show()

    # Three types
    N = 20
    mu = 3. / (2 * N)
    beta = 0.1
    m = [[1,1,1], [0,1,1], [0,0,1]]

    decomposition_heatmaps_3(N, m, mu=mu, beta=beta, index_1=0, index_2=1)
    pyplot.show()
