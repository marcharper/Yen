"""
Systematically produce many yen-related plots.
"""

import math

import matplotlib
#matplotlib.use('AGG')
font = {'size': 20}
matplotlib.rc('font', **font)
from matplotlib import pyplot
import colormaps as cmaps
pyplot.register_cmap(name='viridis', cmap=cmaps.viridis)
pyplot.set_cmap(cmaps.viridis)

from decompositions import *


def ensure_directory(directory):
    """Checks if a directory exists, if not makes it."""
    if not os.path.isdir(directory):
        os.mkdir(directory)

def ensure_digits(num, s):
    """Prepends a string s with zeros to enforce a set num of digits."""
    if len(s) < num:
        return "0"*(num - len(s)) + s
    return s

# Sample matrices for fitness landscapes

def two_type_matrices():
    matrices = [
        [[1, 1], [0, 1]], # tournament
        [[1, 1], [1, 1]], # neutral
        [[2, 2], [1, 1]], # classic Moran
        [[1, 2], [2, 1]], # hawk-dove
        [[1, 3], [2, 1]], # asymmetric hawk-dove
        [[2, 1], [1, 2]], # coordination
    ]
    return matrices

def three_type_matrices():
    """Returns the matrices in I.M. Bomze's classifications."""
    matrices = list(bomze_matrices())
    return matrices

# Yen Decompositions Figures

# Two type populations

def decomposition_bar_charts(N=30, directory="two_type_decompositions"):
    # Decomposition Bar Charts, two types
    ensure_directory(directory)
    for i, m in enumerate(two_type_matrices()):
        decomposition_bar_chart(N, m)
        filename = os.path.join(directory, "%s.png" % (i,))
        pyplot.savefig(filename)
        pyplot.clf()

# Three type populations

def heatmaps_bomze(N=40, mu=None, beta=0.1, directory="three_type_decompositions"):
    if not mu:
        mu = 3. / (2 * N)
    ensure_directory(directory)
    matrices = list(three_type_matrices())
    for i, m in enumerate(matrices):
        for index_1, index_2 in [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]:
            print i, index_1, index_2
            fig = decomposition_heatmaps_3(N, m, mu=mu, beta=beta, index_1=index_1, index_2=index_2)
            j = ensure_digits(2, str(i))
            filename = os.path.join(directory, "%s_%s_%s.png" % (j, index_1, index_2))
            pyplot.savefig(filename, dpi=200)
            pyplot.close(fig)
            pyplot.clf()

def max_decomp_plots(N=40, mu=None, beta=0.1, directory="three_type_max_decomp"):
    if not mu:
        #mu = 1./N
        mu = 3. / (2 * N)
    ensure_directory(directory)
    matrices = list(three_type_matrices())
    for i, m in enumerate(matrices):
        print i
        fig = decomposition_maximum_component_figure(N, m, mu=mu, beta=beta)
        j = ensure_digits(2, str(i))
        filename = os.path.join(directory, "%s.png" % (j,))
        pyplot.savefig(filename, dpi=200)
        pyplot.close(fig)
        pyplot.clf()


def max_decomp_test(N=30, mu=None, beta=0.1, directory="three_type_max_decomp"):
    if not mu:
        #mu = 1./N
        mu = 3. / (2 * N)
    ensure_directory(directory)
    matrices = list(three_type_matrices())
    m = matrices[7]
    fig = decomposition_maximum_component_figure(N, m, mu=mu, beta=beta, cmap=cmaps.viridis)
    filename = os.path.join(directory, "test.png")
    pyplot.savefig(filename, dpi=400)
    pyplot.close(fig)
    pyplot.clf()


if __name__ == "__main__":
    print "Generating figures -- this will take some time."
    decomposition_bar_charts(N=40)
    heatmaps_bomze(N=60)
    max_decomp_plots(N=60)


    N = 60
    mu = 1./ math.pow(N, 1. / 2)
    m = list(bomze_matrices())[16]
    figure = decomposition_heatmaps_3(N=N, m=m, mu=mu, beta=1, index_1=0, index_2=1)
    pyplot.show()

    #max_decomp_test(N=60)

