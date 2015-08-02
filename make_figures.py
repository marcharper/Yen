"""
Systematically produce many yen-related plots.
"""

import matplotlib
matplotlib.use('AGG')

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
        [[1, 0], [0, 1]], # neutral
        [[2, 2], [1, 1]], # class Moran
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
    N = 20
    ensure_directory(directory)
    for i, m in enumerate(two_type_matrices()):
        decomposition_bar_chart(N, m)
        filename = os.path.join(directory, "%s.png" % (i,))
        pyplot.savefig(filename)
        pyplot.clf()

# Three type populations

def heatmaps_bomze(N=40, mu=None, beta=0.1, directory="three_type_decompositions"):
    if not mu:
        mu = 1./N
    ensure_directory(directory)
    for i, m in enumerate(three_type_matrices()):
        for index_1, index_2 in [(0,1), (1,2), (2,0)]:
            fig = decomposition_heatmaps_3(N, m, mu=mu, beta=beta, index_1=index_1, index_2=index_2)
            filename = os.path.join(directory, "%s_%s_%s.png" % (i,index_1, index_2))
            pyplot.savefig(filename, dpi=600)
            pyplot.close(fig)
            pyplot.clf()

if __name__ == "__main__":
    decomposition_bar_charts()
    #heatmaps_bomze(N=20)

