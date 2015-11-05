Yen
===

Python scripts for computing the yen and related quantities of trajectories
of finite Markov processes.

Examples: Two Types
-------------------

The yen can be decomposed into various interesting components. For fitness 
landscapes defined by the following matrices:

```
    matrices = [
        [[1, 1], [0, 1]], # tournament
        [[1, 1], [1, 1]], # neutral
        [[2, 2], [1, 1]], # class Moran
        [[1, 2], [2, 1]], # hawk-dove
        [[1, 3], [2, 1]], # asymmetric hawk-dove
        [[2, 1], [1, 2]], # coordination
    ]
```

We can plot the decompositions as follows:

<div style="text-align:center">
<img src ="/two_type_decompositions/0.png" width="50%"/><br/>
<img src ="/two_type_decompositions/1.png" width="50%"/><br/>
<img src ="/two_type_decompositions/2.png" width="50%"/><br/>
<img src ="/two_type_decompositions/3.png" width="50%"/><br/>
<img src ="/two_type_decompositions/4.png" width="50%"/><br/>
<img src ="/two_type_decompositions/5.png" width="50%"/><br/>
</div>

Examples: Three Types
---------------------

For populations of three types, there are three directions in which one can compute the yen. For a rock-paper-scissors matrix::

```
    [0, -1, 1]
    [1, 0, -1]
    [-1, 1, 0]
```

We have three decomposition directions::

<div style="text-align:center">
<img src ="/three_type_decompositions/16_0_1.png" width="50%"/><br/>
<img src ="/three_type_decompositions/16_1_2.png" width="50%"/><br/>
<img src ="/three_type_decompositions/16_2_0.png" width="50%"/><br/>
</div>

Stationary distributions and phase portraits are avaialable [here](http://people.mbi.ucla.edu/marcharper/stationary_stable/3x3/incentive.html).

We can also plot which "evolutionary force" contributes most to the action of
the dynamic at each point, encoded by the following colors:

* adaptation: red
* relative_fitness: green
* drift: blue
* mutation: purple

<div style="text-align:center">
<img src ="/three_type_max_decomp/16.png" width="80%"/><br/>
<img src ="/three_type_max_decomp/47.png" width="80%"/><br/>
</div>
