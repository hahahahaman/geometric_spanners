import networkx as nx
import matplotlib.pyplot as plt
import math, random

# what is a geometric graph?
# set of points in euclidean space
# I will use networkx Graph object
def random_weighted_geometric_graph(n, radius, dim=2, pos=None):
    r"""Return the random geometric graph in the unit cube.

    The random geometric graph model places n nodes uniformly at random
    in the unit cube  Two nodes `u,v` are connected with an edge if
    `d(u,v)<=r` where `d` is the Euclidean distance and `r` is a radius
    threshold.

    Parameters
    ----------
    n : int
        Number of nodes
    radius: float
        Distance threshold value
    dim : int, optional
        Dimension of graph
    pos : dict, optional
        A dictionary keyed by node with node positions as values.

    Returns
    -------
    Graph

    Examples
    --------
    >>> G = nx.random_geometric_graph(20,0.1)

    Notes
    -----
    This uses an `n^2` algorithm to build the graph.  A faster algorithm
    is possible using k-d trees.

    The pos keyword can be used to specify node positions so you can create
    an arbitrary distribution and domain for positions.  If you need a distance
    function other than Euclidean you'll have to hack the algorithm.

    E.g to use a 2d Gaussian distribution of node positions with mean (0,0)
    and std. dev. 2

    >>> import random
    >>> n=20
    >>> p=dict((i,(random.gauss(0,2),random.gauss(0,2))) for i in range(n))
    >>> G = nx.random_geometric_graph(n,0.2,pos=p)

    References
    ----------
    .. [1] Penrose, Mathew, Random Geometric Graphs,
       Oxford Studies in Probability, 5, 2003.
    """
    G=nx.Graph()
    G.name="Random Geometric Graph"
    G.add_nodes_from(range(n))
    if pos is None:
        # random positions
        for n in G:
            G.node[n]['pos']=[random.random() for i in range(0,dim)]
    else:
        nx.set_node_attributes(G,'pos',pos)
    # connect nodes within "radius" of each other
    # n^2 algorithm, could use a k-d tree implementation
    nodes = G.nodes(data=True)
    while nodes:
        u,du = nodes.pop()
        pu = du['pos']
        for v,dv in nodes:
            pv = dv['pos']
            d = sum(((a-b)**2 for a,b in zip(pu,pv)))
            if d <= radius**2:
                G.add_edge(u,v, weight=d)
    return G

def euclidean_distance(p, q, dimension=2):
    """Euclidean distance of p,q tuples of given dimension"""
    return sum(((a-b)**2 for a,b in zip(p, q)))

def geometric_spanner_stretch_factor(g, dimension=2):
    """INPUT: Geometric spanner, OUTPUT: stretch factor"""

    pos = nx.get_node_attributes(g, 'pos')
    dists = nx.floyd_warshall(g)

    print(dists)

    max_stretch_factor = 0

    # print(dists[0][9])
    # print(euclidean_distance(pos[0], pos[9]))
    nodes = g.number_of_nodes()
    for i in range(nodes):
        for j in range(nodes):
            if(i == j):
                continue

            stretch_factor = dists[i][j] / euclidean_distance(pos[i],
                                                              pos[j],
                                                              dimension)

            if(stretch_factor > max_stretch_factor):
                max_stretch_factor = stretch_factor
    return max_stretch_factor

G = random_weighted_geometric_graph(10, 10)
print(geometric_spanner_stretch_factor(G))
# print(euclidean_distance([0,-1], [0,1]))
