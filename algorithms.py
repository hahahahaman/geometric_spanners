import networkx as nx
import matplotlib.pyplot as plt
import math, random

# what is a geometric graph?
# set of points in euclidean space
# I will use networkx Graph object

def euclidean_distance(p, q):
    """Euclidean distance of p,q tuples"""
    return sum(((a-b)**2 for a,b in zip(p, q)))

def random_weighted_geometric_graph(n, radius, dim=2, pos=None):
    """Return the random geometric graph in the unit cube.

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
            d = euclidean_distance(pu, pv)
            if d <= radius**2:
                G.add_edge(u,v, weight=d)
    return G


def geometric_graph_stretch_factor(g, dimension=2):
    """
    INPUT: Geometric graph
    OUTPUT: stretch factor (disjoint graph will return INF)
    """

    pos = nx.get_node_attributes(g, 'pos')
    dists = nx.floyd_warshall(g) # get minimum path between all vertices

    # print(dists)

    max_stretch_factor = 0

    # print(dists[0][9])
    # print(euclidean_distance(pos[0], pos[9]))

    num_nodes = g.number_of_nodes()

    # get stretch factor between all points
    # and find max stretch factor
    for i in range(num_nodes-1):
        for j in range(i+1, num_nodes):
            stretch_factor = dists[i][j] / euclidean_distance(pos[i], pos[j])

            # find maximum stretch factor
            if(stretch_factor > max_stretch_factor):
                max_stretch_factor = stretch_factor

    return max_stretch_factor

def matching(c1, c2):
    if(c1.number_of_nodes == 0 or c2.number_of_nodes == 0):
        return nx.Graph()

g = random_weighted_geometric_graph(100, 0.2)
print(geometric_graph_stretch_factor(G))

# print(euclidean_distance([0,-1], [0,1]))
