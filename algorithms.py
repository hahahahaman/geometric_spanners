import networkx as nx
import matplotlib.pyplot as plt
import math

# what is a geometric graph?
# set of points in euclidean space
# I will use networkx Graph object

def euclidean_distance(p, q, dimension=2):
    sum = 0
    for i in range(dimension):
        sum += (p[i] - q[i]) ** 2
    return sum ** (1/2)

def geometric_spanner_stretch_factor(g, dimension=2):
    """INPUT: Geometric spanner, OUTPUT: stretch factor"""

    pos = nx.get_node_attributes(g, 'pos')
    dists = nx.floyd_warshall(g)

    print(dists)

    max_stretch_factor = 0

    print(dists[0][9])
    print(euclidean_distance(pos[0], pos[9]))
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

G = nx.random_geometric_graph(10, 0.125)
print(geometric_spanner_stretch_factor(G))
# print(euclidean_distance([0,-1], [0,1]))
