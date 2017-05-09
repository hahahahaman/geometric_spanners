from __future__ import generators
import networkx as nx
import matplotlib.pyplot as plt
import math, pickle, random

random.seed()

def euclidean_distance(p, q):
    """Euclidean distance of p,q tuples"""
    return sum(((a-b)**2 for a,b in zip(p, q))) ** (1/2)

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
            d = euclidean_distance(pu,pv)
            if d <= radius:
                G.add_edge(u,v, weight=d)
    return G


def geometric_graph_stretch_factor(g, dimension=2):
    """
    INPUT: Geometric graph
    OUTPUT: stretch factor (disjoint graph will return INF)
    """

    pos = nx.get_node_attributes(g, 'pos')
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()

    min_path_lengths = 0

    # get minimum path between all vertices
    # dijkstra for all vertices is O(|V||E|log|V|)
    # floyd-warshall is O(|V|^3)
    if(num_edges < (num_nodes ** 2)/math.log2(num_nodes)):
        # print("dijkstra")
        min_path_lengths = nx.all_pairs_dijkstra_path_length(g)
    else:
        # print("floyd-warshall")
        min_path_lengths = nx.floyd_warshall(g)

    # print(min_path_lengths)

    max_stretch_factor = 0

    # print(dists[0][9])
    # print(euclidean_distance(pos[0], pos[9]))

    # get stretch factor between all points
    # and find max stretch factor
    for i in range(num_nodes-1):
        for j in range(i+1, num_nodes):
            stretch_factor = min_path_lengths[i][j] / euclidean_distance(pos[i], pos[j])

            # print(dists[i][j])
            # print(euclidean_distance(pos[i], pos[j]))

            # find maximum stretch factor
            if(stretch_factor > max_stretch_factor):
                max_stretch_factor = stretch_factor

    return max_stretch_factor

def random_points(n):
    points = []
    for i in range(n):
        points.append((random.random(),random.random()))
    return points

def random_convex_points(n):
    points = []
    totalx = 0
    totaly = 0
    for i in range(n-1):
        points.append((random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)))
        totalx += points[i][0]
        totaly += points[i][1]

    points.append((-totalx, -totaly))
    points.sort(key=lambda p: math.atan2(p[1],p[0]))

    totalx = 0
    totaly = 0
    convex_points = []
    for p in points:
        totalx += p[0]
        totaly += p[1]
        convex_points.append((totalx, totaly))
    return convex_points

def plot_edges(s, color='k', linestyle='-'):
    for i in range(len(s)-1):
        plt.plot([s[i][0], s[i+1][0]], [s[i][1], s[i+1][1]], color=color, linestyle=linestyle)

def matching(c1, c2, dists):
    """
    INPUT: Two linearly separated chains C1 and C2,
    such that the vertices of C1 U C2 are in convex position

    OUTPUT: A set of edges that forms a matching between the points
    of C1 and the points of C2
    """

    if(c1 == [] or c2 == []): return []

    closest = (c1[0], c2[0])
    indices = (0, 0)
    for i in range(len(c1)):
        a = c1[i]
        for j in range(len(c2)):
            b = c2[j]
            # print(a, ":", b)
            if dists[a][b] < dists[closest[0]][closest[1]]:
                closest = (a,b)
                indices = (i,j)

    # print(len(c1), ":", indices[0])
    # print(len(c2), ":", indices[1])

    c1x = c1[:indices[0]]
    c1y = c1[indices[0]+1:]

    c2x = c2[:indices[1]]
    c2y = c2[indices[1]+1:]
    # print("c1x:", c1x)
    # print("c1y:", c1y)
    # print("c2x:", c2x)
    # print("c2y:", c2y)
    return [closest] + matching(c1x, c2y, dists) + matching(c1y, c2x, dists)
    # return [closest] + matching(c1x, c2x, dists) + matching(c1y, c2y, dists)

'''
def diametrical_pair_indices(s):
    num_points = len(s)
    if(num_points == 0):
        return []

    p = 0
    q = 0
    max_dist = 0

    for i in range(num_points-1):
        for j in range(i+1, num_points):
            dist =  euclidean_distance(s[i], s[j])
            if(dist > max_dist):
                p = i
                q = j
                max_dist = dist

    return (p,q)
'''

def deg3_plane_spanner(s):
    U,L = hulls(s)
    graph = nx.Graph()

    # Upper hull and lower hull share the first and last vertices in their list
    # Lower hull is clockwise, must be reversed
    hull = U[1:] + list(reversed(L[:-1]))

    point_to_index = {}
    dists = {}
    for i in range(len(hull)):
        point_to_index[hull[i]] = i
        dists[hull[i]] = {}

    for i in range(len(hull)-1):
        for j in range(i+1, len(hull)):
            d = euclidean_distance(hull[i], hull[j])
            dists[hull[i]][hull[j]] = d
            dists[hull[j]][hull[i]] = d

    for i in range(len(hull)):
        graph.add_node(i, pos=hull[i])

        j = (i+1) % len(hull)
        graph.add_edge(i, j, weight=euclidean_distance(hull[i], hull[j]))

    dpair = diameter(U, L)
    dpair_index = (point_to_index[dpair[0]], point_to_index[dpair[1]])
    # for i in range(len(hull)):
    #     if(hull[i] == dpair[0] or hull[i] == dpair[1]):
    #         dpair_ind.append(i)

    # plot_edges(dpair, color='r', linestyle=':')

    # chains
    c1 = []
    c2 = []

    second = False
    for i in range(len(hull)-1):
        current = (i+dpair_index[0]+1)%len(hull)
        p = hull[current]

        if(current == dpair_index[1]):
            second = True
            continue

        if(second == True):
            c2.append(p)
        else:
            c1.append(p)

    matching_edges = matching(c1, c2, dists)

    for e in matching_edges:
        graph.add_edge(point_to_index[e[0]], point_to_index[e[1]],
                       weight=dists[e[0]][e[1]])

    return graph

# convex hull (monotone chain by x-coordinate) and diameter of a set of points
# David Eppstein, UC Irvine, 7 Mar 2002

def orientation(p,q,r):
    """Return positive if p-q-r are clockwise, neg if ccw, zero if colinear."""
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(Points):
    """Monotone chain to find upper and lower convex hulls of a set of 2d points."""
    U = []
    L = []
    # Points = Points[np.lexsort((Points[:,1],Points[:,0]))]
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotatingCalipers(U, L):
    """Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines."""
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j] # generated values

        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1

        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1

def diameter(U,L):
    """Given a list of 2d points, returns the pair that"s farthest apart."""
    diam,pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(U,L)])
    return pair

# plt.figure(figsize=(8,8))
# G = random_weighted_geometric_graph(100, 0.2)

graphs = []

n = 100
for i in range(n):
    print(i+1, "/", n)
    points = random_convex_points(i+10)
    G = deg3_plane_spanner(points)
    G.graph['stretch_factor'] = geometric_graph_stretch_factor(G)
    # pos = nx.get_node_attributes(G, 'pos')
    # print("number of nodes:", G.number_of_nodes())
    # print(G.graph['stretch_factor'])
    graphs.append(G)

# print(pos)

filename = 'graphs3.data'
with open(filename, "wb") as f:
    pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

# square
# G = nx.Graph()
# G.add_nodes_from(range(4))
# nx.set_node_attributes(G,'pos',{0:[0,0], 1:[0.5,0], 2:[0.5,0.5], 3:[0.0,0.5]})
# G.add_weighted_edges_from([(0,1,0.5), (1,2,0.5), (2,3,0.5), (3,0,0.5)])

#diamond
# G = nx.Graph()
# G.add_nodes_from(range(4))
# nx.set_node_attributes(G,'pos',{0:[0,0], 1:[0.3,0.3], 2:[0.5,0.5], 3:[0.299,0.3]})
# G.add_weighted_edges_from([(0,1,0.5), (1,2,0.5), (2,3,0.5), (3,0,0.5)]) #wrong

# print(geometric_graph_stretch_factor(G))

# pos = nx.get_node_attributes(G, 'pos')

### plotting

# #find point closest to the center(0.5,0.5)
# dmin = 1
# ncenter = 0
# for n in pos:
#     x,y = pos[n]
#     d=(x-0.5) ** 2 + (y-0.5) ** 2
#     if d < dmin:
#         ncenter = n
#         dmin = d

# #color by path length from node near center
# p=nx.single_source_shortest_path_length(G,ncenter)

# nx.draw_networkx_edges(G,pos,nodelist=[ncenter], alpha=0.4)
# nx.draw_networkx_nodes(G,pos,
#                        nodelist=p.keys(),
#                        node_size=80,
#                        node_color=list(p.values()),
#                        cmap=plt.cm.Oranges_r)

# nx.draw_networkx_edges(G, pos, alpha=0.4)
# nx.draw_networkx_nodes(G, pos, node_size=20, node_color=[0.5,0.5,0.7])

# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.axis('off')
# plt.savefig('deg3-spanner-algo1.png')
# plt.show()
