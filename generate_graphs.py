import pickle
import algorithms as algo

graphs = []

n = 100
for i in range(n):
    print(i+1, "/", n)
    points = algo.random_convex_points(i+10)
    G = algo.deg3_plane_spanner(points)
    G.graph['stretch_factor'] = algo.geometric_graph_stretch_factor(G)
    # pos = nx.get_node_attributes(G, 'pos')
    # print("number of nodes:", G.number_of_nodes())
    # print(G.graph['stretch_factor'])
    graphs.append(G)

# print(pos)

filename = 'graphs.data'
with open(filename, "wb") as f:
    pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
