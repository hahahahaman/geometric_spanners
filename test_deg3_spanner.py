import networkx as nx
import algorithms
import matplotlib.pyplot as plt

n = 10

points = algorithms.random_convex_points(n)
G = algorithms.deg3_plane_spanner(points)
G.graph['stretch_factor'] = algorithms.geometric_graph_stretch_factor(G)
pos = nx.get_node_attributes(G, 'pos')

print("number of nodes:", G.number_of_nodes())
print("stretch factor", G.graph['stretch_factor'])

plt.figure(figsize=(8,8))

nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(G, pos, node_size=20, node_color=[0.5,0.5,0.7])

# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
plt.axis('off')
# plt.savefig('deg3-spanner-algo1.png')
plt.show()
