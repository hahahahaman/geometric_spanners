import networkx as nx
import matplotlib.pyplot as plt
import math, pickle, random

filename = 'graphs3.data'
graphs = []
with open(filename, "rb") as f:
    graphs = pickle.load(f)

max_stretch = 0
min_stretch = 1000000000
min_index = 0
max_index = 0
for i in range(len(graphs)):
    g = graphs[i]
    sf = g.graph['stretch_factor'] 
    if(sf > max_stretch):
        max_stretch = sf
        max_index = i
    if(sf < min_stretch):
        min_stretch = sf
        min_index = i

Gmax = graphs[max_index]
print(max_index, ":", max_stretch)
pos = nx.get_node_attributes(Gmax, 'pos')

plt.figure(figsize=(8,8))
nx.draw_networkx_edges(Gmax, pos, alpha=0.4)
nx.draw_networkx_nodes(Gmax, pos, node_size=20, node_color=[0.5,0.5,0.7])

plt.axis('off')
# plt.savefig('deg3-spanner-algo1.png')
plt.show()

Gmin = graphs[min_index]
print(min_index, ":", min_stretch)
pos = nx.get_node_attributes(Gmin, 'pos')

plt.figure(figsize=(8,8))
nx.draw_networkx_edges(Gmin, pos, alpha=0.4)
nx.draw_networkx_nodes(Gmin, pos, node_size=20, node_color=[0.5,0.5,0.7])

plt.axis('off')
# plt.savefig('deg3-spanner-algo1.png')
plt.show()
