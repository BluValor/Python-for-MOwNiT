import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# G = nx.read_edgelist('../graphs/facebook_combined.txt')
nodes_nr = random.randint(15, 30)
edges_nr = random.randint(50, 80)
min_weight = 1
max_weight = 10
G = nx.gnm_random_graph(nodes_nr, edges_nr)

s = random.randint(0, nodes_nr - 2)
t = random.randint(1, nodes_nr - 1)
E = random.randrange(10, 100)

G.add_edge(s, t)

for (n, m) in G.edges():
    G.edges[n, m]['weight'] = random.randint(min_weight, max_weight)

print(edges_nr)
print(nx.info(G))
print(nx.is_weighted(G))

cmap = plt.cm.get_cmap('coolwarm') # coolwarm
rgbw = cmap(np.linspace(0, 1, max_weight - min_weight + 1))
# print(rgbw)

pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'weight')

nx.draw_networkx_nodes(G, pos, node_size=5, node_color='b')

for (n, m) in G.edges():
    G.edges[n, m]['color'] = rgbw[G[n][m]['weight'] - 1]

colors = [G[n][m]['color'] for n, m in G.edges()]

nx.draw_networkx_edges(G, pos, edgelist=G.edges(data=True), edge_color=colors)

# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
