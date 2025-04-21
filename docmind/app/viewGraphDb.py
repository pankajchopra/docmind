import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml("knowledge_graph.graphml")

# Basic visualization
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=1500, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'relationship'))
plt.savefig("graph_visualization.png", format="PNG")
plt.show()
