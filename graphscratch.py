#
# import networkx as nx
# import matplotlib.pyplot as plt
# 
# # Create a graph
# G = nx.Graph()
# 
# # Add nodes with attributes
# G.add_node(1, label='A', color='red', weight=1, text='wololo', val=2)
# G.add_node(2, label='B', color='blue', weight=2 , text='kwiwi', val=3)
# G.add_node(3, label='C', color='green', weight=0.5, text='lululu', val=4)
# 
# # Add edges
# G.add_edge(1, 2)
# G.add_edge(1, 3)
# G.add_edge(2, 3)
# 
# # Access node attributes
# #print("Node 1 attributes:", G.nodes[1])
# #print("Label of node 2:", G.nodes[2]['label'])
# 
# # Draw the graph with node attributes
# node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
# node_colors = [G.nodes[node]['color'] for node in G.nodes()]
# node_weights = [G.nodes[node]['weight'] * 1000 for node in G.nodes()]  # Scale weights for better visibility
# 
# nx.draw(G, with_labels=True, labels=nod-e_labels, node_color=node_colors, node_size=node_weights, edge_color='black')
# 
# # Display the graph
# plt.title("Graph with Node Attributes")
# plt.show()
# 
# 

import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add nodes
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Add edges with weights
G.add_edge(1, 2, weight=1)
G.add_edge(1, 5, weight=1)
G.add_edge(2, 3, weight=1)
G.add_edge(3, 4, weight=1)
G.add_edge(4, 5, weight=1)
G.add_edge(4, 6, weight=1)
G.add_edge(5, 6, weight=1)
G.add_edge(6, 7, weight=1)

G.remove_edge(1, 2)
# find shortest part 


# Find the shortest path between nodes 1 and 5
shortest_path = nx.shortest_path(G, source=1, target=7, weight='weight')
shortest_path_length = nx.shortest_path_length(G, source=1, target=7, weight='weight')

print("Shortest path:", shortest_path)
print("Shortest path length:", shortest_path_length)

# Draw the graph
pos = nx.spring_layout(G)  # Positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='blue')

# Highlight the shortest path
path_edges = list(zip(shortest_path, shortest_path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, alpha=0.5, edge_color='red')

# Display the graph
plt.title("Shortest Path in the Graph")
plt.show()

# this ends us being a scratchpad 

G = nx.Graph()
G.add_edge(1,2,color='r',weight=2)
G.add_edge(2,3,color='b',weight=4)
G.add_edge(3,4,color='g',weight=6)

pos = nx.circular_layout(G)

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
weights = [G[u][v]['weight'] for u,v in edges]

nx.draw(G, pos, edges=edges, edge_color=colors, width=weights)


# '#%02x%02x%02x' % (B,G,R)


G = nx.Graph()
G.add_edge(0,1,color='#FF0000',weight=10)
G.add_edge(1,2,color='#00FF00',weight=20)
G.add_edge(2,0,color='#0000FF',weight=30)

colors = nx.get_edge_attributes(G,'color').values()
weights = nx.get_edge_attributes(G,'weight').values()

pos = nx.circular_layout(G)
nx.draw(G, pos, 
        edge_color=colors, 
        width=list(weights),
        with_labels=True,
        node_color='orange')


