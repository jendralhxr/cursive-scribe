from collections import deque
import networkx as nx

def custom_bfs_dfs1(graph, start_node):
    queue = deque([start_node])  # Initialize the queue with the start node
    visited = set([start_node])  # Set to keep track of visited nodes
    edges = []  # List to store edges

    def dfs(node):
        stack = [node]  # Use a stack for the DFS traversal
        branch_edges = []
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                # Explore all neighbors of the current node
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)
                        branch_edges.append((current, neighbor))  # Add the edge to the branch
        return branch_edges

    while queue:
        node = queue.popleft()  # Get the next node from the BFS queue
        neighbors = list(graph.neighbors(node))  # Get neighbors of the current node
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
        
        # print all the branch first then traverse
        if unvisited_neighbors:
            for neighbor in unvisited_neighbors:
                # Add the edge from the current node to the neighbor
                edges.append((node, neighbor))
                if len(list(graph.neighbors(neighbor))) > 1:  # If there's a branch
                    branch_edges = dfs(neighbor)  # Explore the branch using DFS
                    edges.extend(branch_edges)  # Add the edges found during DFS
                else:
                    visited.add(neighbor)  # Mark the neighbor as visited
                    queue.append(neighbor)  # Add the neighbor to the BFS queue

    return edges



def custom_bfs_dfs2(graph, start_node):
    queue = deque([start_node])
    visited = set([start_node])
    edges = []

    def dfs(node):
        stack = [node]
        branch_edges = []
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)
                        branch_edges.append((current, neighbor))
        return branch_edges

    while queue:
        node = queue.popleft()
        neighbors = list(graph.neighbors(node))
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]

        # traverse one branch first then backtrack
        if len(unvisited_neighbors) > 1:
            for neighbor in unvisited_neighbors:
                edges.append((node, neighbor))
                branch_edges = dfs(neighbor)
                edges.extend(branch_edges)
        else:
            for neighbor in unvisited_neighbors:
                visited.add(neighbor)
                queue.append(neighbor)
                edges.append((node, neighbor))

    return edges



# Example usage
edges = [
    (0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 24), (24, 42), (42, 61),
    (61, 41), (61, 62), (61, 83), (41, 23), (41, 60), (62, 63), (83, 118),
    (23, 40), (60, 59), (63, 64), (118, 116), (64, 44), (116, 117),
    (44, 43), (117, 115), (43, 84), (115, 114), (84, 85), (114, 58),
    (85, 86), (85, 87)
]

graph = nx.Graph()
graph.add_edges_from(edges)

start_node = 0
print(custom_bfs_dfs1(graph, start_node))
print(custom_bfs_dfs2(graph, start_node))

