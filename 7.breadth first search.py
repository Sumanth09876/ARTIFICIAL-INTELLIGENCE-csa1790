from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    
    print("BFS Traversal Order:")
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(graph[node] - visited)

# Example Graph as an adjacency list
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F', 'G'},
    'D': {'B'},
    'E': {'B', 'H'},
    'F': {'C'},
    'G': {'C'},
    'H': {'E'}
}

# Input: Start BFS from node 'A'
print("Input Graph:", graph)
print("Starting Node: A\n")

# Run BFS
bfs(graph, 'A')
