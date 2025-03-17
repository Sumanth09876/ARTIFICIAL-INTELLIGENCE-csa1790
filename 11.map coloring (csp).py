def is_safe(node, color, assignment, graph):
    for neighbor in graph[node]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True

def backtracking(graph, colors, assignment, node_list, index):
    if index == len(node_list):
        return assignment

    node = node_list[index]
    for color in colors:
        if is_safe(node, color, assignment, graph):
            assignment[node] = color
            result = backtracking(graph, colors, assignment, node_list, index + 1)
            if result:
                return result
            assignment.pop(node)

    return None

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C']
}
colors = ['Red', 'Green', 'Blue']
result = backtracking(graph, colors, {}, list(graph.keys()), 0)
print("Map Coloring Result:", result)
