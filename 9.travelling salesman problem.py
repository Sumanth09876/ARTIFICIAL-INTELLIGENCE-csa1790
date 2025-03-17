from itertools import permutations

def tsp(graph, start):
    vertices = list(graph.keys())
    vertices.remove(start)
    min_path = float('inf')

    for perm in permutations(vertices):
        current_weight = 0
        k = start

        for j in perm:
            current_weight += graph[k][j]
            k = j
        
        current_weight += graph[k][start]
        min_path = min(min_path, current_weight)

    return min_path

graph = {
    'A': {'A': 0, 'B': 10, 'C': 15, 'D': 20},
    'B': {'A': 10, 'B': 0, 'C': 35, 'D': 25},
    'C': {'A': 15, 'B': 35, 'C': 0, 'D': 30},
    'D': {'A': 20, 'B': 25, 'C': 30, 'D': 0}
}

print("Minimum cost of Travelling Salesman Path:", tsp(graph, 'A'))
