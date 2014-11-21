import numpy as np


def is_forest(edges, n_vertices=None):
    if n_vertices is not None and len(edges) > n_vertices - 1:
        return False
    n_vertices = np.max(edges) + 1
    parents = -np.ones(n_vertices)
    visited = np.zeros(n_vertices, dtype=np.bool)
    neighbors = [[] for i in range(n_vertices)]
    for edge in edges:
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    lonely = 0
    while lonely < n_vertices:
        for i in range(lonely, n_vertices):
            if not visited[i]:
                queue = [i]
                lonely = i + 1
                visited[i] = True
                break
            lonely = n_vertices

        while queue:
            node = queue.pop()
            for neighbor in neighbors[node]:
                if not visited[neighbor]:
                    parents[neighbor] = node
                    queue.append(neighbor)
                    visited[neighbor] = True
                elif not parents[node] == neighbor:
                    return False
    return True
