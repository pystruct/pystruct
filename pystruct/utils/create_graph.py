import numpy as np

def create_graph(grid):
    """
    This function creates a graph of vertices and edges from segments returned by SLIC.
    :param grid: A grid of segments as returned by the slic function defined in skimage library
    :return: A graph as [vertices, edges] 
    """
    # get an array of unique labels
    vertices = np.unique(grid)
    
    # get number of vertices
    num_vertices = len(vertices)

    # map these unique labels to [1,...,N], where N is the number of labels (vertices)
    mapping = dict(zip(vertices, np.arange(num_vertices)))
    mapped_grid = np.array([mapping[x] for x in grid.flat]).reshape(grid.shape)

    # create edges, going left to right and top to bottom
    l2r = np.c_[mapped_grid[:, :-1].ravel(), mapped_grid[:, 1:].ravel()]
    t2b = np.c_[mapped_grid[:-1, :].ravel(), mapped_grid[1:, :].ravel()]

    # stack for entire graph 
    edges = np.vstack([l2r, t2b])
    edges = edges[edges[:, 0] != edges[:, 1], :]
    edges = np.sort(edges, axis=1)
    
    # create a edge map, a hashmap
    edge_map = edges[:, 0] + num_vertices * edges[:, 1]
    
    # filter unique connections as edges
    edges = np.unique(edge_map)
    
    # reverse map and form edges as pairs
    edges = [[vertices[edge % num_vertices],
              vertices[edge / num_vertices]] for edge in edges]

    return vertices, edges
