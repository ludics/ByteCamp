import time
import numpy as np
def modularity(adjacency, clusters):
    """Computes graph modularity.
    Args:
        adjacency: sparse adjacency matrix, type: scipy.sparse.csr_matrix
        clusters: type(numpy.array) shape(numvetexs) the same cluters's vetexs shoud have the same label
    Returns:
        The value of graph modularity.
    """
    start = time.time()
    vetexnums = adjacency.shape[0]
    adjacency[range(vetexnums), range(vetexnums)] = 0
    nozero = adjacency.nonzero()
    adjacency[nozero[0], nozero[1]] = 1
    degrees = adjacency.sum(axis=0).A1
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
    print(f'cal modularity time is {time.time() - start}s')
    return result / n_edges


def conductance(adjacency, clusters):
    """Computes graph conductance 
    Args:
    adjacency: sparse adjacency matrix, type: scipy.sparse.csr_matrix
    clusters: type(numpy.array) shape(numvetexs) the same cluters's vetexs shoud have the same label
    Returns:
    The average conductance value of the graph clusters.
    """
    start = time.time()
    inter = 0  # Number of inter-cluster edges.
    intra = 0  # Number of intra-cluster edges.
    vetexnums = adjacency.shape[0]
    adjacency[range(vetexnums), range(vetexnums)] = 0
    cluster_indices = np.zeros(adjacency.shape[0], dtype=np.bool)
    for cluster_id in np.unique(clusters):
        cluster_indices[:] = 0
        cluster_indices[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_indices, :]
        inter += np.sum(adj_submatrix[:, cluster_indices])
        intra += np.sum(adj_submatrix[:, ~cluster_indices])
    print(f'cal conductance is {time() - start}s')
    return intra / (inter + intra)

def fscore(adjacency, clusters, gt):
    """Computes graph conductance 
    Args:
        adjacency: sparse adjacency matrix, type: scipy.sparse.csr_matrix
        clusters: type(numpy.array) shape(numvetexs) the same cluters's vetexs shoud have the same label
    Returns:
        The f1 score of the graph clusters.
    """