import numpy as np
import scipy.sparse as sp

def change_g(gid, edge_list, g):
    g[gid] = {}
    for u in edge_list:
        g[gid][u] = edge_list[u]
        g[u][gid] = edge_list[u]

#When a new vertex with gid and edge_list comes, change_g(gid, edge_list, g)

def subgraph_to_solve(gid, edge_list, c, clusters, g):
    '''
        gid is the new vertex's gid
        edge_list saves all the edges connect this new vertex with the origin graph
        c[i] claims which cluster video i is in
        clusters[i] saves the list of video gids in the i-th cluster
        g is the graph

        the second returned value is an alias sparse matrix saving the subgraph to solve
        the first returned v_list saves the correspondence of the alias
    '''
    v_list = [gid]
    for u in edge_list:
        if c[u]==-1: v_list.append(u)
        else:
            v_list = v_list + clusters[c[u]]
            del clusters[c[u]]
    dict = {}
    for i, v in enumerate(v_list):
        dict[v] = i
    row, col, value = [], [], []
    for v in v_list:
        for u in g[v]:
            if u in v_list:
                row.append(dict[v])
                col.append(dict[u])
                value.append(e_list[v][u])
    
    return v_list, sp.csr_matrix((value, (row, col)))


def new_vertex(gid, edge_list, g, c, clusters):
    ''' a new vertex with gid and edge_list comes '''
    change_g(gid, edge_list, g)
    v_list, X = subgraph_to_solve(gid, edge_list, c, clusters, g)
    
#DBSCAN the subgraph and maintain the global c and clusters.
    size = len(v_list)
    clustering = DBSCAN(eps=eps(size), min_samples=min_samples(size), metric='precompute    d', n_jobs=10).fit(X)
    labels = clustering.labels_
    m = max(c)+1
#with v_list[i] we can find the origin gid for alias i in the whole graph.
    for i in range(size):
        c[v_list[i]] = labels[i]+m
        if (labels[i]+m) in clusters:
            clusters[labels[i]+m].append(v_list[i])
        else:
            clusters[labels[i]+m] = [v_list[i]]

