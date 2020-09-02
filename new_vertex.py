import numpy as np
import scipy.sparse as sp

def change_g(gid, edge_list, g):
    g[gid] = {}
    for u in edge_list:
        g[gid][u] = edge_list[u]
        g[u][gid] = edge_list[u]

#When a new vertex with gid and edge_list comes, change_g(gid, edge_list, g)

def subgraph_to_solve(gid, edge_list, labels, cluster2gids, g):
    '''
        gid is the new vertex's gid
        edge_list saves all the edges connect this new vertex with the origin graph
        labels[i] claims which cluster video i is in
        cluster2gids[i] saves the list of video gids in the i-th cluster
        g is the graph

        the second returned value is an alias sparse matrix saving the subgraph to solve
        the first returned v_list saves the correspondence of the alias
    '''
    v_list = [gid]
    labels.append(-1)
    for u in edge_list:
        if labels[u]==-1: v_list.append(u)
        elif labels[u] in cluster2gids:
            v_list = v_list + cluster2gids[labels[u]]
            del cluster2gids[labels[u]]
    dict = {}
    for i, v in enumerate(v_list):
        dict[v] = i
    row, col, value = [], [], []
    for v in v_list:
        for u in g[v]:
            if u in v_list:
                row.append(dict[v])
                col.append(dict[u])
                value.append(g[v][u])
    
    return v_list, sp.csr_matrix((value, (row, col))) if value else None


def new_vertex(gid, edge_list, g, labels, cluster2gids):
    ''' a new vertex with gid and edge_list comes '''
    change_g(gid, edge_list, g)
    v_list, X = subgraph_to_solve(gid, edge_list, labels, cluster2gids, g)
    if X:
#DBSCAN the subgraph and maintain the global c and clusters.
        size = len(v_list)
        clustering = DBSCAN(eps=eps(size), min_samples=min_samples(size), metric='precomputed', n_jobs=10).fit(X)
        local_labels = clustering.labels_
        m = max(labels)+1
        
        for i in range(size):
            labels[v_list[i]] = local_labels[i]+m
            if (local_labels[i]+m) in cluster2gids:
                cluster2gids[local_labels[i]+m].append(v_list[i])
            else:
                cluster2gids[local_labels[i]+m] = [v_list[i]]

