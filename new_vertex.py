import numpy as np
import scipy.sparse as sp

def change_g(gid, edge_list, g):
    g[gid] = {}
    for u in edge_list:
        g[gid][u] = edge_list[u]
        g[u][gid] = edge_list[u]
    return g

#When a new vertex with gid and edge_list comes, let g = change_g(gid, edge_list, g)

def subgraph_to_solve_after_new_vertex(gid, edge_list, c, clusters, g):
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
        else: v_list = v_list + clusters[c[u]]
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



#DBSCAN the subgraph and maintain the global c and clusters.
#with v_list[i] we can find the origin gid for alias i in the whole graph.
