import numpy as np
import scipy.sparse as sp

def change_g(gid, edge_list, g):
    g[gid] = {}
    for u in edge_list:
        g[gid][u] = edge_list[u]
        g[u][gid] = edge_list[u]
    return g

def subgraph_to_solve_after_new_vertex(gid, edge_list, c, clusters, g):
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
