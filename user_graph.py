import pickle
def make_user_graph(labels, cluster2gid):
    with open("/root/workspace/GraphCluster/adj_list/year_gid_uid_pair_u2v.pkl", "rb") as f:
        u2v = pickle.load(f)
    usr = [-1]*60034550
    for u in u2v:
        for v in u2v[u]:
            usr[v-1] = u
    u_graph = {}
    for u in u2v:
        l = len(u2v[u])
        u_graph[u] = {}
        for v in u2v[u]:
            if labels[v-1]!=-1:
                for x in cluster2gid[labels[v-1]]:
                    if usr[x-1] in u_graph[u]:
                        u_graph[u][usr[x-1]] += 1/l
                    else:
                        u_graph[u][usr[x-1]] = 1/l
    return u_graph
