import tqdm

def make_user_similarity_graph(cluster_by_user):
    '''
        there are cluster_by_user[i][j] works in cluster[i] made by user[j]
    '''
    u_graph = {}
    for cluster_id in tqdm.tqdm(cluster_by_user):
        amp = sum(v for v in cluster_by_user[i].values())
        s = sum(v*v for v in cluster_by_user[i].values())
        for i in cluster_by_user[cluster_id]:
            for j in cluster_by_user[cluster_id]:
                if i!=j:
                    if i not in u_graph:
                        u_graph[i] = {j:tmp[i]*tmp[j]/amp}
                    elif j in u_graph[i]:
                        u_graph[i][j] += tmp[i]*tmp[j]/amp
                    else:
                        u_graph[i][j] = tmp[i]*tmp[j]/amp
    return u_graph

import math
def compute_user_significance(v2u, cluster_to_gid):
    '''
        v2u[gid-1] = uid, cluster_to_gid[i] contains all the gids in cluster[i]
        this function returns the relative significance of each user, computed by the rank of each work
    '''
    significance = [0.0]*4532375
    for i in cluster_to_gid:
        tmp = sorted(cluster_to_gid[i], reverse = True)
        l = len(tmp)
        for j in range(l):
            significance[v2u[tmp[j]]] += math.tan(-math.pi/4+math.pi/2*j/l)*l
    return significance
