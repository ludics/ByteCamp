import tqdm
import gc
import scipy.sparse as sp
from user_cluster import *
import numpy as np


def make_user_similarity_graph(cluster_by_user):
    '''
        there are cluster_by_user[i][j] works in cluster[i] made by user[j]
    '''
    rows, cols, vals = [], [], []
    for cluster_id in tqdm.tqdm(cluster_by_user):
        amp = sum(v for v in cluster_by_user[cluster_id].values())
        for i in cluster_by_user[cluster_id]:
            for j in cluster_by_user[cluster_id]:
                if i!=j:
                    rows.append(i)
                    cols.append(j)
                    vals.append(cluster_by_user[cluster_id][i]*cluster_by_user[cluster_id][j]/amp)
        if (cluster_id + 1) % 2000000 == 0:
            X = sp.csr_matrix((vals, (rows, cols)))
            sp.save_npz('/root/workspace/GraphCluster/user_feature/user_similiar/{}.npy'.format(cluster_id), X)
            del X
            del rows
            del cols
            del vals
            gc.collect()
            rows, cols, vals = [], [], []
            print("Save & gc, cluster id: {}".format(cluster_id))
    X = sp.csr_matrix((vals, (rows, cols)))
    sp.save_npz('/root/workspace/GraphCluster/user_feature/user_similiar/{}.npy'.format(cluster_id), X)
    print("Save & gc, cluster id: {}".format(cluster_id))
    # return u_graph

import math
def compute_user_significance(v2u, cluster_to_gid):
    '''
        v2u[gid] = uid, cluster_to_gid[i] contains all the gids in cluster[i]
    '''
    significance = [0.0]*4532375
    for i in cluster_to_gid:
        tmp = sorted(cluster_to_gid[i], reverse = True)
        l = len(tmp)
        for j in range(l):
            significance[v2u[tmp[j]]] += math.tan(-math.pi/4+math.pi/2*j/l)*l
    return significance


if __name__ == "__main__":
    v2u = np.load('/root/workspace/GraphCluster/adj_list/year_gid_uid_pair_v2u.npy')
    labels = np.load('/root/workspace/GraphCluster/dbscan_result/10-bucket/0b_0.8_1_1b_0.8_1_2b_0.5_3_3b_0.5_3_4b_0.5_3_5b_0.6_3_6b_0.5_3_7b_0.5_3_8b_0.05_3_9b_0.005_1.npy')
    user_in_cluster, cluster_by_user =  user_cluster_nums(labels, v2u)
    del user_in_cluster
    make_user_similarity_graph(cluster_by_user)

