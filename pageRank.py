import numpy as np
import scipy.sparse as sp
import tqdm

def normalize(v):
    v = v**2
    return v/np.sum(v)

def page_rank(cluster2gid, g, v2u):
    significance = [0.0]*4532375
    for cluster_id in tqdm.tqdm(cluster2gid):
        v_list = cluster2gid[cluster_id]
        user_list = []
        for v in v_list:
            if v2u[v-1] not in user_list:
                user_list.append(v2u[v-1])
        user_size = len(user_list)
        if user_size <= 1:
            continue
        dict = {}
        for i, user in enumerate(user_list):
            dict[user] = i
        # tmp_graph = np.zeros((n, n),dtype=float32)
        rows, cols, values = [], [], []
        # print("#######", cluster_id, len(v_list), user_size)
        for v in v_list:
            for u in g[v]:
                if u<=v and u in v_list:
                    rows.append(dict[v2u[v-1]])
                    cols.append(dict[v2u[u-1]])
                    values.append(g[v][u])
        tmp_graph = sp.csr_matrix((values, (rows, cols)), shape = (user_size, user_size))
        # print(tmp_graph.shape)
        prob = np.array([[1 / user_size] * user_size])
        old = np.array([[-1] * user_size])
        MAX_ITER = 1000
        cnt = 0
        while np.sum(np.abs(old-prob)) > 0.01 / user_size and cnt < MAX_ITER:
            prob, old = normalize(prob * tmp_graph), prob
            cnt += 1
            # print(cnt, prob.shape)
        
        for i in range(user_size):
            # print(i, user_list[i], prob[0][i])
            significance[user_list[i]] += len(v_list)*prob[0][i]
            # print(user_list[i], significance[user_list[i]])
        # print(significance)
    return significance

import pickle

labels = np.load("/root/workspace/GraphCluster/dbscan_result/10-bucket/0b_0.8_1_1b_0.8_1_2b_0.5_3_3b_0.5_3_4b_0.5_3_5b_0.6_3_6b_0.5_3_7b_0.5_3_8b_0.05_3_9b_0.005_1.npy")
cluster2gid = {}
for i, x in enumerate(labels):
    if x!=-1:
        if x in cluster2gid:
            cluster2gid[x].append(i+1)
        else:
            cluster2gid[x] = [i+1]
with open("/root/workspace/GraphCluster/adj_list/year_gid_uid_pair_v2e.pkl", "rb") as f:
    g = pickle.load(f)
v2u = np.load("/root/workspace/GraphCluster/adj_list/year_gid_uid_pair_v2u.npy")

significance = page_rank(cluster2gid, g, v2u)

np.save("/root/workspace/GraphCluster/user_feature/significance.npy", significance)
