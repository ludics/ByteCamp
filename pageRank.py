import numpy as np
import scipy.sparse as sp

def normalize(v):
    v = v**2
    return v/np.sum(v)

def page_rank(cluster2gid, g, v2u):
    significance = [0.0]*4532375
    for cluster_id in cluster2gid:
        v_list = cluster2gid[cluster_id]
        user_list = []
        for v in v_list:
            if v2u[v-1] not in user_list:
                user_list.append(v2u[v-1])
        user_size = len(user_list)
        if user_size == 1:
            continue
        dict = {}
        for i, user in enumerate(user_list):
            dict[user] = i
        # tmp_graph = np.zeros((n, n),dtype=float32)
        rows, cols, values = [], [], []
        for v in v_list:
            for u in g[v]:
                if u<=v and u in v_list:
                    rows.append(dict[v2u[v-1]])
                    cols.append(dict[v2u[u-1]])
                    values.append(g[v][u])
        tmp_graph = sp.csr_matrix((values, (rows, cols)))
        prob = np.array([[1 / user_size] * user_size])
        old = np.array([[-1] * user_size])
        MAX_ITER = 1000
        cnt = 0
        while np.sum(np.abs(old-prob)) > 0.01 / user_size or cnt < MAX_ITER:
            prob, old = prob * tmp_graph, prob
            cnt += 1
        

        for i in range(user_size):
            significance[user_list[i]] += len(v_list)*prob[i]

    return significance
