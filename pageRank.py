import numpy as np

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
        n = len(user_list)
        dict = {}
        for i, user in enumerate(user_list):
            dict[user] = i
        tmp_graph = np.zeros((n, n),dtype=float32)
        for v in v_list:
            for u in g[v]:
                if u<=v and u in v_list:
                    tmp_graph[dict[v2u[v-1]]][dict[v2u[u-1]]] += g[v][u]
        prob = np.array([[1/l]*l])
        old = np.array([[-1]*l])
        while np.sum(np.abs(old-prob))>0.01/l:
            prob, old = normalize(prob.dot(tmp_graph)), prob
        
            
        for i in range(n):
            significance[user_list[i]] += len(v_list)*prob[i]

    return significance
