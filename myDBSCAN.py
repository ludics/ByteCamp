def my_dbscan(v_list, e_list, emp, min_sample):
    clusters = {}
    cluster = {}
    visited = {}
    degree = {}
    n = 0
    for v in v_list:
        cluster[v] = -1
        visited[v] = False
        d = 0
        for u in e_list[v]:
            if e_list[v][u]>=emp:
                d += 1
        degree[v] = d
        
    for v in v_list:
        if not visited[v] and degree[v]>=min_sample:
            q = [v]
            k = 0
            visited[v] = True
            while k<len(q):
                x = q[k]
                if degree[x]>=min_sample:
                    for u in e_list[x]:
                        if e_list[x][u]>=emp:
                            q.append(u)
                            visited[u] = True
                k += 1
            clusters[n] = q
            for i in q:
                cluster[i] = n
            n += 1
    return cluster, clusters
