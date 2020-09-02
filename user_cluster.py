def user_cluster_nums(labels, v2u):
    '''
        user_in_cluster[i][j] = k represents:
        user i has k works in cluster j
        cluster_by_user[i][j] = k represents:
        cluster i has k works made by user j
    '''
    user_in_cluster, cluster_by_user = {}, {}
    for v in range(60034550):
        if labels[v]!=-1:
            if v2u[v] not in user_in_cluster:
                user_in_cluster[v2u[v]] = {}
                user_in_cluster[v2u[v]][labels[v]] = 1
            elif labels[v] in user_in_cluster[v2u[v]]:
                user_in_cluster[v2u[v]][labels[v]] += 1
            else:
                user_in_cluster[v2u[v]][labels[v]] = 1
            if labels[v] not in cluster_by_user:
                cluster_by_user[labels[v]] = {}
                cluster_by_user[labels[v]][v2u[v]] = 1
            elif v2u[v] in cluster_by_user[labels[v]]:
                cluster_by_user[labels[v]][v2u[v]] += 1
            else:
                cluster_by_user[labels[v]][v2u[v]] = 1
    return user_in_cluster, cluster_by_user
