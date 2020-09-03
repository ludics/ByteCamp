import scipy.sparse as sp
from multiprocessing import Pool



def make_user_similarity_graph(cluster_by_user):
    '''
        there are cluster_by_user[i][j] works in cluster[i] made by user[j]
    '''
    row, col, value = [], [], []
    for i in cluster_by_user:
        amp = sum(v for v in cluster_by_user[i].values())
        s = sum(v*v for v in cluster_by_user[i].values())
        tmp = {}
        for j in cluster_by_user[i]:
            if cluster_by_user[i][j]*cluster_by_user[i][j]*amp/s>0.01:
                tmp[j] = cluster_by_user[i][j]/s
        for i in tmp:
            for j in tmp:
                if i!=j:
                    if i not in u_graph:
                        u_graph[i] = {j:tmp[i]*tmp[j]*amp}
                    elif j in u_graph[i]:
                        u_graph[i][j] += tmp[i]*tmp[j]*amp
                    else:
                        u_graph[i][j] = tmp[i]*tmp[j]*amp
    return u_graph


if __name__ == "__main__":
    pool = Pool(processes=10):






