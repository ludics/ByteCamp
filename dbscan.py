from sklearn.cluster import DBSCAN
import scipy.sparse as sp
import numpy as np
import sys
from multiprocessing import Pool
import time


paramenter_list = [
    # (0.8, 1),
    # (0.2, 1),
    # (0.1, 1),
    # (0.05, 1),
    # (0.01, 1),
    # (0.2, 2),
    # (0.1, 2),
    # (0.05, 2),
    # (0.01, 2),
    (0.8, 5),
    # (0.2, 5),
    # (0.1, 5),
    # (0.05, 5),
    # (0.01, 5),
    # (0.8, 10),
    # (0.2, 10),
    (0.1, 10),
    (0.05, 10)
    # (0.01, 10)
]

X = sp.load_npz('/root/workspace/GraphCluster/month_dis.npz')


def make_dbscan(num):
    global X
    global paramenter_list
    start = time.time()
    eps, min_samples = paramenter_list[num]
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=3).fit(X)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # print('{}Estimated number of noise points: %d'.format(n_noise_))
    np.save('./result/month_result_{}_{}.npy'.format(eps, min_samples), labels)
    np.save('./result/month_core_{}_{}.npy'.format(eps, min_samples), clustering.core_sample_indices_)   
    print('eps={}, min_samples={}, Estimated number of clusters: {}'.format(eps, min_samples, n_clusters_))
    print('eps={}, min_samples={}, Estimated number of noise points: {}'.format(eps, min_samples, n_noise_))
    print('eps={}, min_samples={}, time cost: {:.3f}'.format(eps, min_samples, time.time() - start))


if __name__ == "__main__":
    # eps = float(sys.argv[1])
    # min_samples = int(sys.argv[2])
    # Compute DBSCAN
    # Number of clusters in labels, ignoring noise if present.
    pool = Pool(processes=len(paramenter_list))
    for i in range(len(paramenter_list)):
        pool.apply_async(make_dbscan, (i,))
    pool.close()
    pool.join()
