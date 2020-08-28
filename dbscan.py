from sklearn.cluster import DBSCAN
import scipy.sparse as sp
import numpy as np
import sys


if __name__ == "__main__":
    X = sp.load_npz('/root/workspace/GraphCluster/month_dis.npz')
    eps = float(sys.argv[1])
    min_samples = int(sys.argv[2])
    # Compute DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=10).fit(X)
    labels = clustering.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    np.save('./result/month_result_{}_{}.npy'.format(eps, min_samples), labels)
    np.save('./result/month_core_{}_{}.npy'.format(eps, min_samples), clustering.core_sample_indices_)
