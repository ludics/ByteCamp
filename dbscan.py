from sklearn.cluster import DBSCAN
import scipy.sparse as sp
import numpy as np


if __name__ == "__main__":
    X = sp.load_npz('/root/workspace/GraphCluster/month_dis.npz')

    # Compute DBSCAN
    clustering = DBSCAN(eps=0.1, min_samples=10, metric='precomputed', n_jobs=10).fit(X)
    labels = clustering.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    np.save('./month_result_0.1_10.npy', labels)
    np.save('./month_core_0.1_10.npy', clustering.core_sample_indices_)
