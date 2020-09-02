import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.cluster import DBSCAN
import scipy.sparse as sp
import numpy as np
import sys
import os
from multiprocessing import Pool
import time
# /root/workspace/GraphCluster/sparse_matrix/10-bucket
num_list = [i for i in range(10)]
file_names = [ "/root/workspace/GraphCluster/sparse_matrix/10-bucket/year_bucket{}_dis.npz".format(i) for i in num_list]

paramenter_list = [
    1: {
        (0.8, 3),
        (0.5, 3),
        (0.3, 3),
        (0.1, 3),
        (0.5, 5),
        (0.3, 5),
    },
    2: {
        (0.5, 5),
        (0.3, 5), 
        (0.6, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
    },
    3: {
        (0.5, 5),
        (0.3, 5), 
        (0.6, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
    },
    4: {
        (0.5, 5),
        (0.3, 5), 
        (0.6, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
    },
    5: {
        (0.5, 5),
        (0.3, 5), 
        (0.6, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
        (0.1, 3),
        (0.05,3),
    },
    6: {
        (0.5, 5),
        (0.3, 5), 
        (0.6, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
        (0.1, 3),
        (0.05,3),
    },
    7: {
        (0.5, 5),
        (0.3, 5), 
        (0.08, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
        (0.1, 3),
        (0.05,3),
    },
    8: {
        (0.5, 5),
        (0.3, 5), 
        (0.08, 5),
        (0.15, 5),
        (0.25, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
        (0.1, 3),
        (0.15, 3),
        (0.25, 3),
        (0.05,3),
    },
    9: {
        (0.5, 5),
        (0.3, 5), 
        (0.08, 5),
        (0.15, 5),
        (0.25, 5),
        (0.5, 3),
        (0.3, 3), 
        (0.6, 3),
        (0.2, 3),
        (0.1, 3),
        (0.15, 3),
        (0.25, 3),
        (0.05,3),
    },
    # (0.8, 1),
    # (0.8, 5),
    # (0.8, 10),
    # (0.2, 1),
    # (0.2, 5),
    # (0.2, 10),
    # (0.2, 25),
    # (0.1, 1),
    # (0.1, 5),
    # (0.1, 10),
    # (0.1, 25),
    # (0.05, 1),
    # (0.05, 5),
    # (0.05, 10),
    # (0.01, 1),
    # (0.01, 5),
    # (0.01, 10),
    # (0.01, 25),
]

# X = sp.load_npz('/root/workspace/GraphCluster/sparse_matrix/year_dis.npz')

def make_dbscan(file_name, bucket_index, config):
    # global X
    X = sp.load_npz(file_name)
    # global paramenter_list
    start = time.time()
    eps, min_samples = config
    # paramenter_list[num]
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=3).fit(X)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # print('{}Estimated number of noise points: %d'.format(n_noise_))
    save_dir = '/root/workspace/GraphCluster/dbscan_result/10-bucket/'
    np.save(os.path.join(save_dir,'{}'.format(bucket_idx),'year_result_{}bucket_{}_{}.npy'.format(bucket_index, eps, min_samples)), labels)
    np.save(os.path.join(save_dir,'{}'.format(bucket_idx),'year_core_{}bucket_{}_{}.npy'.format(bucket_index, eps, min_samples)), clustering.core_sample_indices_)
    res = pd.DataFrame(columns=('bucket_index', 'config', 'n_clusters', 'n_noise', 'core_sample_indices_', 'time'))
    res = res.append({
        'bucket_index': bucket_index,
        'config': config,
        'n_clusters': n_clusters_,
        'n_noise': n_noise_,
        'core_sample_indices_': len(clustering.core_sample_indices_),
        'time': time.time() - start,
    }, ignore_index=True)
    res.to_csv(os.path.join(save_dir, 'log.csv'), mode='a', encoding='utf-8', header=False, index=False)

if __name__ == "__main__":
    # eps = float(sys.argv[1])
    # min_samples = int(sys.argv[2])
    # Compute DBSCAN
    # Number of clusters in labels, ignoring noise if present.
    pool = Pool(processes=10)
    # make_dbscan(file_names[0], 0, (0.8, 1))

    # pool.apply_async(make_dbscan, (file_names[0], 0, (0.8, 1)))
    for bucket_idx in num_list[1:]:
        for config_ in paramenter_list[bucket_idx]:
            pool.apply_async(make_dbscan, (file_names[bucket_idx], bucket_idx, config_))
    pool.close()
    pool.join()
