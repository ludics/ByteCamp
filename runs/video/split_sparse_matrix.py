import sys
import numpy as np
import scipy.sparse as sp
import time
import pickle
import os


if __name__ == "__main__":
    verbose = False
    if len(sys.argv) > 2:
        verbose = True
    start = time.time()

    labels_ori = np.load(sys.argv[1])
    labels = np.sort(labels_ori)
    count = 0
    cluster2size = {}
    for i in range(labels.shape[0]-1):
        count += 1
        if labels[i+1] > labels[i]:
            cluster2size[labels[i]] = count
            count = 0
    cluster2size[labels[-1]] = count + 1
    size_list = np.sort(np.array(list(cluster2size.values())))
    size_set = set(size_list)
    size2num = {}
    for i in size_set:
        size2num[i] = np.where(size_list==i)[0].shape[0]
    
    if verbose:
        print("Start at {:.3f}".format(time.time() - start))
    start = time.time()

    cluster2gid = {}
    for i in range(labels_ori.shape[0]):
        if labels_ori[i] in cluster2gid:
            cluster2gid[labels_ori[i]].append(i)
        else:
            cluster2gid[labels_ori[i]] = [i]

    level_pos = [1, 5, 100, 10000, 30000000]
    bucket2cluster = {}
    for k in range(len(level_pos)-1):
        bucket2cluster[k] = []

    for cluster_id, cluster_size in cluster2size.items():
        for k in range(len(level_pos)-1):
            if level_pos[k] < cluster_size <= level_pos[k+1]:
                bucket2cluster[k].append(cluster_id)
    bucket2gid = {}
    for k in range(len(level_pos)-1):
        bucket2gid[k] = []
    for k, cluster_id_list in bucket2cluster.items():
        for cluster_id in cluster_id_list:
            bucket2gid[k] += cluster2gid[cluster_id]

    if verbose:
        print("Make bucket2gid, bucket2cluster, bucket2eid at {:.3f}".format(time.time() - start))
    start = time.time()

    X = sp.load_npz('/root/workspace/GraphCluster/sparse_matrix/year_dis.npz')
    X_coo = sp.coo_matrix(X)
    gid2eid = {}
    for i in range(X_coo.row.shape[0]):
        if X_coo.row[i] in gid2eid:
            gid2eid[X_coo.row[i]].append(i)
        else:
            gid2eid[X_coo.row[i]] = [i]

    bucket2eid = {}
    for k in range(len(level_pos)-1):
        bucket2eid[k] = []
    for k, gid_list in bucket2gid.items():
        for gid in gid_list:
            bucket2eid[k] += gid2eid[gid]
    bucket_ori2re = {}
    bucket_re2ori = {}
    for k in range(len(level_pos)-1):
        bucket_ori2re[k] = {}
        bucket_re2ori[k] = {}
    for k, gid_list in bucket2gid.items():
        for i, gid in enumerate(gid_list):
            bucket_ori2re[k][gid] = i
            bucket_re2ori[k][i] = gid

    if verbose:
        print("Load X, make gid2eid, bucket2eid, bucket2gidmap at {:.3f}".format(time.time() - start))
    start = time.time()

    bucket2X = {}
    for k in range(len(level_pos)-1):
        ori_row, ori_col = X_coo.row[bucket2eid[k]], X_coo.col[bucket2eid[k]]
        re_row, re_col = ori_row.copy(), ori_col.copy()
        for i in range(ori_col.shape[0]):
            re_row[i] = bucket_ori2re[k][ori_row[i]]
            re_col[i] = bucket_ori2re[k][ori_col[i]]
        bucket2X[k] = sp.csr_matrix((X_coo.data[bucket2eid[k]], (re_row, re_col)))    
    
    if verbose:
        print("Make bucket2X at {:.3f}".format(time.time() - start))
    start = time.time()

    result_path = '/root/workspace/GraphCluster/sparse_matrix/4-bucket/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for k, X in bucket2X.items():
        sp.save_npz(result_path + 'year_bucket{}_dis.npz'.format(k), X)

    with open(result_path + 'bucket_ori2re.pkl', 'wb') as f:
        pickle.dump(bucket_ori2re, f, protocol = pickle.HIGHEST_PROTOCOL)
    with open(result_path + 'bucket_re2ori.pkl', 'wb') as f:
        pickle.dump(bucket_re2ori, f, protocol = pickle.HIGHEST_PROTOCOL)

    if verbose:
        print("Save & all done at {:.3f}".format(time.time() - start))
    start = time.time()

