import pandas as pd
import scipy.sparse as sp
import numpy as np
import time


if __name__ == "__main__":
    start = time.time()
    year_data = pd.read_csv('/data00/tiger/graph_data/year_gid_uid_pair.csv', header=None)
    print("Read csv {}s".format(time.time() - start))
    year_data.columns = ['gid1', 'uid1', 'gid2', 'uid2', 'weight']
    dis_dict = {}
    start = time.time()
    for i in range(year_data.shape[0]):
        dis_dict[(year_data['gid1'][i] - 1, year_data['gid2'][i] - 1)] = year_data['weight'][i]
        dis_dict[(year_data['gid2'][i] - 1, year_data['gid1'][i] - 1)] = year_data['weight'][i]
        if i % 10000000 == 0:
            print("{} done {}s".format(i, time.time() - start))
    start = time.time()
    for i in range(60034550):
        dis_dict[(i, i)] = 1.0
        if i % 10000000 == 0:
            print("{} 2 done {}s".format(i, time.time() - start))
    del year_data
    print("make dict end")
    start = time.time()
    values = list(dis_dict.values())
    values = np.array(values, dtype=np.float32)
    keys = list(dis_dict.keys())
    rows, cols = zip(*keys)
    print("preprocess end for {}s".format(time.time() - start))
    start = time.time()
    X = sp.csr_matrix((values, (rows, cols)))
    sp.save_npz('./year_sim.npz', X)
    dis = 1 - values
    Y = sp.csr_matrix((dis, (rows, cols)))
    sp.save_npz('./year_dis.npz', Y)
    print("make sparse matrix and save done for {}s".format(time.time() - start))
