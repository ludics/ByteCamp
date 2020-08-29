import pandas as pd
import scipy.sparse as sp
import numpy as np
import time
from multiprocessing import Pool

start = time.time()
year_data = pd.read_csv('/data00/tiger/graph_data/year_gid_uid_pair.csv', header=None)
year_data.columns = ['gid1', 'uid1', 'gid2', 'uid2', 'weight']
print("Read csv {}s".format(time.time() - start))


def make_dict(num):
    global year_data
    dis_dict = {}
    split_num = int(np.ceil(year_data.shape[0]) / 46. )
    start = split_num * num
    end = min( split_num * (num + 1), year_data.shape[0])
    start = time.time()
    for i in range(start, end):
        dis_dict[(year_data['gid1'][i] - 1, year_data['gid2'][i] - 1)] = year_data['weight'][i]
        dis_dict[(year_data['gid2'][i] - 1, year_data['gid1'][i] - 1)] = year_data['weight'][i]
        if i % 1000000 == 0:
            print("{} done {} in {}s".format(num, i, time.time() - start))
    print("{} all done in {}s".format(num, time.time() - start))
    return dis_dict


if __name__ == "__main__":
    dis_dict = {}
    start = time.time()
    result = []
    pool = Pool(processes=46)
    for i in range(46):
        result.append(pool.apply_async(make_dict, (i, )))
    pool.close()
    pool.join()
    for res in result:
        dis_dict.update(res.get())
    # del year_data
    start = time.time()
    for i in range(60034550):
        dis_dict[(i, i)] = 1.0
        if i % 10000000 == 0:
            print("Add diag done {} in {}s".format(i, time.time() - start))
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
