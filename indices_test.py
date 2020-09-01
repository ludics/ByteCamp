import scipy.sparse as sp
import numpy as np
import modularity_conductance as indices
import read_ground_truth from gt
import pickle

gt_v, gt_e = read_ground_truth()
with open("/home/junyu/graph_data/year_gid_uid_pair_u2v.pkl", "rb") as f:
    g = pickle.load(f)

c = np.load("/root/workspace/ByteCamp/result/year_result_0.8_5.npy")

m = sp.load_npz("/root/workspace/GraphClustre/year_sim.npz")
m = sp.coo_matrix(m)
    
def test(g, m, c, t):
    return indices.modularity(g, c, t), indices.conductance(m, c, t), indices.f1_score(c, gt_v, gt_e)
