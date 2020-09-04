from sklearn.cluster import DBSCAN
import scipy.sparse as sp
import numpy as np
import sys
import tqdm
import pickle
import os

def plagiarism(c, g, k=1.0):
	"""
	计算原创性分数
	Args:
	g: true graph
	c: type(numpy.array) shape(numvetexs) the same cluters's vetexs shoud have the same label
	k: decay coefficient
	Returns:
	plag: How many works I plagiarized.
	orig: Score of works I created.
	"""
	# clusters = {}
	length = len(g)
    # .shape[0]
	orig = [0.0]*length
	plag = [0.0]*length
	print("[DEBUG] begin calculating ...")
	for stealer in tqdm.tqdm(range(length-2, -1, -1)):
		if c[stealer]!=-1:
			for creater in g[stealer]:
				if creater<stealer and c[stealer]==c[creater]:
					plag[stealer] += g[stealer][creater]
					orig[creater] += g[stealer][creater]*(k+orig[stealer])
	return plag, orig

if __name__ == '__main__':
	graph_file = os.path.join("/home/junyu/graph_data", "year_gid_uid_pair_v2e.pkl")
	column_file = sys.argv[1]
    # "/root/workspace/ByteCamp/result/year_result_0.01_10.npy"
	with open(column_file, "rb") as f:
		column = np.load(f)
	with open(graph_file, "rb") as f:
		graph = pickle.load(f)

	res_plag, res_orig = plagiarism(column, graph)

	np.save('./result/orig_score.npy', res_orig)
	np.save('./result/plag_score.npy', res_plag)