import csv
import collections
import sys
import json
import pickle
import pandas 
import tqdm

if __name__ == '__main__':
	datafile = sys.argv[1]
	with open(datafile, 'r') as f:
		# datas = csv.reader(f)
		datas = pd.read_csv('/data00/tiger/graph_data/year_gid_uid_pair.csv', header=None,
			dtype={0: "int32", 1: "int32", 2: "int32", 3: "int32", 4: "float32"})
    	datas.columns = ['gid1', 'uid1', 'gid2', 'uid2', 'weight']
		edge_nums = 0
		v2e_map = collections.defaultdict(dict)
		u2v_map = collections.defaultdict(set)
		print("[DEBUG] DATA LOADED!")
		for gid1, uid1, gid2, uid2, w in tqdm.tqdm(datas):
			print(gid, uid1, gid2, uid2, w)
			exit(0)
			gid1, uid1, gid2, uid2, w = np.int(gid1), int(uid1), int(gid2), int(uid2), float(w)
			# import ipdb;ipdb.set_trace()
			if min(gid1, gid2) in v2e_map[max(gid1, gid2)]:
				continue
			edge_nums += 1
			u2v_map[uid1].add(gid1)
			u2v_map[uid2].add(gid2)
			gid1, gid2 = (gid1, gid2) if gid1 < gid2 else (gid2, gid1)
			# assert gid1 not in v2e_map[gid2], f'edge {gid1} has been in vertex {gid2}'
			v2e_map[gid2][gid1] = w
			v2e_map[gid1][gid2] = w
			
		
		vetex_nums = sum([len(u2v_map[key]) for key in u2v_map])
		print(f'total vetex num is {vetex_nums}')
		print(f"total owning edge's vetex num is {len(v2e_map)}")
		print(f'total edge num is {edge_nums}')
		print(f'total user num is {len(u2v_map)}')
		# assert edge_nums == sum([len(v2e_map[key]) for key in v2e_map]), 'edge nums not compatible'
		vetex = set()
		for uid in u2v_map:
			vetex |= u2v_map[uid]
			u2v_map[uid] = list(u2v_map[uid])
		assert len(vetex) == vetex_nums, 'diff user has same video'
		v2e_map['vetexs'] = list(vetex)
		# with open(datafile.replace('.csv', '_v2e.json'), 'w') as f:
		# 	json.dump(v2e_map, f, indent=4)

		# with open(datafile.replace('.csv', '_u2v.json'), 'w') as f:
		# 	json.dump(u2v_map, f, indent=4)

		print("[DEBUG] BEGIN SAVE !")
		
		with open(datafile.replace('.csv', '_v2e_32.pkl'), 'wb') as f:
			pickle.dump(v2e_map, f, protocol = pickle.HIGHEST_PROTOCOL)
		
		with open(datafile.replace('.csv', '_u2v_32.pkl'), 'wb') as f:
			pickle.dump(u2v_map, f, protocol = pickle.HIGHEST_PROTOCOL)