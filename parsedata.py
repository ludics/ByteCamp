import csv
import collections
import sys
import json
import pickle

if __name__ == '__main__':
	datafile = sys.argv[1]
	with open(datafile, 'r') as f:
		datas = csv.reader(f)
		edge_nums = 0
		v2e_map = collections.defaultdict(dict)
		u2v_map = collections.defaultdict(set)
		for gid1, uid1, gid2, uid2, w in datas:
			gid1, uid1, gid2, uid2, w = int(gid1), int(uid1), int(gid2), int(uid2), float(w)
			# import ipdb;ipdb.set_trace()
			if min(gid1, gid2) in v2e_map[max(gid1, gid2)]:
				break
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
		# vetex = set()
		# for uid in u2v_map:
		# 	vetex |= u2v_map[uid]
		# 	u2v_map[uid] = list(u2v_map[uid])
		# assert len(vetex) == vetex_nums, 'diff user has same video'
		v2e_map['vetexs'] = list(vetex)
		with open(datafile.replace('.csv', '_v2e.json'), 'w') as f:
			json.dump(v2e_map, f, indent=4)

		with open(datafile.replace('.csv', '_u2v.json'), 'w') as f:
			json.dump(u2v_map, f, indent=4)
		
		with open(datafile.replace('.csv', '_v2e.pkl'), 'wb') as f:
			pickle.dump(v2e_map, f, protocol = pickle.HIGHEST_PROTOCOL)
		
		with open(datafile.replace('.csv', '_u2v.pkl'), 'wb') as f:
			pickle.dump(u2v_map, f, protocol = pickle.HIGHEST_PROTOCOL)