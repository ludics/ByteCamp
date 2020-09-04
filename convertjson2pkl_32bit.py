import json
import pickle
import sys
import tqdm
import numpy as np
if __name__ == '__main__':
	datafile = sys.argv[1]
	with open(datafile, 'rb') as f:
		data = json.load(f)
	print("[DEBUG] data load OK")
	save_dict = {}
	save_dict['vetexs'] = [np.int32(i) for i in data['vetexs']]
	del data['vetexs']
	for key in tqdm.tqdm(data):
		save_dict[np.int32(key)] = {}
		for gid, w in data[key].items():
			save_dict[np.int32(key)][np.int32(gid)] = np.float32(w)
	del data
	print("[DEBUG] Begin Save")
	with open(datafile.replace('.json', '_32bit.pkl'), 'wb') as f:
		pickle.dump(save_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

