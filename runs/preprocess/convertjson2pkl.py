import json
import pickle
import sys
if __name__ == '__main__':
	datafile = sys.argv[1]
	with open(datafile, 'r') as f:
		data = json.load(f)
	save_dict = {}
	save_dict['vetexs'] = data['vetexs']
	del data['vetexs']
	for key in data:
		save_dict[int(key)] = {}
		for gid, w in data[key].items():
			save_dict[int(key)][int(gid)] = w
	with open(datafile.replace('json', 'pkl'), 'wb') as f:
		pickle.dump(save_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

