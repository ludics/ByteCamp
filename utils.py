import numpy as np

def get_size_dist(name_dir):
    labels_ori = np.load(name_dir)
    labels = np.sort(labels_ori)
    count = 0
    cluster2size = {}
    for i in range(labels.shape[0]-1):
        count += 1
        if labels[i+1] > labels[i]:
            cluster2size[labels[i]] = count
            count = 0
    cluster2size[labels[-1]] = count + 1
    return np.sort(np.array(list(cluster2size.values())))
