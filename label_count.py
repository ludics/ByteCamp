import sys
import numpy as np


if __name__ == "__main__":
    labels = np.load(sys.argv[1])
    labels = np.sort(labels)
    count = 0
    labels_count_dict = {}
    for i in range(labels.shape[0]-1):
        count += 1
        if labels[i+1] > labels[i]:
            labels_count_dict[labels[i]] = count
            count = 0
    labels_count_dict[labels[-1]] = count + 1
    count_list = np.sort(np.array(list(labels_count_dict.values())))

    cnt = {}
    clusters = {}
    for i in labels:
        if i in cnt:
            
