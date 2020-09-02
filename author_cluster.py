import numpy as np
import pandas as np
import collections


# 统计作者所在的cluster类别
def author_cluster(author_edge, cluster):
    author_cnts = {}
    for author, vectors in author_edge.items():
        cnt = collections.Counter()
        for edge in vectors:
            cnt[cluster[edge]] += 1
        author_cnts[author] = cnt.most_common(200)

    return author_cnts


if __name__ == '__main__':
    cluster_file = sys.argv[1]
    with open(cluster_file, "rb") as f:
		cluster = np.load(f)
    author_edge_file = "/root/workspace/GraphCluster/adj_list/year_gid_uid_pair_u2v.pkl"
    with open(author_edge_file, "rb") as f:
		author_edge = pickle.load(f)

    author_cnts = author_cluster(author_edge, cluster)
