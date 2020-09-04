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

