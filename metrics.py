import pandas
import time
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import collections 
import argparse 
import scipy.sparse as scp
import os.path as osp
import os
import pandas
def cal_modularity(adjacency, clusters):
    """Computes graph modularity.
    Args:
        adjacency: sparse adjacency matrix, type: scipy.sparse.csr_matrix
        clusters: type(numpy.array) shape(numvetexs) the same cluters's vetexs shoud have the same label
    Returns:
        The value of graph modularity.
    """
    start = time.time()
    vetexnums = adjacency.shape[0]
    adjacency[range(vetexnums), range(vetexnums)] = 0
    nozero = adjacency.nonzero()
    adjacency[nozero[0], nozero[1]] = 1
    degrees = adjacency.sum(axis=0).A1
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
    print(f'cal modularity time is {time.time() - start}s')
    return result / n_edges


def cal_conductance(adjacency, clusters):
    """Computes graph conductance 
    Args:
    adjacency: sparse adjacency matrix, type: scipy.sparse.csr_matrix
    clusters: type(numpy.array) shape(numvetexs) the same cluters's vetexs shoud have the same label
    Returns:
    The average conductance value of the graph clusters.
    """
    start = time.time()
    inter = 0  # Number of inter-cluster edges.
    intra = 0  # Number of intra-cluster edges.
    vetexnums = adjacency.shape[0]
    adjacency[range(vetexnums), range(vetexnums)] = 0
    cluster_indices = np.zeros(adjacency.shape[0], dtype=np.bool)
    for cluster_id in np.unique(clusters):
        cluster_indices[:] = 0
        cluster_indices[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_indices, :]
        inter += np.sum(adj_submatrix[:, cluster_indices])
        intra += np.sum(adj_submatrix[:, ~cluster_indices])
    print(f'cal conductance is {time() - start}s')
    return intra / (inter + intra)


def f1_score(clusters, gt_v, gt_e):
    """
    args:
        c: cluster_labe, the same cluters's vetexs shoud have the same label (list / array)
        gt_v: list, groudtruth vetex
        gt_e: list[tuple], groudtruth edge
    return:
        f1score, precision, recall
    """
    start = time.time()
    tp_dict = collections.defaultdict(int)
    index = np.array(gt_v) - 1
    TP = 0
    for e in gt_e:
        if clusters[e[0]-1] == clusters[e[1]-1]:
            tp_dict[clusters[e[0]-1]] += 1
            TP += 1
    clusters = clusters[index]
    precision = 0
    for c in np.unique(clusters):
        csize = (np.where(clusters == c)[0]).shape[0]
        if csize == 1:
            print(f'label {c} nums is zero')
            continue
        kc = (csize * csize - csize) >> 1
        precision += tp_dict[c] / kc * csize

    precision /= len(gt_v) 
    recall = TP / len(gt_e)

    fscore = 2 * precision * recall / (precision + recall)
    print(f'evaluate fscore time is {time.time()-start}s')
    return 2 * precision * recall / (precision + recall), precision, recall

def read_ground_truth():
    gt_v = set()
    gt_e = set()
    with open("/data00/graph_data/byte_camp_truth_pair.txt", "r") as f:
        for l in f.readlines():
            [a, b] = list(map(lambda x: int(x), l.strip().split(',')))
            gt_v.add(a)
            gt_v.add(b)
            #assert (a, b) not in gt_e, f'{[a, b]} this edge has occured'
            #assert (b, a) not in gt_e, f'{[b, a]} this edge has occured'
            gt_e.add((a,b))
            
    return list(gt_v), list(gt_e)


def drawgraph(edge_list):
    """
    args:
        edge_list: list[tuple]
    """
    g = nx.Graph(edge_list)
    nx.draw(g)
    plt.savefig(name)
    

def parse_args():
    parser = argparse.ArgumentParser(description='some evaluation method')
    parser.add_argument('--sparse_matrix', default='/root/workspace/GraphCluster/sparse_matrix/year_sim.npz')
    parser.add_argument('--predict_root', default='/root/workspace/GraphCluster/dbscan_result/1-bucket/')
    parser.add_argument('--predict_file', default='year_result_0.01_5.npy')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    #draw graph
    #cal modularity
    #start = time.time()
    #adj_matrix = scp.load_npz(args.sparse_matrix)
    #print(f'loading matirx time is {time.time()-start}s')
    # predict_file = osp.join(args.predict_root, args.predict_file)
    # predict = np.load(predict_file)
    #modularity = cal_modularity(adj_matrix, predict)
    #print(f'modularity is {modularity}')
    ##cal conductance
    #conductance = cal_conductance(adj_matrix, predict)
    #print(f'conductance is {conductance}')
    #cal fscore
    files = [f for f in os.listdir(args.predict_root) if f.startswith('year_result')]
    eps = []
    minpoint = []
    precisons = []
    recalls = []
    fscores = []
    gt_v, gt_e = read_ground_truth()
    for f in files:
        items = f.rsplit('.', 1)[0].split('_')
        eps.append(float(items[-2]))
        minpoint.append(float(items[-1]))
        predict = np.load(osp.join(args.predict_root, f))
        f1, pre, recall = f1_score(predict, gt_v, gt_e)
        fscores.append(f1)
        precisons.append(pre)
        recalls.append(recall)
        print(f'f1score:{f1}, precision:{pre}, recall:{recall}')
    dataframe = pandas.DataFrame({'eps':eps, 'minpoint':minpoint, 'f1score':fscores, \
        'precision':precisons, 'recall':recalls})
    dataframe.to_csv('dbscan_1_bucket.csv', index=False, sep=',')
    
    
    
    
