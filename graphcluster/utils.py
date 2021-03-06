import numpy as np
import pandas
import time
import networkx as nx
from matplotlib import pyplot as plt
import collections 
import argparse 
import scipy.sparse as scp
import os.path as osp
import os
import pandas
import random
import queue
import seaborn
import json


def read_ground_truth():
    gt_v = set()
    gt_e = []
    with open("data/graph_data/byte_camp_truth_pair.txt", "r") as f:
        for l in f:
            [a, b] = list(map(lambda x: int(x), l.split(',')))
            gt_v.add(a)
            gt_v.add(b)
            gt_e.append([a,b])
    return gt_v, gt_e


# 计算两个概率list的相对熵
def calculate_kl(prob_list_1, prob_list_2):
    KL = 0.0
    for i, prob_a in enumerate(prob_list_1):
        KL +=  prob_a * np.log(prob_a / prob_list_2[i], 2)

    return KL


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


def cal_modularity(adjacency, clusters, sel_v):
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
    index = np.array(sel_v) - 1
    clusters = clusters[index]
    adjacency = adjacency[index,:][:,index]
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


def cal_conductance(adjacency, clusters, sel_v):
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
    index = np.array(sel_v) - 1
    clusters = clusters[index]
    adjacency = adjacency[index,:][:,index]
    cluster_indices = np.zeros(adjacency.shape[0], dtype=np.bool)
    for cluster_id in np.unique(clusters):
        cluster_indices[:] = 0
        cluster_indices[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_indices, :]
        inter += np.sum(adj_submatrix[:, cluster_indices])
        intra += np.sum(adj_submatrix[:, ~cluster_indices])
    print(f'cal conductance is {time.time() - start}s')
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
        if clusters[e[0]-1] != -1 and clusters[e[0]-1] == clusters[e[1]-1]:
            tp_dict[clusters[e[0]-1]] += 1
            TP += 1
    clusters = clusters[index]
    precision = 0
    for c in np.unique(clusters):
        if c == -1:
            continue
        csize = (np.where(clusters == c)[0]).shape[0]
        if csize == 1:
            #print(f'label {c} nums is one')
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
    with open("byte_camp_truth_pair.txt", "r") as f:
        for l in f.readlines():
            [a, b] = list(map(lambda x: int(x), l.strip().split(',')))
            gt_v.add(a)
            gt_v.add(b)
            if a != b:
                    a, b = min(a, b), max(a, b)
                    #assert (a, b) not in gt_e, f'{[a, b]} this edge has occured'
                    #assert (b, a) not in gt_e, f'{[b, a]} this edge has occured'
                    gt_e.add((a,b))
            
    return list(gt_v), list(gt_e)


def draw_gt(edge_list, id):
    """
    args:
        edge_list: list[tuple]
    """
    start = time.time()
    # edge_list = random.sample(edge_list, 5000)
    g = nx.Graph(edge_list)
    pos = nx.nx_agraph.graphviz_layout(g)
    nx.draw(g, node_size=40, width=0.5, pos=pos)
    plt.savefig(f'gt_{id}.png')
    print(f'draw graph time is {time.time() - start}s')
    plt.close()
    return pos
    
def draw_origin(matrix, sel_v, id, pos, index):
    # sel_v = np.array(sel_v) - 1
    sel_v = np.array(index)
    sel_matrix = matrix[sel_v,:][:,sel_v]
    g = nx.Graph(sel_matrix)
    nx.draw(g, node_size=40, width=0.5, pos=pos)
    plt.savefig(f'origin_{id}.png')
    plt.close()
    

    

def draw_pred(sel_e, predict, id, pos):
    import ipdb;
    sel_e = np.array(sel_e) - 1
    predict = predict[sel_e]
    # print(predict)
    matrix = np.zeros((len(sel_e), len(sel_e)))
    for c in np.unique(predict):
        if c == -1: continue
        index = np.where(predict == c)[0]
        # print(index)
        for i in index:
            matrix[i,index] = 1
    # print(matrix)
    g = nx.Graph(matrix)
    nx.draw(g, node_size=40, width=0.5, pos=pos)
    plt.savefig(f'predict_{id}.png')
    plt.close()
    
def bfs(v, c):
    q = queue.Queue()
    q.put(v)
    color[v] = c
    while(q.qsize()):
        cur = q.get()
        for i in adj_matrix[cur]:
            if color[i] == 0:
                q.put(i)
                color[i] = c


def select_vetexs(gt_v, gt_e):
    adj_matrix = collections.defaultdict(list)
    for e in gt_e:
        adj_matrix[e[0]].append(e[1])
        adj_matrix[e[1]].append(e[0])
    color = collections.defaultdict(int)
    c = 1
    for v in gt_v:
        if color[v] == 0:
            q = queue.Queue()
            q.put(v)
            color[v] = c
            while(q.qsize()):
                cur = q.get()
                for i in adj_matrix[cur]:
                    if color[i] == 0:
                        q.put(i)
                        color[i] = c
            c += 1
    c2v = collections.defaultdict(list)
    for k, v in color.items():
        c2v[v].append(k)
    vetexs = []
    maxnum = 0
    for c, vs in c2v.items():
        if  30 < len(vs):
            print(len(vs))
            vetexs.append(vs)
            

    return_edges = []
    return_vs = []
    for i in range(len(vetexs)):
        vs = set(vetexs[i])
        
        return_edge = []
        for e in gt_e:
            if e[0] in vs and e[1] in vs:
                return_edge.append(e)
        return_vs.append(list(vs))
        return_edges.append(return_edge)

    return return_edges, return_vs
    

def drawgrah(gt_v, gt_e, adj_matrix, clusters):
    # edges, vetexs = select_vetexs(gt_v, gt_e)
    index_dict = json.load(open('index.json'))
    sel_vs = list(map(int, index_dict.keys()))
    # import ipdb;ipdb.set_trace()
    sel_v = list(random.sample(sel_vs, 60))
    sel_e = []
    for e in gt_e:
        if e[0] in sel_v and e[1] in sel_v:
            sel_e.append(e)
    pos = draw_gt(sel_e, 4)
    new_pos = {}
    index = []
    for i, v in enumerate(sel_v):
        new_pos[i] = pos[v]
        index.append(index_dict[str(v)])
    draw_pred(sel_v, predict, 4, new_pos)
    draw_origin(adj_matrix, sel_v, 4, new_pos, index)
    # for i in range(3):
    #     sel_v = sel_vs[i*20:(i+1)*20]
    #     sel_e = []
    #     for e in gt_e:
    #         if e[0] in sel_v and e[1] in sel_v:
    #             sel_e.append(e)
    #     pos = draw_gt(sel_e, i)
    #     index = []
    #     new_pos = {}
    #     for j, v in enumerate(sel_v):
    #         index.append(index_dict[str(v)])
    #         new_pos[j] = pos[int(v)]
    #     draw_pred(sel_v, predict, i, new_pos)

    #     draw_origin(adj_matrix, sel_v, i, new_pos, index)


def parse_args():
    parser = argparse.ArgumentParser(description='some evaluation method')
    parser.add_argument('--sparse_matrix', default='/root/workspace/GraphCluster/sparse_matrix/year_sim.npz')
    parser.add_argument('--predict_root', default='/root/workspace/GraphCluster/dbscan_result/10-bucket/')
    parser.add_argument('--predict_file', default='/Users/sabrina/Documents/bytecamp/ByteCamp/0b_0.8_1_1b_0.8_1_2b_0.5_3_3b_0.5_3_4b_0.5_3_5b_0.6_3_6b_0.5_3_7b_0.5_3_8b_0.05_3_9b_0.05_1.npy')

    args = parser.parse_args()
    return args

def drawmatrix(gt_v, gt_e, matrix, predict):
    
    index = []
    sel_e = []
    for i, v in enumerate(gt_v):
        index.extend(v)
        sel_e.extend(gt_e[i])
    size = len(index)
    matrix = np.zeros((size, size))
    index = np.array(index) - 1
    predict = predict[index]
    for c in np.unique(predict):
        if c == -1:continue 
        ids = np.where(predict == c)[0]
        for i in ids:
            matrix[i,ids] = 1
    ax = seaborn.heatmap(matrix, vmin=0, vmax=1, cmap='GnBu')
    fig = ax.get_figure()
    fig.savefig('matrix_predict.png')
    #index = random.sample(gt_v, 1000)
    
    # index = np.sort(index)
    sel_matrix = matrix[index,:][:,index]
    #origin_img = (sel_matrix.toarray()*255).astype('int')
    img = sel_matrix.toarray()
    ax = seaborn.heatmap(img, vmin=0, vmax=1, cmap='GnBu')
    fig = ax.get_figure()
    fig.savefig('matrix_origin.png')
    index_dict = {}
    for i, v in enumerate(index):
        index_dict[v] = i
    matrix_gt = np.zeros((size, size))
    for e in sel_e:
        v1, v2 = index_dict[e[0]], index_dict[e[1]]
        matrix_gt[v1][v2] = 1
        matrix_gt[v2][v1] = 1
    ax = seaborn.heatmap(matrix_gt, vmin=0, vmax=1, cmap='GnBu')
    fig = ax.get_figure()
    fig.savefig('matrix_gt.png')


def visualize(args):
    gt_v, gt_e = read_ground_truth()
    adj_matrix = scp.load_npz(args.sparse_matrix)
    predict_file = osp.join(args.predict_root, args.predict_file)
    predict = np.load(predict_file)
    drawgrah(gt_v, gt_e, adj_matrix, predict)

def evaluate(args):
    precisons = []
    recalls = []
    fscores = []
    params = []
    moduls = []
    conductants = []
    matrix = scp.load_npz(args.sparse_matrix)
    gt_v, gt_e = read_ground_truth()
    files = [f for f in os.listdir(args.predict_root) if f.endswith('npy') and f.startswith('year_result')][:7]
    for f in files:
        param = f.rsplit('.', 1)[0]
        params.append(param)

        predict = np.load(osp.join(args.predict_root, f))
        f1, pre, recall = f1_score(predict, gt_v, gt_e)
        fscores.append(f1)
        precisons.append(pre)
        recalls.append(recall)
        print(f'f1score:{f1}, precision:{pre}, recall:{recall}')
        modularity = cal_modularity(matrix, predict, gt_v)
        moduls.append(modularity)
        print(f'modularity is {modularity}')
        ##cal conductance
        conductance = cal_conductance(matrix, predict, gt_v)
        conductants.append(conductance)
        print(f'conductance is {conductance}')
    dataframe = pandas.DataFrame({'params':params, 'f1score':fscores, 'modularity':moduls,\
        'precision':precisons, 'recall':recalls, 'conductance':conductants})
    dataframe.to_csv('dbscan_1_bucket.csv', index=False, sep=',')

if __name__ == '__main__':
    args = parse_args()
    random.seed(123)
    #draw graph
    #drawgrah(args)
    #evaluate
    evaluate(args)

    




