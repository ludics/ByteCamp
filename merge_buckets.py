import numpy as np
import os
import os.path as osp
import sys
import pickle
from metrics import f1_score, read_ground_truth
import argparse
import pandas as pd
import time

VIDEO_NUM = 60034550


def parse_args():
    parser = argparse.ArgumentParser(description='some evaluation method')
    parser.add_argument('--sparse_matrix', default='/root/workspace/GraphCluster/sparse_matrix/year_sim.npz')
    parser.add_argument('--predict_root', default='/root/workspace/GraphCluster/dbscan_result/1-bucket/')
    parser.add_argument('--predict_file', default='year_result_0.01_5.npy')
    parser.add_argument('--bucket_num', default=10)
    parser.add_argument('--result_dir', default='/root/workspace/GraphCluster/dbscan_result/10-bucket/')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    result_dir = args.result_dir
    bucket_num = int(args.bucket_num)
    bucket_sub_labels_dict = {}

    for k in range(bucket_num):
        bucket_sub_labels_dict[k] = {}
        result_dir_k = osp.join(result_dir, 'candidate', str(k))
        names = [name for name in os.listdir(result_dir_k) if "result" in name]
        for name in names:
            full_path = osp.join(result_dir_k, name)
            para_info = '_'.join('.'.join(full_path.split('.')[:-1]).split('_')[-3:])
            bucket_sub_labels_dict[k][para_info] = np.load(full_path)
    
    with open(osp.join(result_dir, 'bucket_re2ori_list.pkl'), 'rb') as f:
        bucket_re2ori = pickle.load(f)
    labels = np.array([-1] * VIDEO_NUM).astype('int32')
    merge_dir = osp.join(result_dir, 'merged')
    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir)
    params = {}
    params[0] = '0bucket_0.8_1'
    sub_labels_0 = bucket_sub_labels_dict[0]['0bucket_0.8_1']
    sub_labels_0[np.where(sub_labels_0 >= 0)] *= 10
    labels[bucket_re2ori[0]] = sub_labels_0

    gt_v, gt_e = read_ground_truth()
    precisons = []
    recalls = []
    fscores = []
    all_params = []
    max_f1 = 0
    count = 0
    start = time.time()
    for param_1, sub_labels_1 in bucket_sub_labels_dict[1].items():
        params[1] = param_1
        sub_labels_1[np.where(sub_labels_1 >= 0)] *= 10
        sub_labels_1[np.where(sub_labels_1 >= 0)] += 1
        labels[bucket_re2ori[1]] = sub_labels_1
        for param_2, sub_labels_2 in bucket_sub_labels_dict[2].items():
            params[2] = param_2
            sub_labels_2[np.where(sub_labels_2 >= 0)] *= 10
            sub_labels_2[np.where(sub_labels_2 >= 0)] += 2
            labels[bucket_re2ori[2]] = sub_labels_2
            for param_3, sub_labels_3 in bucket_sub_labels_dict[3].items():
                params[3] = param_3
                sub_labels_3[np.where(sub_labels_3 >= 0)] *= 10
                sub_labels_3[np.where(sub_labels_3 >= 0)] += 3
                labels[bucket_re2ori[3]] = sub_labels_3
                for param_4, sub_labels_4 in bucket_sub_labels_dict[4].items():
                    params[4] = param_4
                    sub_labels_4[np.where(sub_labels_4 >= 0)] *= 10
                    sub_labels_4[np.where(sub_labels_4 >= 0)] += 4
                    labels[bucket_re2ori[4]] = sub_labels_4
                    for param_5, sub_labels_5 in bucket_sub_labels_dict[5].items():
                        params[5] = param_5
                        sub_labels_5[np.where(sub_labels_5 >= 0)] *= 10
                        sub_labels_5[np.where(sub_labels_5 >= 0)] += 5
                        labels[bucket_re2ori[5]] = sub_labels_5
                        for param_6, sub_labels_6 in bucket_sub_labels_dict[6].items():
                            params[6] = param_6
                            sub_labels_6[np.where(sub_labels_6 >= 0)] *= 10
                            sub_labels_6[np.where(sub_labels_6 >= 0)] += 6
                            labels[bucket_re2ori[6]] = sub_labels_6
                            for param_7, sub_labels_7 in bucket_sub_labels_dict[7].items():
                                params[7] = param_7
                                sub_labels_7[np.where(sub_labels_7 >= 0)] *= 10
                                sub_labels_7[np.where(sub_labels_7 >= 0)] += 7
                                labels[bucket_re2ori[7]] = sub_labels_7
                                for param_8, sub_labels_8 in bucket_sub_labels_dict[8].items():
                                    params[8] = param_8
                                    sub_labels_8[np.where(sub_labels_8 >= 0)] *= 10
                                    sub_labels_8[np.where(sub_labels_8 >= 0)] += 8
                                    labels[bucket_re2ori[8]] = sub_labels_8
                                    for param_9, sub_labels_9 in bucket_sub_labels_dict[9].items():
                                        params[9] = param_9
                                        sub_labels_9[np.where(sub_labels_9 >= 0)] *= 10
                                        sub_labels_9[np.where(sub_labels_9 >= 0)] += 9
                                        labels[bucket_re2ori[9]] = sub_labels_9
                                        f1, pre, recall = f1_score(labels, gt_v, gt_e)
                                        this_params = '_'.join(list(params.values())).replace("ucket", "")
                                        all_params.append(this_params)
                                        fscores.append(f1)
                                        precisons.append(pre)
                                        recalls.append(recall)
                                        np.save(osp.join(result_dir, this_params + '.npy'), labels)
                                        if f1 > max_f1:
                                            max_f1 = f1
                                            print(f'count:{count}, params:{this_params}')
                                            print(f'\tf1score:{f1}, precision:{pre}, recall:{recall}')
                                        if count % 1000 == 0:
                                            print(f'time cost:{time.time() - start}, count:{count}')
                                            start = time.time()
                                            # print('\tf1score:{f1}, precision:{pre}, recall:{recall}')                                           
                                        count += 1

    dataframe = pd.DataFrame({'all_params':all_params, 'f1score': fscores, 'precision': precisons, 'recall': recalls})
    dataframe.to_csv(osp.join(result_dir, 'dbscan_10_bucket.csv'), index=False, sep=',')
