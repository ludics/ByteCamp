import numpy as np
import os
import os.path as osp
import sys
import pickle


VIDEO_NUM = 60034550


def merge_bucket(left_re2ori, left_sub_labels_dict, labels, merge_dir, all_params):
    this_bucket_order = min(left_re2ori.keys())
    this_re2ori = left_re2ori[this_bucket_order]
    this_sub_labels_dict = left_sub_labels_dict[this_bucket_order]
    shift = max(labels) + 1
    for param, sub_labels in this_sub_labels_dict.items():
        sub_labels[np.where(sub_labels > 0)] += shift
        labels[this_re2ori] = sub_labels
        all_params = '_'.join([all_params, param])
        print('bucket {} {} done'.format(this_bucket_order, all_params))
        if this_bucket_order == 9:
            all_path = osp.join(merge_dir, all_params) + '.npy'
            print('save {}'.format(all_path))
            np.save(all_path, labels)
        else:
            this_left_re2ori = {k: v for k, v in left_re2ori.items() if k > this_bucket_order}
            this_left_sub_labels_dict = {k: v for k, v in left_sub_labels_dict.items() if k > this_bucket_order}
            merge_bucket(this_left_re2ori, this_left_sub_labels_dict, labels, merge_dir, all_params)

if __name__ == "__main__":
    result_dir = sys.argv[1]
    bucket_num = int(sys.argv[2])
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
    merge_bucket(bucket_re2ori, bucket_sub_labels_dict, labels, merge_dir, 'all_labels')



    


