# ByteCamp

## 项目介绍

本项目为字节跳动夏令营2020研发赛道算法组F-01小组项目：基于视频关联网络的相关性挖掘。

代码组织结构如下：

```
.
├── README.md
├── data
│   ├── adj_list
│   ├── graph_data
│   ├── sparse_matrix
│   └── user_feature
├── graphcluster
│   ├── my_dbscan.py
│   ├── user
│   │   ├── plagiarism.py
│   │   ├── user_cluster_stat.py
│   │   ├── user_graph.py
│   │   ├── user_similarity.py
│   │   └── utils.py
│   ├── utils.py
│   └── video
├── result
│   ├── user_dbscan
│   └── video_dbscan
└── runs
    ├── preprocess
    │   ├── convertjson2pkl.py
    │   ├── csv2pkl.py
    │   └── make_sparse_matrix.py
    ├── user
    │   ├── author_cluster.py
    │   ├── make_user_significance.py
    │   └── user_dbscan.py
    └── video
        ├── dbscan.py
        ├── dbscan_try_param.py
        ├── merge_buckets.py
        ├── online_cluster.py
        └── split_sparse_matrix.py
```

其中，`graphcluster` 为库函数文件夹，`runs` 为运行脚本文件夹，`data` 为原始数据与中间数据文件夹，`result` 为聚类结果文件夹。


本项目主要分为三个部分：数据预处理、视频聚类和视频作者信息挖掘。相应的代码分别保存在preprocess、video和user目录下。

-  数据预处理部分：将原始数据图处理成邻接表和稀疏矩阵格式，进行数据统计。同时保存作者与视频的对应关系，便于后续使用。
-  视频聚类部分：搜索出网络上的所有极大联通子图，根据其规模进行分类，对不同的子图使用不同参数执行聚类算法(DBSCAN)，最后将所有子图上的聚类结果合并得到聚类结果。
-  作者信息挖掘部分：利用聚类结果挖掘诸如作者相似性(make_user_similarity_graph)、作者原创性(page_rank)、作者重要性(compute_user_significance)等信息，并在此基础上进一步尝试挖掘有效的信息。
