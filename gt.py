def read_ground_truth():
    gt_v = set()
    gt_e = []
    with open("/data00/graph_data/byte_camp_truth_pair.txt", "r") as f:
        for l in f:
            [a, b] = list(map(lambda x: int(x), l.split(',')))
            gt_v.add(a)
            gt_v.add(b)
            gt_e.append([a,b])
    return gt_v, gt_e
