import random
def modularity(g, c, times):
    s = 0
    for _ in range(times):
        i = random.randint(1, 60034549)
        j = random.randint(0, i-1)
        if c[i]==c[j]:
            if i in g[j]:
                s += g[j][i]-degree[i]*degree[j]/2.0/276307620
            else:
                s -= degree[i]*degree[j]/2.0/276307620
    return s/times

def conductance(m, c, times):
    s = 0
    l = len(m.row)
    for _ in range(times):
        i = random.randint(0, l-1)
        if c[m.row[i]]!=c[m.col[i]]: s += 1
    return s/((times<<1)-s)

def f1_score(c, gt_v, gt_e):
    part_c = {}
    for v in gt_v:
        part_c[v] = c[v]
    TP = 0
    for e in gt_e:
        if part_c[e[0]]==part_c[e[1]]:
            TP += 1
    d = {}
    for x in part_c:
        if part_c[x] in d:
            d[part_c[x]] += 1
        else:
            d[part_c[x]] = 1
    total = 0
    for i in d:
        total += (d[i]*d[i]-d[i])>>1
    
    precision = TP/len(gt_e)
    recall = TP/total
