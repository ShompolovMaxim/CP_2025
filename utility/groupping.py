def group_by_dist(dists, group_size):
    grouped = [False] * len(dists)
    groups = []

    i = 0
    while i < len(dists):
        if grouped[i]:
            i += 1
            continue
        cur_dists = [(dists[i][j], j) for j in range(len(dists))]
        cur_dists.sort()
        group = []
        j = 0
        while len(group) < group_size and j < len(cur_dists):
            if not grouped[cur_dists[j][1]]:
                group.append(cur_dists[j][1])
                grouped[cur_dists[j][1]] = True
            j += 1
        if len(group) < group_size:
            groups[-1] = groups[-1] + group
        else:
            groups.append(group)
    return groups

def group_by_dist_with_l_diverse(dists, sensitives, group_size, l):
    grouped = [False] * len(dists)
    groups = []

    i = 0
    while i < len(dists):
        if grouped[i]:
            i += 1
            continue
        i_dists = [(dists[i][j], j) for j in range(len(dists))]
        i_dists.sort()
        group = []
        group_sensitives = []
        j = 0
        while len(group) < group_size and len(group) < l and j < len(i_dists):
            if not grouped[i_dists[j][1]] and sensitives[j] not in group_sensitives:
                group.append(i_dists[j][1])
                grouped[i_dists[j][1]] = True
                group_sensitives.append(sensitives[j])
            j += 1
        j = 0
        while len(group) < group_size and j < len(i_dists):
            if not grouped[i_dists[j][1]]:
                group.append(i_dists[j][1])
                grouped[i_dists[j][1]] = True
            j += 1
        if len(group) < group_size:
            groups[-1] = groups[-1] + group
        else:
            groups.append(group)
    return groups