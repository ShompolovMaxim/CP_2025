def group_by_dist(df_dists, group_size):
    grouped = [False] * len(df_dists)
    groups = []

    i = 0
    while i < len(df_dists):
        if grouped[i]:
            i += 1
            continue
        dists = [(df_dists[i][j], j) for j in range(len(df_dists))]
        dists.sort()
        group = []
        j = 0
        while len(group) < group_size and j < len(df_dists):
            if not grouped[dists[j][1]]:
                group.append(dists[j][1])
                grouped[dists[j][1]] = True
            j += 1
        if len(group) < group_size:
            groups[-1] = groups[-1] + group
        else:
            groups.append(group)
    return groups