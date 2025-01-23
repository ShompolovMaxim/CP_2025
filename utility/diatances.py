import numpy as np

def dfs_hamming_distances(df):
    hamming = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dist = (~(df[i] == df[j])).sum()
            hamming[i][j] = dist
            hamming[j][i] = dist
    return hamming

def dfs_rank_distances(df):
    sorted_columns = []
    for i in range(len(df[0])):
        sorted_columns.append(sorted(df[:, i].tolist()))

    ranks = []
    for i in range(len(df[0])):
        rng = dict()
        for j in range(len(sorted_columns[i])):
            if sorted_columns[i][j] not in rng:
                rng[sorted_columns[i][j]] = j
        ranks.append(rng)

    df_ranks = np.zeros((len(df), len(df[0])))
    for i in range(len(df)):
        for j in range(len(df[0])):
            df_ranks[i][j] = ranks[j][df[i][j]]

    my_dist = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        if i % 100 == 0:
            print(i)
        for j in range(i + 1, len(df)):
            dist = np.abs(df_ranks[i] - df_ranks[j]).sum()
            my_dist[i][j] = dist
            my_dist[j][i] = dist

    return my_dist