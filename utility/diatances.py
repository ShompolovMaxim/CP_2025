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

    rngs = []
    for i in range(len(df[0])):
        rng = dict()
        for j in range(len(sorted_columns[i])):
            if sorted_columns[i][j] not in rng:
                rng[sorted_columns[i][j]] = j
        rngs.append(rng)

    df_rngs = np.zeros((len(df), len(df[0])))
    for i in range(len(df)):
        for j in range(len(df[0])):
            df_rngs[i][j] = rngs[j][df[i][j]]

    grouped = [False] * len(df)
    groups = []
    my_dist = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        if i % 100 == 0:
            print(i)
        for j in range(i + 1, len(df)):
            dist = np.abs(df_rngs[i] - df_rngs[j]).sum()
            my_dist[i][j] = dist
            my_dist[j][i] = dist

    return my_dist