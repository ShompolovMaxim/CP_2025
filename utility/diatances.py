import numpy as np
from importlib_metadata import pass_none


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

def dfs_rank_absolute_dist(df, value_class):
    sorted_columns = []
    for i in range(len(df[0])):
        sorted_columns.append(sorted(df[:, i].tolist()))

    ranks = []
    for i in range(len(df[0])):
        if value_class[i] == 'real':
            ranks.append(None)
            continue
        rng = dict()
        for j in range(len(sorted_columns[i])):
            if sorted_columns[i][j] not in rng:
                rng[sorted_columns[i][j]] = j
        ranks.append(rng)

    df_mins = [None] * len(df[0])  # Rewrite in numpy
    df_maxs = [None] * len(df[0])
    for i in range(len(df[0])):
        if value_class[i] != 'real':
            continue
        for j in range(len(df)):
            if df_mins[i] is None or df_mins[i] > df[j][i]:
                df_mins[i] = df[j][i]
            if df_maxs[i] is None or df_maxs[i] < df[j][i]:
                df_maxs[i] = df[j][i]

    df_values_dists = np.zeros((len(df), len(df[0])))
    for i in range(len(df)):
        for j in range(len(df[0])):
            if value_class[j] == 'real':
                df_values_dists[i][j] = 0 if df_maxs[i] == df_mins[i] or df_maxs[i] is None or df_mins[i] is None\
                    else (df[i][j] - df_mins[i]) / (df_maxs[i] - df_mins[i])
            else:
                df_values_dists[i][j] = ranks[j][df[i][j]] / len(df)

    my_dist = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        if i % 100 == 0:
            print(i)
        for j in range(i + 1, len(df)):
            dist = np.abs(df_values_dists[i] - df_values_dists[j]).sum()
            my_dist[i][j] = dist
            my_dist[j][i] = dist

    return my_dist