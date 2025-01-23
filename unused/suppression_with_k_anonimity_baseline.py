import copy
import numpy as np
import sys
from utility.k_anonymity import is_k_anonimus

sys.setrecursionlimit(6000 * 20)

def suppression_k_anonimity_baseline_recursive(df, k, row=0, col=0, k_suppressed=0):
    if col == 0 and row == len(df):
        if is_k_anonimus(df, k):
            return copy.deepcopy(df), k_suppressed
        else:
            return None, None

    next_row = row
    next_col = col + 1
    if col + 1 == len(df[0]):
        next_row = row + 1
        next_col = 0
        
    
    cur = df[row][col]
    df[row][col] = None
    best_df_with_suppression, min_suppressed_with_suppression = \
                        suppression_k_anonimity_baseline_recursive(df, k, next_row, next_col, k_suppressed + 1)
    df[row][col] = cur

    best_df_without_suppression, min_suppressed_without_suppression = \
                        suppression_k_anonimity_baseline_recursive(df, k, next_row, next_col, k_suppressed)

    if best_df_with_suppression is None:
        return best_df_without_suppression, min_suppressed_without_suppression

    if best_df_without_suppression is None:
        best_df_with_suppression[row][col] = None
        return best_df_with_suppression, min_suppressed_with_suppression

    if min_suppressed_with_suppression < min_suppressed_without_suppression:
        best_df_with_suppression[row][col] = None
        return best_df_with_suppression, min_suppressed_with_suppression
    else:
        return best_df_without_suppression, min_suppressed_without_suppression


def suppression_k_anonimity_baseline(df, k):
    df_is_list = type(df) == list
    df = np.array(df, dtype=object)
    
    if len(df) == 0:
        return df, 0

    suppressed_df, k_supperssed = suppression_k_anonimity_baseline_recursive(df, k)

    if df_is_list:
        return suppressed_df.tolist() if suppressed_df is not None else None, k_supperssed

    return suppressed_df, k_supperssed


def suppression_k_anonimity_time_optimal(df, k):
    df_is_list = type(df) == list
    df = np.array(df, dtype=object)

    if len(df) == 0:
        return df, 0

    if len(df) < k:
        return None, None

    grouped = [False] * len(df)
    groups = []
    hamming = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(i+1,len(df)):
            dist = (~(df[i] == df[j])).sum()
            hamming[i][j] = dist
            hamming[j][i] = dist

    i = 0
    while i<len(df):
        if grouped[i]:
            i+=1
            continue
        dists = [(hamming[i][j], j) for j in range(len(df))]
        dists.sort()
        group = []
        j = 0
        while len(group) < k and j < len(df):
            if not grouped[dists[j][1]]:
                group.append(dists[j][1])
                grouped[dists[j][1]] = True
            j += 1
        if len(group) < k:
            groups[-1] = groups[-1] + group
        else:
            groups.append(group)
        
    
    n_suppressions = 0
    suppressed_df = np.zeros(df.shape, dtype=object)
    for group in groups:
        mask = df[group[0]] == df[group[0]]
        for i in range(1, len(group)):
            mask = mask & (df[group[0]] == df[group[i]])
        mask = ~mask
        for i in group:
            row = df[i].copy()
            row[mask] = None
            n_suppressions += mask.sum()
            suppressed_df[i] = row

    if df_is_list:
        return suppressed_df.tolist() if suppressed_df is not None else None, n_suppressions

    return suppressed_df, n_suppressions


def generalization_k_anonimity_time_optimal(df, k):
    df_is_list = type(df) == list
    df = np.array(df, dtype=object)

    if len(df) == 0:
        return df, 0

    if len(df) < k:
        return None, None

    sorted_columns = []
    for i in range(len(df[0])):
        sorted_columns.append(sorted(df[:,i].tolist()))

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
        for j in range(i+1,len(df)):
            dist = np.abs(df_rngs[i] - df_rngs[j]).sum()
            #for m in range(len(df[0])):
            #    dist += abs(rngs[m][df[i][m]] - rngs[m][df[j][m]])
            my_dist[i][j] = dist
            my_dist[j][i] = dist

    i = 0
    while i<len(df):
        if grouped[i]:
            i+=1
            continue
        dists = [(my_dist[i][j], j) for j in range(len(df))]
        dists.sort()
        group = []
        j = 0
        while len(group) < k and j < len(df):
            if not grouped[dists[j][1]]:
                group.append(dists[j][1])
                grouped[dists[j][1]] = True
            j += 1
        if len(group) < k:
            groups[-1] = groups[-1] + group
        else:
            groups.append(group)
        
    n_suppressions = 0
    suppressed_df = np.zeros(df.shape, dtype=object)
    for group in groups:
        mask = df[group[0]] == df[group[0]]
        mn = df[group[0]].copy()
        mx = df[group[0]].copy()
        for i in range(1, len(group)):
            mask = mask & (df[group[0]] == df[group[i]])
            mn = np.minimum(mn, df[group[i]])
            mx = np.maximum(mx, df[group[i]])
        rng = np.array(list(map(lambda x: "[" + str(x[0]) + ", " + str(x[1]) + "]", zip(mn, mx))))
        mask = ~mask
        for i in group:
            row = df[i].copy()
            row[mask] = rng[mask]
            n_suppressions += mask.sum()
            suppressed_df[i] = row

    if df_is_list:
        return suppressed_df.tolist() if suppressed_df is not None else None, n_suppressions

    return suppressed_df, n_suppressions

    
        
    
