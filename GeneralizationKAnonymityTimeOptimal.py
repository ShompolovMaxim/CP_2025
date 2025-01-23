from Depersonalizator import Depersonalizator
import numpy as np
from utility.diatances import dfs_rank_distances

class GeneralizationKAnonymityTimeOptimal(Depersonalizator):
    def __init__(self, k):
        super().__init__([0])
        self.k = k

    def __depersonalize__(self, df):
        if len(df) == 0:
            return df, 0

        if len(df) < self.k:
            return None, None

        grouped = [False] * len(df)
        groups = []
        my_dist = dfs_rank_distances(df)

        i = 0
        while i < len(df):
            if grouped[i]:
                i += 1
                continue
            dists = [(my_dist[i][j], j) for j in range(len(df))]
            dists.sort()
            group = []
            j = 0
            while len(group) < self.k and j < len(df):
                if not grouped[dists[j][1]]:
                    group.append(dists[j][1])
                    grouped[dists[j][1]] = True
                j += 1
            if len(group) < self.k:
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

        return suppressed_df, n_suppressions