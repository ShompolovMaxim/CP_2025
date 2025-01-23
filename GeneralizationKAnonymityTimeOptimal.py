from Depersonalizator import Depersonalizator
import numpy as np
from utility.diatances import dfs_rank_distances
from utility.groupping import group_by_dist
from utility.GeneralizationRange import GeneralizationRange


class GeneralizationKAnonymityTimeOptimal(Depersonalizator):
    def __init__(self, k):
        super().__init__([0])
        self.k = k

    def __depersonalize__(self, df):
        if len(df) == 0:
            return df, 0

        if len(df) < self.k:
            return None, None

        my_dist = dfs_rank_distances(df)
        groups = group_by_dist(my_dist, self.k)

        n_generalizations = 0
        generalized_df = np.zeros(df.shape, dtype=object)
        for group in groups:
            mask = df[group[0]] == df[group[0]]
            mn = df[group[0]].copy()
            mx = df[group[0]].copy()
            for i in range(1, len(group)):
                mask = mask & (df[group[0]] == df[group[i]])
                mn = np.minimum(mn, df[group[i]])
                mx = np.maximum(mx, df[group[i]])
            rng = np.array(list(map(lambda x: GeneralizationRange(x[0], x[1]), zip(mn, mx))))
            mask = ~mask
            for i in group:
                row = df[i].copy()
                row[mask] = rng[mask]
                n_generalizations += mask.sum()
                generalized_df[i] = row

        return generalized_df, n_generalizations