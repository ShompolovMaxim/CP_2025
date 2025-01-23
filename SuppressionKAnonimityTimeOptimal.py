from Depersonalizator import Depersonalizator
import numpy as np
from utility.diatances import dfs_hamming_distances
from utility.groupping import group_by_dist

class SuppressionKAnonymityTimeOptimal(Depersonalizator):
    def __init__(self, k):
        super().__init__([0])
        self.k = k

    def __depersonalize__(self, df):
        if len(df) == 0:
            return df, 0

        if len(df) < self.k:
            return None, None

        hamming = dfs_hamming_distances(df)
        groups = group_by_dist(hamming, self.k)

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

        return suppressed_df, n_suppressions