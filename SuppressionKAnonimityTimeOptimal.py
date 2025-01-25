from Depersonalizator import Depersonalizator
import numpy as np
from utility.diatances import dfs_hamming_distances
from utility.groupping import group_by_dist

class SuppressionKAnonymityTimeOptimal(Depersonalizator):
    def __init__(self, k):
        super().__init__([0])
        self.k = k

    def __depersonalize__(self, identifiers, quasi_identifiers, sensitives):
        if len(quasi_identifiers) == 0:
            return None, quasi_identifiers, 0

        if len(quasi_identifiers) < self.k:
            return None, None, None

        hamming = dfs_hamming_distances(quasi_identifiers)
        groups = group_by_dist(hamming, self.k)

        n_suppressions = 0
        suppressed_df = np.zeros(quasi_identifiers.shape, dtype=object)
        for group in groups:
            mask = quasi_identifiers[group[0]] == quasi_identifiers[group[0]]
            for i in range(1, len(group)):
                mask = mask & (quasi_identifiers[group[0]] == quasi_identifiers[group[i]])
            mask = ~mask
            for i in group:
                row = quasi_identifiers[i].copy()
                row[mask] = None
                n_suppressions += mask.sum()
                suppressed_df[i] = row

        return None, suppressed_df, n_suppressions