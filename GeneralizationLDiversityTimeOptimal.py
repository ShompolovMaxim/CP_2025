from Depersonalizator import Depersonalizator
import numpy as np
from utility.diatances import dfs_rank_distances
from utility.groupping import group_by_dist_with_l_diverse
from utility.GeneralizationRange import GeneralizationRange


class GeneralizationLDiversityTimeOptimal(Depersonalizator):
    def __init__(self, k, l):
        super().__init__([0])
        self.k = k
        self.l = l

    def __depersonalize__(self, identifiers, quasi_identifiers, sensitives):
        if len(quasi_identifiers) == 0:
            return None, quasi_identifiers, 0

        if len(quasi_identifiers) < self.k:
            return None, None, None

        my_dist = dfs_rank_distances(quasi_identifiers)
        groups = group_by_dist_with_l_diverse(my_dist, sensitives, self.k, self.l)

        n_generalizations = 0
        generalized_df = np.zeros(quasi_identifiers.shape, dtype=object)
        for group in groups:
            mask = quasi_identifiers[group[0]] == quasi_identifiers[group[0]]
            mn = quasi_identifiers[group[0]].copy()
            mx = quasi_identifiers[group[0]].copy()
            for i in range(1, len(group)):
                mask = mask & (quasi_identifiers[group[0]] == quasi_identifiers[group[i]])
                mn = np.minimum(mn, quasi_identifiers[group[i]])
                mx = np.maximum(mx, quasi_identifiers[group[i]])
            rng = np.array(list(map(lambda x: GeneralizationRange(x[0], x[1]), zip(mn, mx))))
            mask = ~mask
            for i in group:
                row = quasi_identifiers[i].copy()
                row[mask] = rng[mask]
                n_generalizations += mask.sum()
                generalized_df[i] = row

        return None, generalized_df, n_generalizations