from Depersonalizator import Depersonalizator
import numpy as np
from utility.diatances import dfs_rank_general_dist
from utility.groupping import group_by_dist
from utility.GeneralizationRange import GeneralizationRange


class GeneralizationKAnonymityTimeOptimal(Depersonalizator):
    def __init__(self, k, quasi_identifiers_types = None):
        super().__init__([0])
        self.k = k
        self.quasi_identifiers_types = quasi_identifiers_types

    def __depersonalize__(self, identifiers, quasi_identifiers, sensitives):
        if len(quasi_identifiers) == 0:
            return None, quasi_identifiers, 0

        if len(quasi_identifiers) < self.k:
            return None, None, None

        if self.quasi_identifiers_types is None:
            self.quasi_identifiers_types = ['unordered'] * len(quasi_identifiers[0])

        my_dist = dfs_rank_general_dist(quasi_identifiers, self.quasi_identifiers_types)
        groups = group_by_dist(my_dist, self.k)

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
            rng = np.array(list(map(lambda x: GeneralizationRange(x[0], x[1], x[2], x[3]),
                                    zip(mn, mx, self.quasi_identifiers_types, np.transpose(quasi_identifiers[group])))))
            mask = ~mask
            for i in group:
                row = quasi_identifiers[i].copy()
                row[mask] = rng[mask]
                n_generalizations += mask.sum()
                generalized_df[i] = row

        return None, generalized_df, n_generalizations