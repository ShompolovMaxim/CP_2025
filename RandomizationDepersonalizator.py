from collections.abc import Iterable
import numpy as np
import random
import bisect
from Depersonalizator import Depersonalizator

class RandomOrdered:
    def __init__(self, sorted_values):
        self.sorted_values = sorted_values

    def __call__(self, x):
        pos_left = bisect.bisect_left(self.sorted_values, x)
        pos = pos_left
        shift = int(np.random.normal(0, 1, 1)[0] * len(self.sorted_values) / 4)
        new_pos = pos + shift
        if new_pos < 0:
            new_pos = 0
        if new_pos >= len(self.sorted_values):
            new_pos = len(self.sorted_values) - 1
        return self.sorted_values[new_pos]

class RandomizationBaselineDepersonalizator(Depersonalizator):
    def __init__(self, *, range_narrowing = 10, min_rand = None, max_rand = None, n_rand = 10, seed = None, rand_add = None, quasi_identifiers_types = None):
        super().__init__([0])
        self.range_narrowing = range_narrowing
        self.min_rand = min_rand
        self.max_rand = max_rand
        self.n_rand = n_rand
        self.seed = seed
        self.rand_add = rand_add
        self.quasi_identifiers_types = quasi_identifiers_types

    def __depersonalize__(self, identifiers, quasi_identifiers, sensitives):
        if len(quasi_identifiers) == 0:
            return None, quasi_identifiers

        if self.seed is not None:
            random.seed(self.seed)

        if self.rand_add is None:
            self.rand_add = [None] * len(quasi_identifiers[0])
        else:
            self.rand_add = self.rand_add.copy()

        if self.quasi_identifiers_types is None:
            self.quasi_identifiers_types = ['unordered'] * len(quasi_identifiers[0])

        if not isinstance(self.min_rand, Iterable):
            self.min_rand = [self.min_rand] * len(quasi_identifiers[0])
        if not isinstance(self.max_rand, Iterable):
            self.max_rand = [self.max_rand] * len(quasi_identifiers[0])

        for i in range(len(self.rand_add)):
            if self.rand_add[i] is None:
                if self.quasi_identifiers_types[i] == 'real':
                    #min_rand = self.min_rand[i] if self.min_rand[i] is not None else -(np.max(quasi_identifiers[:, i]) - np.min(quasi_identifiers[:, i])) / self.range_narrowing
                    #max_rand = self.max_rand[i] if self.max_rand[i] is not None else (np.max(quasi_identifiers[:, i]) - np.min(quasi_identifiers[:, i])) / self.range_narrowing
                    #self.rand_add[i] = lambda x: x + random.choice(
                    #    [min_rand + j * (max_rand - min_rand) / (self.n_rand - 1) for j in range(self.n_rand)]
                    #)
                    hist, bin_edges = np.histogram(quasi_identifiers[:, i].astype(float), bins=30, density=True)
                    new_values = np.random.choice(bin_edges[:-1], size=len(quasi_identifiers[:, i]), p=hist * np.diff(bin_edges))
                    self.rand_add[i] = RandomOrdered(np.sort(new_values))
                elif self.quasi_identifiers_types[i] == 'unordered':
                    self.rand_add[i] = lambda x: x if random.choice([0,1]) == 0 else (
                        random.choice(quasi_identifiers[:, i].tolist())
                    )
                elif self.quasi_identifiers_types[i] == 'ordered':
                    values = sorted(quasi_identifiers[:, i].tolist())
                    self.rand_add[i] = RandomOrdered(values)

        for i in range(len(quasi_identifiers)):
            for j in range(len(quasi_identifiers[0])):
                quasi_identifiers[i][j] = self.rand_add[j](quasi_identifiers[i][j])

        return None, quasi_identifiers

if __name__ == '__main__':
    df = [
        [1, 1000, 0.1],
        [0, 0, 0],
        [3, 3000, 0.3],
        [5, 5000, 0.5],
    ]
    print(RandomizationBaselineDepersonalizator(quasi_identifiers_types=['ordered']*3).depersonalize(df))
