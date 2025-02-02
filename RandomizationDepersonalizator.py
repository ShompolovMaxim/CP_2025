from collections.abc import Iterable
import numpy as np
import random
from Depersonalizator import Depersonalizator

class RandomizationBaselineDepersonalizator(Depersonalizator):
    def __init__(self, *, range_narrowing = 10, min_rand = None, max_rand = None, n_rand = 10, seed = None, rand_add = None):
        super().__init__([0])
        self.range_narrowing = range_narrowing
        self.min_rand = min_rand
        self.max_rand = max_rand
        self.n_rand = n_rand
        self.seed = seed
        self.rand_add = rand_add

    def __depersonalize__(self, identifiers, quasi_identifiers, sensitives):
        if len(quasi_identifiers) == 0:
            return None, quasi_identifiers

        if self.seed is not None:
            random.seed(self.seed)

        if self.rand_add is None:
            self.rand_add = [None] * len(quasi_identifiers[0])
        else:
            self.rand_add = self.rand_add.copy()

        if not isinstance(self.min_rand, Iterable):
            self.min_rand = [self.min_rand] * len(quasi_identifiers[0])
        if not isinstance(self.max_rand, Iterable):
            self.max_rand = [self.max_rand] * len(quasi_identifiers[0])

        for i in range(len(self.rand_add)):
            if self.rand_add[i] is None:
                min_rand = self.min_rand[i] if self.min_rand[i] is not None else -(np.max(quasi_identifiers[:, i]) - np.min(quasi_identifiers[:, i])) / self.range_narrowing
                max_rand = self.max_rand[i] if self.max_rand[i] is not None else (np.max(quasi_identifiers[:, i]) - np.min(quasi_identifiers[:, i])) / self.range_narrowing
                self.rand_add[i] = lambda x: x + random.choice(
                    [min_rand + j * (max_rand - min_rand) / (self.n_rand - 1) for j in range(self.n_rand)]
                )

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
    print(RandomizationBaselineDepersonalizator().depersonalize(df))
