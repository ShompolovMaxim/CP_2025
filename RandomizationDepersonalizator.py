import numpy as np
import random
from Depersonalizator import Depersonalizator

class RandomizationDepersonalizator(Depersonalizator):
    def __init__(self, min_rand = -1, max_rand = 1, n_rand = 10, seed = None, rand_add = None):
        super().__init__([0])
        self.min_rand = min_rand
        self.max_rand = max_rand
        self.n_rand = n_rand
        self.seed = seed
        self.rand_add = rand_add

    def __depersonalize__(self, df):
        if len(df) == 0:
            return df, 0

        if self.seed is not None:
            random.seed(self.seed)

        if self.rand_add is None:
            rand_add = [None] * len(df[0])
        else:
            rand_add = self.rand_add.copy()

        for i in range(len(rand_add)):
            if rand_add[i] is None:
                rand_add[i] = lambda x: x + random.choice(
                    [self.min_rand + j * (self.max_rand - self.min_rand) / (self.n_rand - 1) for j in range(self._rand)])

        for i in range(len(df)):
            for j in range(len(df[0])):
                df[i][j] = rand_add[j](df[i][j])

        return df
