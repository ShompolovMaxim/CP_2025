import sys

sys.path.append('../')

import unittest
import numpy as np
import random
from AggregationGreedyByOneEqualSizedGroups import AggregationGreedyByOneEqualSizedGroups
from utility.GeneralizationRange import GeneralizationRange

class TestAggregationGreedyByOneEqualSizedGroups(unittest.TestCase):

    def test_initially_k_anonymus(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(group_size, 1)

    def test_normal_1(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 3],
        ]
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(group_size, 2)

    def test_normal_2(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 4],
            [2, 2, 2, 2],
            [2, 2, 2, 3],
        ]
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(group_size, 3)

    def test_normal_3(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 2],
            [2, 2, 2, 3],
            [2, 2, 2, 4],
        ]
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(group_size, 2)

    def test_generalize_everything(self):
        df = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ]
        k = 4
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        none_df = [[GeneralizationRange(1, 4, 'real', None)] * 4]*4
        self.assertEqual(k_anonymus_df, none_df)
        self.assertEqual(group_size, 4)

    def test_k_equals_one(self):
        df = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ]
        k = 1
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_anonymus_df, df)
        self.assertEqual(group_size, 1)

    def test_numpy(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        df = np.array(df)
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertTrue((df == k_anonymus_df).all())
        self.assertEqual(group_size, 1)

    def test_string_data(self):
        df = [
            ["a", "a", "a", "a"],
            ["a", "a", "a", "a"],
            ["b", "b", "b", "b"],
            ["b", "b", "b", "b"],
        ]
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['ordered']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(group_size, 1)

    def test_float_data(self):
        df = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(group_size, 1)

    def test_mixed_data(self):
        df = [
            [1.0, 1.0, "a", 1],
            [1.0, 1.0, "a", 1],
            [2.0, 2.0, "b", 2],
            [2.0, 2.0, "b", 2],
        ]
        k = 2
        k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real', 'real', 'ordered', 'real']).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(group_size, 1)

    def test_random_df(self):
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        for i in range(1000):
            rows = random.randint(4, 50)
            cols = random.randint(1, 5)
            df = np.random.randint(0, 3, (rows, cols))
            k = random.randint(2, 4)
            k_anonymus_df, group_size = AggregationGreedyByOneEqualSizedGroups(k, ['real']*cols).depersonalize(df)


if __name__ == '__main__':
    unittest.main()
