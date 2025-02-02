import sys

sys.path.append('../')

import unittest
import numpy as np
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
        k_anonymus_df, k_suppressions = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 1)

    def test_normal_1(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 3],
        ]
        k = 2
        k_anonymus_df, k_suppressions = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_suppressions, 2)

    def test_normal_2(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 4],
            [2, 2, 2, 2],
            [2, 2, 2, 3],
        ]
        k = 2
        k_anonymus_df, k_suppressions = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_suppressions, 3)

    def test_normal_3(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 2],
            [2, 2, 2, 3],
            [2, 2, 2, 4],
        ]
        k = 2
        k_anonymus_df, k_suppressions = AggregationGreedyByOneEqualSizedGroups(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_suppressions, 2)

if __name__ == '__main__':
    unittest.main()
