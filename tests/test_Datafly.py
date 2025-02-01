import sys

sys.path.append('../')

import unittest
import numpy as np
from Datafly import Datafly
from utility.GeneralizationRange import GeneralizationRange

class TestDatafly(unittest.TestCase):

    def test_initially_k_anonymus(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        k = 2
        k_anonymus_df, k_changes = Datafly(k, ['real']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_changes, 0)

    def test_generalize_last_element(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 3],
        ]
        k = 2
        k_anonymus_df, k_suppressions = Datafly(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_suppressions, 2)

    def test_normal_1(self):
        df = [
            [1, 1, 1, 1],
            [1, 2, 1, 1],
            [2, 3, 2, 2],
            [2, 4, 2, 3],
        ]
        k = 2
        k_anonymus_df, k_suppressions = Datafly(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_suppressions, 6)

    def test_big_k(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        k = 5
        k_anonymus_df, k_suppressions = Datafly(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_anonymus_df, None)
        self.assertEqual(k_suppressions, None)

if __name__ == '__main__':
    unittest.main()
