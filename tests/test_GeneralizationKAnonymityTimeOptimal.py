import sys

sys.path.append('../')

import unittest
import numpy as np
from GeneralizationKAnonymityTimeOptimal import GeneralizationKAnonymityTimeOptimal
from utility.GeneralizationRange import GeneralizationRange

class TestGeneralizationKAnonymityTimeOptimal(unittest.TestCase):

    def test_initially_k_anonymus(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        k = 2
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_generalize_last_element(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 3],
        ]
        k = 2
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_suppressions, 2)

    def test_generalize_everything(self):
        df = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ]
        k = 4
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real']*4).depersonalize(df)
        none_df = [[GeneralizationRange(1, 4, 'real', None)] * 4]*4
        self.assertEqual(k_anonymus_df, none_df)
        self.assertEqual(k_suppressions, 16)

    def test_big_k(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        k = 5
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_anonymus_df, None)
        self.assertEqual(k_suppressions, None)

    def test_k_equals_one(self):
        df = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ]
        k = 1
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real']*4).depersonalize(df)
        self.assertEqual(k_anonymus_df, df)
        self.assertEqual(k_suppressions, 0)

    def test_numpy(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        df = np.array(df)
        k = 2
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real']*4).depersonalize(df)
        self.assertTrue((df == k_anonymus_df).all())
        self.assertEqual(k_suppressions, 0)

    def test_string_data(self):
        df = [
            ["a", "a", "a", "a"],
            ["a", "a", "a", "a"],
            ["b", "b", "b", "b"],
            ["b", "b", "b", "b"],
        ]
        k = 2
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['ordered']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_float_data(self):
        df = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
        k = 2
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real']*4).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_mixed_data(self):
        df = [
            [1.0, 1.0, "a", 1],
            [1.0, 1.0, "a", 1],
            [2.0, 2.0, "b", 2],
            [2.0, 2.0, "b", 2],
        ]
        k = 2
        k_anonymus_df, k_suppressions = GeneralizationKAnonymityTimeOptimal(k, ['real', 'real', 'ordered', 'real']).depersonalize(df)
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)



if __name__ == '__main__':
    unittest.main()
