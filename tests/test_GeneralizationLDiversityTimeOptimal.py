import sys

sys.path.append('../')

import unittest
import numpy as np
from GeneralizationLDiversityTimeOptimal import GeneralizationLDiversityTimeOptimal
from utility.GeneralizationRange import GeneralizationRange

class TestGeneralizationLDiversityTimeOptimal(unittest.TestCase):

    def test_initially_k_anonymus(self):
        df = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 1],
        ]
        k = 2
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real']*4).depersonalize(df, sensitives_ids=[4])
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_generalize_last_element(self):
        df = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 3, 1],
        ]
        k = 2
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real']*4).depersonalize(df, sensitives_ids=[4])
        self.assertEqual(k_suppressions, 2)

    def test_generalize_everything(self):
        df = [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 1],
            [3, 3, 3, 3, 1],
            [4, 4, 4, 4, 1],
        ]
        k = 4
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real']*4).depersonalize(df, sensitives_ids=[4])
        general_quasi_identifiers = [[GeneralizationRange(1, 4, 'real', None)] * 4 + [1]]*4
        self.assertEqual(k_anonymus_df, general_quasi_identifiers)
        self.assertEqual(k_suppressions, 16)

    def test_big_k(self):
        df = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2],
            [1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1],
        ]
        k = 5
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real']*4).depersonalize(df, sensitives_ids=[4])
        self.assertEqual(k_anonymus_df, [[1], [2], [2], [1]])
        self.assertEqual(k_suppressions, None)

    def test_k_equals_one(self):
        df = [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 2],
            [4, 4, 4, 4, 1],
        ]
        k = 1
        l = 1
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real']*4).depersonalize(df, sensitives_ids=[4])
        self.assertEqual(k_anonymus_df, df)
        self.assertEqual(k_suppressions, 0)

    def test_numpy(self):
        df = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 1],
        ]
        df = np.array(df)
        k = 2
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real']*4).depersonalize(df, sensitives_ids=[4])
        self.assertTrue((df == k_anonymus_df).all())
        self.assertEqual(k_suppressions, 0)

    def test_string_data(self):
        df = [
            ["a", "a", "a", "a", 1],
            ["a", "a", "a", "a", 2],
            ["b", "b", "b", "b", 2],
            ["b", "b", "b", "b", 1],
        ]
        k = 2
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['ordered']*4).depersonalize(df, sensitives_ids=[4])
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_float_data(self):
        df = [
            [1.0, 1.0, 1.0, 1.0, 1],
            [1.0, 1.0, 1.0, 1.0, 2],
            [2.0, 2.0, 2.0, 2.0, 2],
            [2.0, 2.0, 2.0, 2.0, 1],
        ]
        k = 2
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real']*4).depersonalize(df, sensitives_ids=[4])
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_mixed_data(self):
        df = [
            [1.0, 1.0, "a", 1, 1],
            [1.0, 1.0, "a", 1, 2],
            [2.0, 2.0, "b", 2, 2],
            [2.0, 2.0, "b", 2, 1],
        ]
        k = 2
        l = 2
        k_anonymus_df, k_suppressions = GeneralizationLDiversityTimeOptimal(k, l, ['real', 'real', 'ordered', 'real']).depersonalize(df, sensitives_ids=[4])
        self.assertEqual(df, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)



if __name__ == '__main__':
    unittest.main()
