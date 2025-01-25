import sys

sys.path.append('../')

import unittest
import numpy as np
from SuppressionLDiversityBaseline import SuppressionLDiversityBaseline

class TestSuppressionLDiversityBaseline(unittest.TestCase):

    def test_initially_l_diverse(self):
        quasi_identifiers = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 2
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertEqual(quasi_identifiers, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_suppress_last_element(self):
        quasi_identifiers = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 3],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 2
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertEqual(k_suppressions, 2)

    def test_suppress_everything(self):
        quasi_identifiers = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 2
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        none_df = [
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
        ]
        self.assertEqual(k_anonymus_df, none_df)
        self.assertEqual(k_suppressions, 16)

    def test_big_k(self):
        quasi_identifiers = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 5
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertEqual(k_anonymus_df, None)
        self.assertEqual(k_suppressions, None)

    def test_k_equals_one(self):
        quasi_identifiers = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 1
        l = 1
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertEqual(k_anonymus_df, quasi_identifiers)
        self.assertEqual(k_suppressions, 0)

    def test_numpy(self):
        df = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        sensitives = [[1], [2], [2], [1]]
        quasi_identifiers = np.array(df)
        k = 2
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertTrue((df == k_anonymus_df).all())
        self.assertEqual(k_suppressions, 0)

    def test_string_data(self):
        quasi_identifiers = [
            ["a", "a", "a", "a"],
            ["a", "a", "a", "a"],
            ["b", "b", "b", "b"],
            ["b", "b", "b", "b"],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 2
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertEqual(quasi_identifiers, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_float_data(self):
        quasi_identifiers = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 2
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertEqual(quasi_identifiers, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)

    def test_mixed_data(self):
        quasi_identifiers = [
            [1.0, 1.0, "a", 1],
            [1.0, 1.0, "a", 1],
            [2.0, 2.0, "b", 2],
            [2.0, 2.0, "b", 2],
        ]
        sensitives = [[1], [2], [2], [1]]
        k = 2
        l = 2
        _, k_anonymus_df, k_suppressions = SuppressionLDiversityBaseline(k, l).depersonalize(quasi_identifiers=quasi_identifiers, sensitives=sensitives)
        self.assertEqual(quasi_identifiers, k_anonymus_df)
        self.assertEqual(k_suppressions, 0)



if __name__ == '__main__':
    unittest.main()
