from Depersonalizator import Depersonalizator
import copy
from utility.metrics import is_l_diverse

class SuppressionLDiversityBaseline(Depersonalizator):
    def __init__(self, k, l):
        super().__init__([0])
        self.k = k
        self.l = l

    def __depersonalize__(self, identifiers, quasi_identifiers, sensitives, row=0, col=0, k_suppressed=0):
        if col == 0 and row == len(quasi_identifiers):
            if is_l_diverse(quasi_identifiers, sensitives, self.k, self.l):
                return None, copy.deepcopy(quasi_identifiers), k_suppressed
            else:
                return None, None, None

        next_row = row
        next_col = col + 1
        if col + 1 == len(quasi_identifiers[0]):
            next_row = row + 1
            next_col = 0

        cur = quasi_identifiers[row][col]
        quasi_identifiers[row][col] = None
        _, best_df_with_suppression, min_suppressed_with_suppression = \
            self.__depersonalize__(identifiers, quasi_identifiers, sensitives, next_row, next_col, k_suppressed + 1)
        quasi_identifiers[row][col] = cur

        _, best_df_without_suppression, min_suppressed_without_suppression = \
            self.__depersonalize__(identifiers, quasi_identifiers, sensitives, next_row, next_col, k_suppressed)

        if best_df_with_suppression is None:
            return None, best_df_without_suppression, min_suppressed_without_suppression

        if best_df_without_suppression is None:
            best_df_with_suppression[row][col] = None
            return None, best_df_with_suppression, min_suppressed_with_suppression

        if min_suppressed_with_suppression < min_suppressed_without_suppression:
            best_df_with_suppression[row][col] = None
            return None, best_df_with_suppression, min_suppressed_with_suppression
        else:
            return None, best_df_without_suppression, min_suppressed_without_suppression