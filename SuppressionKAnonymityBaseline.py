from Depersonalizator import Depersonalizator
import copy
from utility.metrics import is_k_anonimus

class SuppressionKAnonymityBaseline(Depersonalizator):
    def __init__(self, k):
        super().__init__([0])
        self.k = k

    def __depersonalize__(self, df, row=0, col=0, k_suppressed=0):
        if col == 0 and row == len(df):
            if is_k_anonimus(df, self.k):
                return copy.deepcopy(df), k_suppressed
            else:
                return None, None

        next_row = row
        next_col = col + 1
        if col + 1 == len(df[0]):
            next_row = row + 1
            next_col = 0

        cur = df[row][col]
        df[row][col] = None
        best_df_with_suppression, min_suppressed_with_suppression = \
            self.__depersonalize__(df, next_row, next_col, k_suppressed + 1)
        df[row][col] = cur

        best_df_without_suppression, min_suppressed_without_suppression = \
            self.__depersonalize__(df, next_row, next_col, k_suppressed)

        if best_df_with_suppression is None:
            return best_df_without_suppression, min_suppressed_without_suppression

        if best_df_without_suppression is None:
            best_df_with_suppression[row][col] = None
            return best_df_with_suppression, min_suppressed_with_suppression

        if min_suppressed_with_suppression < min_suppressed_without_suppression:
            best_df_with_suppression[row][col] = None
            return best_df_with_suppression, min_suppressed_with_suppression
        else:
            return best_df_without_suppression, min_suppressed_without_suppression


