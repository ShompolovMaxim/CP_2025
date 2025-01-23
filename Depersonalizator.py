from abc import abstractmethod, ABC
import numpy as np


class Depersonalizator(ABC):
    def __init__(self, output_defaults):
        self.output_defaults = output_defaults

    def depersonalize(self, df):
        df_is_list = type(df) == list
        df = np.array(df, dtype=object)

        if len(df) == 0:
            if self.output_defaults is not None:
                return df, *self.output_defaults
            else:
                return df

        out_df, *other_output = self.__depersonalize__(df)

        if df_is_list:
            return out_df.tolist() if out_df is not None else None, *other_output

        return out_df, *other_output

    @abstractmethod
    def __depersonalize__(self, df):
        pass
