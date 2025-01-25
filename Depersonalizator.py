from abc import abstractmethod, ABC
import numpy as np


class Depersonalizator(ABC):
    def __init__(self, output_defaults):
        self.output_defaults = output_defaults

    def depersonalize(self, identifiers=None, quasi_identifiers=None, sensitives=None):
        df_is_list = type(quasi_identifiers) == list
        df = np.array(quasi_identifiers, dtype=object)
        sensitives = np.array(sensitives, dtype=object)

        if len(df) == 0:
            if self.output_defaults is not None:
                return identifiers, quasi_identifiers, *self.output_defaults
            else:
                return identifiers, quasi_identifiers

        out_identifiers, out_quasi_identifiers, *other_output = self.__depersonalize__(identifiers, df, sensitives)

        if df_is_list:
            return out_identifiers, out_quasi_identifiers.tolist() if out_quasi_identifiers is not None else None, *other_output

        return out_identifiers, out_quasi_identifiers, *other_output

    @abstractmethod
    def __depersonalize__(self, identifiers, quasi_identifiers, sensitives):
        pass
