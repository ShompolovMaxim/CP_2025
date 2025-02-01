import numpy as np

class GeneralizationRange:
    def __init__(self, mn, mx, column_type, column_values):
        self.column_type = column_type
        if column_type == 'real' or column_type == 'ordered':
            self.min = mn
            self.max = mx
        else:
            self.min = None
            self.max = None
        if column_type == 'unordered':
            self.values_set = np.unique(column_values)
        else:
            self.values_set = None

    def __repr__(self):
        if self.column_type == 'real' or self.column_type == 'ordered':
            return "[" + str(self.min) + ", " + str(self.max) + "]"
        if self.column_type == 'unordered':
            return str(self.values_set)
        return 'Incorrect column type'

    def __eq__(self, other):
        if self.column_type == 'real' or self.column_type == 'ordered':
            return self.min == other.min and self.max == other.max
        if self.column_type == 'unordered':
            return set(self.values_set) == set(other.values_set)
        return False

