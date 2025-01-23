class GeneralizationRange:
    def __init__(self, mn, mx):
        self.min = mn
        self.max = mx

    def __repr__(self):
        return "[" + str(self.min) + ", " + str(self.max) + "]"

    def __eq__(self, other):
        return self.min == other.min and self.max == other.max

