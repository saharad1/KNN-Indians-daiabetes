from point import Point
from numpy import mean, var


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class MinMaxNormalizer:
    """
    This is the class of the min-max normalizer, that is calculated by the his formula.
    """
    def __init__(self):
        self.min_max_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.min_max_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.min_max_list.append([min(values), max(values)])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.min_max_list[i][0]) /
                               (self.min_max_list[i][1] - self.min_max_list[i][0]) for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class SumNormalizer:
    """
    This is the class of the sum normalizer, that is calculated by the his formula.
    """
    def __init__(self):
        self.sum_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.sum_list = []
        for i in range(len(all_coordinates[0])):
            values = [abs(x[i]) for x in all_coordinates]
            self.sum_list.append([sum(values)])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] / self.sum_list[i][0]) for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class ZNormalizer:
    """
    This is the class of the Z-normalizer, that is calculated by the his formula.
    """
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1)**0.5])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new
