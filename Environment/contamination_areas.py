import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from Environment.plot import Plots
import math


class DetectContaminationAreas():
    def __init__(self, X_test, vehicles=4, area=100):
        self.X_test = X_test
        self.radio = area / vehicles

    def areas_levels(self, mean):
        dict_ = {}
        array_action_zones = list()
        coordinate_action_zones = list()
        i = 0
        for i in range(len(mean)):
            if mean[i] >= 0.33:
                array_action_zones.append(mean[i])
                coordinate_action_zones.append(self.X_test[i])
        while True:
            i += 1
            list_zone = list()
            max_action_zone = max(array_action_zones)
            max_coordinate = max_action_zone.index(max_action_zone)
            x_max = max_coordinate[0]
            y_max = max_coordinate[1]
            coordinate_array = np.array(coordinate_action_zones)
            for i in range(len(array_action_zones)):
                if math.sqrt(
                        (x_max - coordinate_array[i, 0]) ** 2 - (y_max - coordinate_array[i, 1]) ** 2) <= self.radio:
                    list_zone.append(array_action_zones)
                    del array_action_zones[i]
                    del coordinate_action_zones[i]
            dict_["action_zone%s" % i] = list_zone
            if len(array_action_zones) == 0:
                break