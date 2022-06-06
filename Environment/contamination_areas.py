import numpy as np
import math
import copy


class DetectContaminationAreas():
    def __init__(self, X_test, benchmark, vehicles=4, area=100):
        self.coord = copy.copy(X_test)
        self.coord_real = copy.copy(X_test)
        self.radio = area / vehicles
        self.benchmark = copy.copy(benchmark)
        self.ava = np.array(X_test)

    def real_peaks(self):
        array_max_x = list()
        array_max_y = list()
        array_action_zones = list()
        coordinate_action_zones = list()
        for i in range(len(self.benchmark)):
            if self.benchmark[i] >= 0.33:
                array_action_zones.append(self.benchmark[i])
                coordinate_action_zones.append(self.coord[i])
        while True:
            max_action_zone = max(array_action_zones)
            max_index = array_action_zones.index(max_action_zone)
            max_coordinate = coordinate_action_zones[max_index]
            x_max = max_coordinate[0]
            array_max_x.append(x_max)
            y_max = max_coordinate[1]
            array_max_y.append(y_max)
            coordinate_array = np.array(coordinate_action_zones)
            m = 0
            for i in range(len(array_action_zones)):
                if math.sqrt(
                        (x_max - coordinate_array[i, 0]) ** 2 + (y_max - coordinate_array[i, 1]) ** 2) <= self.radio:
                    index_del = i - m
                    del array_action_zones[index_del]
                    del coordinate_action_zones[index_del]
                    m += 1
            if len(array_action_zones) == 0:
                break
        max_peaks = np.column_stack((array_max_x, array_max_y))

        return max_peaks

    def areas_levels(self, mu):
        dict_ = {}
        dict_coord_ = {}
        dict_impor_ = {}
        dict_index_ = {}
        dict_bench_ = {}
        array_max_x = list()
        array_max_y = list()
        array_action_zones = list()
        coordinate_action_zones = list()
        mean = mu.flat
        j = 0
        impo = 50
        action_zones = list()
        action_zones_index = list()
        bench_az = list()
        array_max_x_bench = list()
        array_max_y_bench = list()
        max_bench_list = list()

        for i in range(len(mean)):
            if mean[i] >= 0.33:
                array_action_zones.append(mean[i])
                coordinate_action_zones.append(self.coord[i])
                bench_az.append(self.benchmark[i])
        while True:
            max_action_zone = max(array_action_zones)
            max_index = array_action_zones.index(max_action_zone)
            max_coordinate = coordinate_action_zones[max_index]
            x_max = max_coordinate[0]
            array_max_x.append(x_max)
            y_max = max_coordinate[1]
            array_max_y.append(y_max)
            coordinate_array = np.array(coordinate_action_zones)
            max_bench = max(bench_az)
            max_bench_list.append(max_bench)
            max_bench_index = bench_az.index(max_bench)
            max_coordinate_bench = coordinate_action_zones[max_bench_index]
            x_max_bench = max_coordinate_bench[0]
            array_max_x_bench.append(x_max_bench)
            y_max_bench = max_coordinate_bench[1]
            array_max_y_bench.append(y_max_bench)
            m = 0
            for i in range(len(array_action_zones)):
                if math.sqrt(
                        (x_max - coordinate_array[i, 0]) ** 2 + (y_max - coordinate_array[i, 1]) ** 2) <= self.radio:
                    index_del = i - m
                    del array_action_zones[index_del]
                    del coordinate_action_zones[index_del]
                    del bench_az[index_del]
                    m += 1
            if len(array_action_zones) == 0:
                break
        center_peaks = np.column_stack((array_max_x, array_max_y))
        center_peaks_bench = np.column_stack((array_max_x_bench, array_max_y_bench))
        for w in range(len(array_max_x)):
            list_zone = list()
            list_coord = list()
            list_impo = list()
            del_list = list()
            coordinate_array = np.array(self.coord)
            for i in range(len(self.coord)):
                if math.sqrt(
                        (array_max_x[w] - coordinate_array[i, 0]) ** 2 + (array_max_y[w] - coordinate_array[i, 1]) ** 2) <= self.radio:
                    list_zone.append(mu[i])
                    list_coord.append(self.coord[i])
                    list_impo.append(impo)
                    del_list.append(i)
            m = 0
            for i in range(len(del_list)):
                index_del = del_list[i] - m
                del self.coord[index_del]
                m += 1
            array_list_coord = np.array(list_coord)
            index = list()
            bench = list()
            for i in range(len(array_list_coord)):
                x = array_list_coord[i, 0]
                y = array_list_coord[i, 1]
                for p in range(len(self.ava)):
                    if x == self.ava[p, 0] and y == self.ava[p, 1]:
                        index.append(p)
                        bench.append(self.benchmark[p])
                        action_zones.append(self.benchmark[p])
                        action_zones_index.append(p)
                        break
            dict_["action_zone%s" % j] = list_zone
            dict_coord_["action_zone%s" % j] = list_coord
            dict_impor_["action_zone%s" % j] = list_impo
            dict_index_["action_zone%s" % j] = index
            dict_bench_["action_zone%s" % j] = bench
            impo -= 10
            j += 1

        return dict_, dict_coord_, dict_impor_, j, center_peaks, dict_index_, dict_bench_, action_zones, center_peaks_bench, max_bench_list
