import numpy as np
import math


class DetectContaminationAreas():
    def __init__(self, X_test, vehicles=4, area=100):
        self.X_test = X_test
        self.radio = area / vehicles

    def areas_levels(self, mu):
        dict_ = {}
        dict_coord_ = {}
        dict_impor_ = {}
        max_x = list()
        max_y = list()
        array_action_zones = list()
        coordinate_action_zones = list()
        mean = mu.flat
        j = 0
        impo = 50
        for i in range(len(mean)):
            if mean[i] >= 0.33:
                array_action_zones.append(mean[i])
                coordinate_action_zones.append(self.X_test[i])
        while True:
            j += 1
            list_zone = list()
            list_coord = list()
            list_impo = list()
            del_list = list()
            max_action_zone = max(array_action_zones)
            max_index = array_action_zones.index(max_action_zone)
            max_coordinate = coordinate_action_zones[max_index]
            x_max = max_coordinate[0]
            max_x.append(x_max)
            y_max = max_coordinate[1]
            max_y.append(y_max)
            coordinate_array = np.array(coordinate_action_zones)
            #if len(max_x) != 1:
             #   for i in range(len(max_x)):
              #      for j in range(i + 1, len(max_x)):
               #         if math.sqrt((max_x[i] - max_x[j]) ** 2 + (max_y[i] - max_y[j]) ** 2) <= self.radio:

            for i in range(len(array_action_zones)):
                if math.sqrt(
                        (x_max - coordinate_array[i, 0]) ** 2 + (y_max - coordinate_array[i, 1]) ** 2) <= self.radio:
                    list_zone.append(array_action_zones[i])
                    list_coord.append(coordinate_action_zones[i])
                    list_impo.append(impo)
                    del_list.append(i)
            print(del_list)
            m = 0
            for i in range(len(del_list)):
                index_del = del_list[i] - m
                del array_action_zones[index_del]
                del coordinate_action_zones[index_del]
                m += 1
            dict_["action_zone%s" % j] = list_zone
            dict_coord_["action_zone%s" % j] = list_coord
            dict_impor_["action_zone%s" % j] = list_impo
            #print(dict_["action_zone%s" % i])
            #print(dict_impor_["action_zone%s" % i])
            impo -= 10
            if len(array_action_zones) == 0:
                break
       # for i in range(len(max_x)):
        #    for j in range(i + 1, len(max_x)):
         #       if math.sqrt((max_x[i] - max_x[j]) ** 2 + (max_y[i] - max_y[j]) ** 2) <= self.radio:
          #          y = i + 1
           #         z = j + 1
            #        list1 = list(dict_["action_zone%s" % y])
             #       list11 = list(dict_coord_["action_zone%s" % y])
              #      list12 = list(dict_impor_["action_zone%s" % y])
               #     list2 = list(dict_["action_zone%s" % z])
                #    list21 = list(dict_coord_["action_zone%s" % z])
                 #   list22 = list(dict_impor_["action_zone%s" % z])

        return dict_, dict_coord_, dict_impor_, j
