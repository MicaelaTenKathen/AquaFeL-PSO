import numpy as np
from sys import path

class Limits:
    def __init__(self, part, s_n, file=0):
        self.part = part
        self.file = file
        self.s_n = s_n

    def new_limit(self):
        with open(path[-1] + '/Data/bounds.npy'.format(self.file), 'rb') as bn:
            df_bounds = np.load(bn)
        x_int = int(self.part[0])
        y_int = int(self.part[1])
        s_1 = self.s_n[0]
        s_2 = self.s_n[1]
        s_3 = self.s_n[2]
        s_4 = self.s_n[3]
        if x_int >= xs:
            part[0] = xs - 1
            x_int = int(part[0])
        if y_int >= ys:
            part[1] = ys - 1
            y_int = (part[1])
        if secure[x_int, y_int] == 0:
            s, n = 0, 0
            bn = list()
            for i in range(len(df_bounds)):
                if int(y_int) == df_bounds[i, 2]:
                    s += 1
                    bn.append(df_bounds[i, :])
            if s == 0:
                if part[1] < df_bounds[0, 2]:
                    part[1] = df_bounds[0, 2] + 2
                    for i in range(len(df_bounds)):
                        if df_bounds[i, 2] == int(part[1]):
                            s += 1
                            bn.append(df_bounds[i, :])
                else:
                    part[1] = df_bounds[-1, 2] - 2
                    for i in range(len(df_bounds)):
                        if df_bounds[i, 2] == int(part[1]):
                            s += 1
                            bn.append(df_bounds[i, :])
            bn = np.array(bn)
            if n_data == 1.0:
                if s_ant[0] > 1 and s_1:
                    part = ratio_s(part_ant[g, 0], part_ant[g, 1], secure, part)
                    s_1 = False
                else:
                    if part[0] <= bn[0, 0]:
                        part[0] = bn[0, 0] + 2
                    else:
                        part[0] = bn[0, 1] - 2
                s_ant[0] = s
            elif n_data == 2.0:
                if s_ant[1] > 1 and s_2:
                    part = ratio_s(part_ant[g, 2], part_ant[g, 3], secure, part)
                    s_2 = False
                else:
                    if part[0] <= bn[0, 0]:
                        part[0] = bn[0, 0] + 2
                    else:
                        part[0] = bn[0, 1] - 2
                s_ant[1] = s
            elif n_data == 3.0:
                if s_ant[2] > 1 and s_3:
                    part = ratio_s(part_ant[g, 4], part_ant[g, 5], secure, part)
                    s_3 = False
                else:
                    if part[0] <= bn[0, 0]:
                        part[0] = bn[0, 0] + 2
                    else:
                        part[0] = bn[0, 1] - 2
                s_ant[2] = s
            elif n_data == 4.0:
                if s_ant[3] > 1 and s_4:
                    part = ratio_s(part_ant[g, 6], part_ant[g, 7], secure, part)
                    s_4 = False

                else:
                    if part[0] <= bn[0, 0]:
                        part[0] = bn[0, 0] + 2
                    else:
                        part[0] = bn[0, 1] - 2
                s_ant[3] = s
        else:
            s = 0
            for i in range(len(df_bounds)):
                if int(y_int) == df_bounds[i, 2]:
                    s += 1
            if n_data == 1.0:
                if s_ant[0] > 1 and s_1:
                    part = ratio_s(part_ant[g, 0], part_ant[g, 1], secure, part)
                    s_1 = False
                else:

                    part = part
                s_ant[0] = s
            elif n_data == 2.0:
                if s_ant[1] > 1 and s_2:
                    part = ratio_s(part_ant[g, 2], part_ant[g, 3], secure, part)
                    s_2 = False

                else:
                    part = part
                s_ant[1] = s
            elif n_data == 3.0:
                if s_ant[2] > 1 and s_3:
                    part = ratio_s(part_ant[g, 4], part_ant[g, 5], secure, part)
                    s_3 = False

                else:
                    part = part
                s_ant[2] = s
            elif n_data == 4.0:
                if s_ant[3] > 1 and s_4:
                    part = ratio_s(part_ant[g, 6], part_ant[g, 7], secure, part)
                    s_4 = False

                else:
                    part = part
                s_ant[3] = s
        s_n = [s_1, s_2, s_3, s_4]
        return part, s_n


    def Z_var_mean(mu, sigma, X_test, grid):
        Z_var = np.zeros([grid.shape[0], grid.shape[1]])
        Z_mean = np.zeros([grid.shape[0], grid.shape[1]])
        for i in range(len(X_test)):
            Z_var[X_test[i][0], X_test[i][1]] = sigma[i]
            Z_mean[X_test[i][0], X_test[i][1]] = mu[i]
        Z_var[Z_var == 0] = np.nan
        Z_mean[Z_mean == 0] = np.nan
        return Z_var, Z_mean