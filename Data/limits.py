import numpy as np
from sys import path


class Limits:
    def __init__(self, g, part, s_n, grid, xs, ys, n_data, part_ant, s_ant, file=0):
        self.part = part
        self.file = file
        self.s_n = s_n
        self.secure = grid
        self.n_data = n_data
        self.part_ant = part_ant
        self.g = g
        self.s_ant = s_ant
        self.xs = xs
        self.ys = ys
        self.x_int = self.part_ant[self.g, 2]
        self.y_int = self.part_ant[self.g, 3]
        return

    def ratio_s(self):
        x_int = int(self.x_int)
        y_int = int(self.y_int)
        x_left = x_int + 2
        x_right = x_int - 2
        y_up = y_int + 2
        y_down = y_int - 2
        x_i = int(self.part[0])
        y_i = int(self.part[1])
        if self.secure[x_right, y_down] == 1:
            self.part[0] = x_right
            self.part[1] = y_down
        else:
            if self.secure[x_int, y_down] == 1:
                self.part[1] = y_down
                self.part[0] = x_int
            else:
                if self.secure[x_left, y_i] == 1:
                    self.part[0] = x_left
                    self.part[1] = y_int
                else:
                    if self.secure[x_right, y_i] == 1:
                        self.part[0] = x_right
                        self.part[1] = y_int
                    else:
                        if self.secure[x_i, y_up] == 1:
                            self.part[1] = y_up
                            self.part[0] = x_int
                        else:
                            self.part[0] = x_i
                            self.part[1] = y_i
        return self.part

    def new_limit(self):
        with open(path[-1] + '/Data/bounds.npy'.format(self.file), 'rb') as bn:
            df_bounds = np.load(bn)
        x_int = int(self.part[0])
        y_int = int(self.part[1])
        s_1 = self.s_n[0]
        s_2 = self.s_n[1]
        s_3 = self.s_n[2]
        s_4 = self.s_n[3]
        if x_int >= self.xs:
            self.part[0] = self.xs - 1
            x_int = int(self.part[0])
        if y_int >= self.ys:
            self.part[1] = self.ys - 1
            y_int = (self.part[1])
        if self.secure[x_int, y_int] == 0:
            s, n = 0, 0
            bn = list()
            for i in range(len(df_bounds)):
                if int(y_int) == df_bounds[i, 2]:
                    s += 1
                    bn.append(df_bounds[i, :])
            if s == 0:
                if self.part[1] < df_bounds[0, 2]:
                    self.part[1] = df_bounds[0, 2] + 2
                    for i in range(len(df_bounds)):
                        if df_bounds[i, 2] == int(self.part[1]):
                            s += 1
                            bn.append(df_bounds[i, :])
                else:
                    self.part[1] = df_bounds[-1, 2] - 2
                    for i in range(len(df_bounds)):
                        if df_bounds[i, 2] == int(self.part[1]):
                            s += 1
                            bn.append(df_bounds[i, :])
            bn = np.array(bn)
            if self.n_data == 1.0:
                if self.s_ant[0] > 1 and s_1:
                    self.part = Limits(self.g, self.part, self.s_n, self.secure, self.xs, self.ys, self.n_data, self.part_ant, self.s_ant).ratio_s()
                    s_1 = False
                else:
                    if self.part[0] <= bn[0, 0]:
                        self.part[0] = bn[0, 0] + 2
                    else:
                        self.part[0] = bn[0, 1] - 2
                self.s_ant[0] = s
            elif self.n_data == 2.0:
                if self.s_ant[1] > 1 and s_2:
                    self.part = Limits(self.g, self.part, self.s_n, self.secure, self.xs, self.ys, self.n_data, self.part_ant, self.s_ant).ratio_s()
                    s_2 = False
                else:
                    if self.part[0] <= bn[0, 0]:
                        self.part[0] = bn[0, 0] + 2
                    else:
                        self.part[0] = bn[0, 1] - 2
                self.s_ant[1] = s
            elif self.n_data == 3.0:
                if self.s_ant[2] > 1 and s_3:
                    self.part = Limits(self.g, self.part, self.s_n, self.secure, self.xs, self.ys, self.n_data, self.part_ant, self.s_ant).ratio_s()
                    s_3 = False
                else:
                    if self.part[0] <= bn[0, 0]:
                        self.part[0] = bn[0, 0] + 2
                    else:
                        self.part[0] = bn[0, 1] - 2
                self.s_ant[2] = s
            elif self.n_data == 4.0:
                if self.s_ant[3] > 1 and s_4:
                    self.part = Limits(self.g, self.part, self.s_n, self.secure, self.xs, self.ys, self.n_data, self.part_ant, self.s_ant).ratio_s()
                    s_4 = False

                else:
                    if self.part[0] <= bn[0, 0]:
                        self.part[0] = bn[0, 0] + 2
                    else:
                        self.part[0] = bn[0, 1] - 2
                self.s_ant[3] = s
        s_n = [s_1, s_2, s_3, s_4]
        return self.part, s_n



    def Z_var_mean(mu, sigma, X_test, grid):
        Z_var = np.zeros([grid.shape[0], grid.shape[1]])
        Z_mean = np.zeros([grid.shape[0], grid.shape[1]])
        for i in range(len(X_test)):
            Z_var[X_test[i][0], X_test[i][1]] = sigma[i]
            Z_mean[X_test[i][0], X_test[i][1]] = mu[i]
        Z_var[Z_var == 0] = np.nan
        Z_mean[Z_mean == 0] = np.nan
        return Z_var, Z_mean