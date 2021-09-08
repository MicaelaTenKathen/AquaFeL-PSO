from Environment.map import Map
from Data.data_path import bounds_path, grid_path, available_path, secure_path, se_available_path
import pandas as pd
import numpy as np
from sys import path


class Bounds():
    def __init__(self, resolution, xs, ys, load_file=False, file=0, available=None, first=None, last=None, y_first=None,
                 y_last=None, index=None, first_x=None, last_x=None, all_y=None, confirm=None, se_first=None,
                 se_last=None, se_available=None):
        self.resolution = resolution
        self.xs = xs
        self.ys = ys
        self.load_file = load_file
        self.file = file
        if available is None:
            self.available = list()
        if first is None:
            self.first = list()
        if last is None:
            self.last = list()
        if y_first is None:
            self.y_first = list()
        if y_last is None:
            self.y_last = list()
        if first_x is None:
            self.first_x = list()
        if last_x is None:
            self.last_x = list()
        if index is None:
            self.index = list()
        if all_y is None:
            self.all_y = list()
        if confirm is None:
            self.confirm = list()
        if se_first is None:
            self.se_first = list()
        if se_last is None:
            self.se_last = list()
        if se_available is None:
            self.se_available = list()
        return

    def map_bound(self):
        global grid_ant, grid_y, df_bounds
        if self.load_file:
            with open(path[-1] + '/Data/bounds.npy'.format(self.file), 'rb') as bn:
                df_bounds = np.load(bn)

            with open(path[-1] + '/Data/grid.npy'.format(self.file), 'rb') as gd:
                grid = np.load(gd)

            with open(path[-1] + '/Data/available.npy'.format(self.file), 'rb') as av:
                available = np.load(av)

            return df_bounds, grid, available

        else:
            grid, resolution = Map(self.resolution, self.xs, self.ys).black_white()
            bound = True

            f, o = True, False
            for j in range(len(grid[1])):
                for i in range(len(grid)):
                    if grid[i, j] == 1:
                        if bound:
                            self.first.append(i)
                            self.y_first.append(j)
                            u = 4 + int(self.y_first[0])
                            if f:
                                if j > u:
                                    if self.y_first[-1] == self.y_last[-1]:
                                        self.first[-5] = self.first[-2]
                                        self.first.insert(-4, self.first[-1])
                                        self.y_first.insert(-4, self.y_first[-5])
                                        self.first[-4] = self.first[-2]
                                        self.first.insert(-3, self.first[-1])
                                        self.y_first.insert(-3, self.y_first[-4])
                                        self.first[-3] = self.first[-2]
                                        self.first.insert(-2, self.first[-1])
                                        self.y_first.insert(-2, self.y_first[-3])
                                        o = True
                                        f = False
                            bound = False
                        self.available.append([i, j])
                        grid_ant = i
                        grid_y = j
                    else:
                        if not bound:
                            self.last.append(grid_ant)
                            self.y_last.append(grid_y)
                            bound = True
                            if o:
                                self.last[-5] = self.last[-2]
                                self.last.insert(-4, self.last[-1])
                                self.last[-4] = self.last[-2]
                                self.last.insert(-3, self.last[-1])
                                self.last[-3] = self.last[-2]
                                self.last.insert(-2, self.last[-1])
                                o = False

            for i in range(len(self.first)):
                if self.first[i] == self.last[i]:
                    self.confirm.append(True)

            if np.array(self.confirm).all():
                for i in range(len(self.first)):
                    self.first_x.append(self.first[i] + 2)
                    self.last_x.append(self.last[i] - 2)
                    self.all_y.append(self.y_first[i])

                for x in range(2):
                    self.first_x.pop(0), self.last_x.pop(0), self.all_y.pop(0)
                    self.first_x.pop(-1), self.last_x.pop(-1), self.all_y.pop(-1)
                bounds = {'First X': self.first_x, 'Last X': self.last_x, 'Y': self.all_y}
                df_bounds = pd.DataFrame(data=bounds)
            else:
                print('An error occurred. Map bound, y array')

            bp = bounds_path()
            gp = grid_path()
            ap = available_path()

            with open(bp, 'wb') as bn:
                np.save(bn, df_bounds)

            with open(gp, 'wb') as gd:
                np.save(gd, grid)

            with open(ap, 'wb') as av:
                np.save(av, self.available)

            return df_bounds, grid, self.available

    def interest_area(self):
        global grid_ant
        if self.load_file:
            with open(path[-1] + '/Data/secure_grid.npy'.format(self.file), 'rb') as sg:
                secure_grid = np.load(sg)

            with open(path[-1] + '/Data/secure_av.npy'.format(self.file), 'rb') as sa:
                se_available = np.load(sa)
            return secure_grid, se_available
        else:
            df_bounds, grid, available = Bounds(self.resolution, self.xs, self.ys).map_bound()
            secure_grid = np.zeros((self.xs, self.ys))

            for i in range(len(df_bounds)):
                secure_grid[np.array(df_bounds)[i, 0], np.array(df_bounds)[i, 2]] = 1
                secure_grid[np.array(df_bounds)[i, 1], np.array(df_bounds)[i, 2]] = 1

            for j in range(len(secure_grid[1])):
                con = False
                uno = 0
                for i in range(len(secure_grid)):
                    if secure_grid[i, j] == 1:
                        con = True
                        uno += 1
                    if con and uno == 1:
                        secure_grid[i, j] = 1

            bound = True

            for j in range(len(secure_grid[1])):
                for i in range(len(secure_grid)):
                    if secure_grid[i, j] == 1:
                        if bound:
                            self.se_first.append([i, j])
                            bound = False
                        self.se_available.append([i, j])
                        grid_ant = [i, j]
                    else:
                        if not bound:
                            self.se_last.append(grid_ant)
                            bound = True

            sp = secure_path()
            seap = se_available_path()

            with open(sp, 'wb') as sg:
                np.save(sg, secure_grid)

            with open(seap, 'wb') as sa:
                np.save(sa, self.se_available)

            return secure_grid, self.se_available, df_bounds
