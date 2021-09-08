"""
Benchmark functions
author: Federico Peralta
repository: https://github.com/FedePeralta/BO_drones/blob/master/bin/Utils/utils.py#L45
"""

import numpy as np
from deap import benchmarks
from skopt.benchmarks import branin as brn
from Environment.bounds import Bounds


class Benchmark_function():
    def __init__(self, e, grid, resolution, xs, ys, w_ostacles=False, obstacles_on=False, randomize_shekel=False, sensor="",
                 no_maxima=10, load_from_db=False, file=0, a=None, c=None, bench=None):
        if c is None:
            self.c = []
        if a is None:
            self.a = []
        if bench is None:
            self.bench = list()
        self.w_obstacles = w_ostacles
        self.e = e
        self.grid = grid
        self.resolution = resolution
        self.obstacles_on = obstacles_on
        self.randomize_shekel = randomize_shekel
        self.sensor = sensor
        self.no_maxima = no_maxima
        self.load_from_db = load_from_db
        self.file = file
        self.a = a
        self.c = c
        self.xs = xs
        self.ys = ys
        return

    def bohachevsky_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.bohachevsky(sol[:2])[0]

    def ackley_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.ackley(sol[:2])[0]

    def rosenbrock_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.rosenbrock(sol[:2])[0]

    def himmelblau_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.himmelblau(sol[:2])[0]

    def branin(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else brn(sol[:2])

    def shekel_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], self.a, self.c)[0]

    def schwefel_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.schwefel(sol[:2])[0]

    def create_map(self):
        if self.load_from_db:
            if self.sensor == "s1":
                file = 0
            elif self.sensor == "s2":
                file = 1
            elif self.sensor == "s3":
                file = 2
            elif self.sensor == "s4":
                file = 3
            elif self.sensor == "s5":
                file = 4
            elif self.sensor == "s6":
                file = 5
            elif self.sensor == "s7":
                file = 6
            elif self.sensor == "s8":
                file = 7
            with open(self.e, 'rb') as g:
                return np.load(g)
        else:
            self.w_obstacles = self.obstacles_on
            xmin = -5
            xmax = 5
            ymin = 0
            ymax = 10

            if self.randomize_shekel:
                no_maxima = np.random.randint(4, 6)
                xmin = 0
                xmax = 10
                ymin = 0
                ymax = 10

                for i in range(no_maxima):
                    self.a.append([1.2 + np.random.rand() * 8.8, 1.2 + np.random.rand() * 8.8])
                    self.c.append(5)
                self.a = np.array(self.a)
                self.c = np.array(self.c).T
            else:
                self.a = np.array([[0.16, 1 / 1.5], [0.9, 0.2 / 1.5]])
                self.c = np.array([0.15, 0.15]).T

            xadd = 0
            yadd = 0
            gr = self.grid

            _x = np.arange(xmin, xmax, self.resolution * (xmax - xmin) / (gr.shape[1])) + xadd
            _y = np.arange(xmin, xmax, self.resolution * (ymax - ymin) / (gr.shape[0])) + yadd
            _x, _y = np.meshgrid(_x, _y)

            map_created = np.fromiter(map(self.shekel_arg0, zip(_x.flat, _y.flat, gr.flat)), dtype=np.float,
                                      count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)

            meanz = np.nanmean(map_created)
            stdz = np.nanstd(map_created)
            map_created = (map_created - meanz) / stdz

            with open('C:/Users/mcjara/OneDrive - Universidad Loyola '
                      'Andalucía/Documentos/PycharmProjects/EGPPSO_ASV/Data/shww1.npy', 'wb') as g:
                np.save(g, map_created)

            return map_created

    def benchmark_total(self):
        df_bounds_or, grid_or, X_test_or = Bounds(self.resolution, self.xs, self.ys).map_bound()
        _z = Benchmark_function(self.e, self.grid, self.resolution, self.xs, self.ys).create_map()
        for i in range(len(X_test_or)):
            self.bench.append(_z[X_test_or[i][0], X_test_or[i][1]])

        bench_function_or = np.array(self.bench)

        secure_grid, X_test, df_bounds = Bounds(self.resolution, self.xs, self.ys).interest_area()

        return bench_function_or, X_test_or, secure_grid, df_bounds, grid_or
