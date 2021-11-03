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
    def __init__(self, e, grid, resolution, xs, ys, w_ostacles=False, obstacles_on=False, randomize_shekel=False,
                 sensor="",
                 no_maxima=10, load_from_db=False, file=0):
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
        self.xs = xs
        self.ys = ys
        self.a = []
        self.c = []
        self.bench = list()
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

    def create_map(self, n):
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
                no_maxima = np.random.randint(3, 6)
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

            with open('./GroundTruth/shww' + str(n) + '.npy', 'wb') as g:
                np.save(g, map_created)

            df_bounds, X_test_or = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()

            for i in range(len(X_test_or)):
                self.bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

            bench_function_or = np.array(self.bench)  # Return solo esto de benchmark function

            return bench_function_or


class Benchmark_function_reset():
    def __init__(self, grid, resolution, xs, ys, navigation_map, initial_seed, w_ostacles=False, obstacles_on=False,
                 randomize_shekel=True):
        self.w_obstacles = w_ostacles
        self.grid = grid
        self.resolution = resolution
        self.obstacles_on = obstacles_on
        self.randomize_shekel = randomize_shekel
        self.xs = xs
        self.ys = ys
        self.a = []
        self.c = []
        self.bench = list()
        self.seed = initial_seed
        self.navigation_map = navigation_map
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

    def shekel_arg(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], self.a, self.c)[0]

    def schwefel_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.schwefel(sol[:2])[0]

    def create_new_map(self):
        self.seed += 1
        self.w_obstacles = self.obstacles_on
        xmin = -5
        xmax = 5
        ymin = 0
        ymax = 10

        if self.randomize_shekel:
            #no_maxima = np.random.randint(3, 6)
            xmin = 0
            xmax = 10
            ymin = 0
            ymax = 10

            #for i in range(no_maxima):
             #   self.a.append([1.2 + np.random.RandomState(self.seed) * 8.8, 1.2 + np.random.RandomState(self.seed) * 8.8])
              #  self.c.append(5)
            #self.a = np.array(self.a)
            #self.c = np.array(self.c).T
            num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=6)
            print(num_of_peaks)
            self.a = np.random.RandomState(self.seed).random(size=(num_of_peaks, 2))
            print(self.a)
            self.c = np.ones((num_of_peaks)) * 0.05
            print(self.c)
            #self.a = np.array(self.a)
            self.c = np.array(self.c).T
        else:
            self.a = np.array([[0.16, 1 / 1.5], [0.9, 0.2 / 1.5]])
            self.c = np.array([0.15, 0.15]).T

        X = np.linspace(0, 1, self.navigation_map.shape[0])
        Y = np.linspace(0, 1, self.navigation_map.shape[1])
        X, Y = np.meshgrid(X, Y)
        map_created = np.fromiter(map(self.shekel_arg, zip(X.flat, Y.flat)), dtype=np.float,
                        count=X.shape[0] * X.shape[1]).reshape(X.shape)

        meanz = np.nanmean(map_created)
        stdz = np.nanstd(map_created)
        map_created = (map_created - meanz) / stdz

        df_bounds, X_test_or = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()

        for i in range(len(X_test_or)):
            self.bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

        bench_function = np.array(self.bench)  # Return solo esto de benchmark function

        return bench_function


class GroundTruth:

    # TODO: Implementar otras funciones de benchmark.
    # TODO: Corregir el estrechamiento cuando el navigation_map no es cuadrado

    def __init__(self, navigation_map, function_type='shekel', initial_seed=0):

        self.navigation_map = navigation_map

        self.function_type = function_type
        self.seed = initial_seed

        # Randomized parameters of Shekel Function #
        num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=5)
        self.A = np.random.RandomState(self.seed).random(size=(num_of_peaks, 2))
        self.C = np.ones((num_of_peaks)) * 0.05

    def sample_gt(self):

        X = np.linspace(0, 1, self.navigation_map.shape[0])
        Y = np.linspace(0, 1, self.navigation_map.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(self.shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(X.shape)
        Z = (Z - np.mean(Z))/np.std(Z) # Normalize Z
        nan_mask = np.copy(self.navigation_map)
        nan_mask[self.navigation_map == 0] = np.nan
        z_nan = nan_mask + Z.T

        return z_nan

    def reset_gt(self):
        self.seed += 1
        num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=5)
        self.A = np.random.RandomState(self.seed).random(size=(num_of_peaks,2))
        self.C = np.ones((num_of_peaks)) * 0.05 #np.random.RandomState(self.seed).normal(0.05,0.01, size = num_of_peaks)

    def shekel_arg0(self,sol):
        return benchmarks.shekel(sol, self.A, self.C)[0]

    def read_gt_deterministically(self, my_seed):
        num_of_peaks = np.random.RandomState(my_seed).randint(low=1, high=5)
        prev_A = np.copy(self.A)
        prev_C = np.copy(self.C)
        self.A = np.random.RandomState(my_seed).random(size=(num_of_peaks, 2))
        self.C = np.random.RandomState(my_seed).normal(0.1, 0.05, size=num_of_peaks)

        # Sample with the provided seed #
        z_nan = self.sample_gt()

        # Restore previous states
        self.A = prev_A
        self.C = prev_C

        return z_nan