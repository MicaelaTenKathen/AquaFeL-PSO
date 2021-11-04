"""
Benchmark functions
author: Federico Peralta
repository: https://github.com/FedePeralta/BO_drones/blob/master/bin/Utils/utils.py#L45
"""

import numpy as np
from deap import benchmarks
from skopt.benchmarks import branin as brn


class Benchmark_function():
    def __init__(self, grid, resolution, xs, ys, initial_seed, w_ostacles=False, obstacles_on=False,
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
            num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=5)
            self.a = np.random.RandomState(self.seed).random(size=(num_of_peaks, 2))
            self.c = np.ones((num_of_peaks)) * 0.05
            #self.a = np.array(self.a)
            self.c = np.array(self.c).T
        else:
            self.a = np.array([[0.16, 1 / 1.5], [0.9, 0.2 / 1.5]])
            self.c = np.array([0.15, 0.15]).T

        X = np.linspace(0, 1, self.grid.shape[1])
        Y = np.linspace(0, 1, self.grid.shape[0])
        X, Y = np.meshgrid(X, Y)
        map_created = np.fromiter(map(self.shekel_arg, zip(X.flat, Y.flat, self.grid.flat)), dtype=np.float,
                        count=X.shape[0] * X.shape[1]).reshape(X.shape)
        meanz = np.nanmean(map_created)
        stdz = np.nanstd(map_created)
        map_created = (map_created - meanz) / stdz

        #df_bounds, X_test_or = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()

        #for i in range(len(X_test_or)):
         #   self.bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

        #bench_function = np.array(self.bench)  # Return solo esto de benchmark function

        return map_created


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

        X = np.linspace(0, 1, self.grid.shape[0])
        Y = np.linspace(0, 1, self.grid.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(self.shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(X.shape)
        Z = (Z - np.mean(Z))/np.std(Z) # Normalize Z
        nan_mask = np.copy(self.grid)
        nan_mask[self.grid == 0] = np.nan
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