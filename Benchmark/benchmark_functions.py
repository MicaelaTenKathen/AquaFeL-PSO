"""
Benchmark functions
author: Federico Peralta
repository: https://github.com/FedePeralta/BO_drones/blob/master/bin/Utils/utils.py#L45
"""

import random

from deap import benchmarks
from skopt.benchmarks import branin as brn

from Environment.bounds import Bounds
from Environment.peaks_zones import ZonesPeaks


class Benchmark_function():
    def __init__(self, grid, resolution, xs, ys, X_test, initial_seed, vehicles, w_ostacles=False, obstacles_on=False,
                 randomize_shekel=True, base_benchmark="shekel"):
        self.w_obstacles = w_ostacles
        self.vehicles = vehicles
        self.grid = grid
        self.X_test = X_test
        self.resolution = resolution
        self.obstacles_on = obstacles_on
        self.randomize_shekel = randomize_shekel
        self.xs = xs
        self.ys = ys
        self.a = list()
        self.a1 = list()
        self.bench = list()
        self.seed = initial_seed
        if base_benchmark == "shekel":
            self.yukyry, self.pirayu, self.sanber, self.aregua = ZonesPeaks(self.X_test).find_zones()
        self.base_benchmark = base_benchmark
        return

    def ackley_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.ackley(sol[:2])[0]

    def bohachevsky_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.bohachevsky(sol[:2])[0]

    def griewank_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.griewank(sol[:2])[0]

    def h1_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.h1(sol[:2])[0]

    def himmelblau_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.himmelblau(sol[:2])[0]

    def rastrigin_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.rastrigin(sol[:2])[0]

    def rosenbrock_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.rosenbrock(sol[:2])[0]

    def schaffer_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.schaffer(sol[:2])[0]

    def schwefel_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.schwefel(sol[:2])[0]

    def shekel_arg(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], self.a, self.c)[0]

    def branin(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else brn(sol[:2])

    def create_new_map(self):
        self.w_obstacles = self.obstacles_on

        if self.randomize_shekel:

            if self.vehicles == 2:
                num_of_peaks = 2
            else:
                num_of_peaks = np.random.RandomState(self.seed).randint(low=2, high=self.vehicles)

            self.c = np.random.RandomState(self.seed).rand(num_of_peaks, 1) * 400 + 120

            index_a1 = np.random.RandomState(self.seed).random(size=(num_of_peaks, 1))
            index_a = list()

            random.seed(self.seed)
            if self.vehicles <= 4:
                zone = random.sample(range(4), num_of_peaks)
            else:
                zone = list()
                for i in range(num_of_peaks):
                    zone.append(random.randint(0, 3))
            for i in range(len(zone)):
                if zone[i] == 0:
                    id1 = index_a1[i] * len(self.yukyry) - 1
                    id2 = self.yukyry[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
                elif zone[i] == 1:
                    id1 = index_a1[i] * len(self.pirayu) - 1
                    id2 = self.pirayu[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
                elif zone[i] == 2:
                    id1 = index_a1[i] * len(self.sanber) - 1
                    id2 = self.sanber[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
                elif zone[i] == 3:
                    id1 = index_a1[i] * len(self.aregua) - 1
                    id2 = self.aregua[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
            self.a = np.array(self.a)
            index_a = np.array(index_a)

        # Como los distintos benchmarks(BM) tienen distintos bounds, vamos a cambiar X1 e Y1 de acuerdo al benchmark
        # seleccionado

        if self.base_benchmark == "ackley":
            stepx, stepy = 60 / self.grid.shape[1], 60 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -30 + stepx / 2, 30, -30 + stepy / 2, 30
            bm_func = self.ackley_arg0
        elif self.base_benchmark == "bohachevsky":
            stepx, stepy = 30 / self.grid.shape[1], 30 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -15 + stepx / 2, 15, -15 + stepy / 2, 15
            bm_func = self.bohachevsky_arg0
        elif self.base_benchmark == "griewank":
            stepx, stepy = 100 / self.grid.shape[1], 100 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -50 + stepx / 2, 50, -50 + stepy / 2, 50
            bm_func = self.griewank_arg0
        elif self.base_benchmark == "h1":
            stepx, stepy = 50 / self.grid.shape[1], 50 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -25 + stepx / 2, 25, -25 + stepy / 2, 25
            bm_func = self.h1_arg0
        elif self.base_benchmark == "himmelblau":
            stepx, stepy = 12 / self.grid.shape[1], 12 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -6 + stepx / 2, 6, -6 + stepy / 2, 6
            bm_func = self.himmelblau_arg0
        elif self.base_benchmark == "rastrigin":
            stepx, stepy = 10 / self.grid.shape[1], 10 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -5 + stepx / 2, 5, -5 + stepy / 2, 5
            bm_func = self.rastrigin_arg0
        elif self.base_benchmark == "rosenbrock":
            stepx, stepy = 4 / self.grid.shape[1], 4 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -2 + stepx / 2, 2, -1 + stepy / 2, 3
            bm_func = self.rosenbrock_arg0
        elif self.base_benchmark == "schaffer":
            stepx, stepy = 40 / self.grid.shape[1], 40 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -20 + stepx / 2, 20, -20 + stepy / 2, 20
            bm_func = self.schaffer_arg0
        elif self.base_benchmark == "schwefel":
            stepx, stepy = 800 / self.grid.shape[1], 800 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -400 + stepx / 2, 400, -400 + stepy / 2, 400
            bm_func = self.schwefel_arg0
        else:
            xmin, xmax, ymin, ymax = 0, self.grid.shape[1], 0, self.grid.shape[0]
            stepx, stepy = 1, 1
            bm_func = self.shekel_arg

        X1 = np.arange(xmin, xmax, stepx)
        Y1 = np.arange(ymin, ymax, stepy)
        X1, Y1 = np.meshgrid(X1, Y1)
        print('Creating', self.base_benchmark, stepx, stepy)

        map_created1 = np.fromiter(map(bm_func, zip(X1.flat, Y1.flat)), dtype=float,
                                   count=X1.shape[0] * X1.shape[1]).reshape(X1.shape)

        map_max = np.max(map_created1)
        map_min = np.min(map_created1)
        map_created = list(map(lambda x: (x - map_min) / (map_max - map_min), map_created1))
        map_created = np.array(map_created)

        bench = list()
        df_bounds_or, X_test_or, bench_list = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()
        for i in range(len(X_test_or)):
            bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

        bench_function_or = np.array(bench)
        return map_created, bench_function_or, None, None


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
        Z = np.fromiter(map(self.shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float,
                        count=X.shape[0] * X.shape[1]).reshape(X.shape)
        Z = (Z - np.mean(Z)) / np.std(Z)  # Normalize Z
        nan_mask = np.copy(self.grid)
        nan_mask[self.grid == 0] = np.nan
        z_nan = nan_mask + Z.T

        return z_nan

    def reset_gt(self):
        self.seed += 1
        num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=5)
        self.A = np.random.RandomState(self.seed).random(size=(num_of_peaks, 2))
        self.C = np.ones(
            (num_of_peaks)) * 0.05  # np.random.RandomState(self.seed).normal(0.05,0.01, size = num_of_peaks)

    def shekel_arg0(self, sol):
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


if __name__ == '__main__':
    import matplotlib.image as img
    import numpy as np
    import matplotlib.pyplot as plt

    my_benchmarks = [
        "ackley",
        "bohachevsky",
        "griewank",
        "h1",
        "himmelblau",
        "rastrigin",
        "rosenbrock",
        "schaffer",
        "schwefel",
        # "shekel"
    ]

    xs = 871
    ys = 600
    grid = np.flipud(img.imread("../Image/snazzy-image-prueba.png")[:, :, 0])

    _, X_test_or, _ = Bounds(1, 600, 871, load_file=False).map_bound()
    for idx, benchmark in enumerate(my_benchmarks):
        m_map, bFunction, no_peaks, index_a = Benchmark_function(grid, 1, xs, ys, None, 42, 0, base_benchmark=benchmark,
                                                                 randomize_shekel=False).create_new_map()
        m_nan_map = np.full_like(m_map, np.nan)
        for point in X_test_or:
            m_nan_map[point[1], point[0]] = m_map[point[1], point[0]]

        plt.subplot(331 + idx)
        plt.title(benchmark)
        plt.imshow(m_nan_map)
        # plt.colorbar()
    plt.show()
