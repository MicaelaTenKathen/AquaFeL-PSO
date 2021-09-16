from Benchmark.benchmark_functions import Benchmark_function
from Environment.map import Map
from Environment.plot import Plots
import time
from PSO.pso_function import PSO
from GaussianProcess.gaussianp import Gaussian_Process
import numpy as np
from Data.utils import Utils
from sys import path

# Configuration

"""
resolution: map resolution
xs: size on the x-axis of the map
ys: size on the y-axis of the map
GEN: maximum number of code iterations
"""

resolution = 1
xs = 100
ys = 150
GEN = 6000

# Map

"""
grid_or: map grid of the surface without security limits
grid_min: minimum limit of the map
grid_max: maximum limit of the map
grid_max_x: maximum limit on the x-axis of the map
grid_max_y: maximum limit on the y-axis of the map
"""

grid_or = Map(xs, ys).black_white()
grid_min, grid_max, grid_max_x, grid_max_y = Map(xs, ys).map_values()

# Benchmark function

"""
n: number of the ground truth
bench_function: benchmark function values
X_test: coordinates of the points that are analyzed by the benchmark function
secure: map grid of the surface with security limits
df_bounds: data of the limits of the surface where the drone can travel
"""

n = 1
ben = Benchmark_function(path[-1] + '/GroundTruth/shww' + str(n) + '.npy'.format(0), grid_or, resolution, xs, ys,
                         w_ostacles=False, obstacles_on=False, randomize_shekel=False, sensor="", no_maxima=10,
                         load_from_db=True, file=0)
bench_function, X_test, secure, df_bounds = ben.benchmark_total(n)

# Variables initialization

seed = [20]  # , 95, 541, 65, 145, 156, 158, 12, 3, 89, 57, 123, 456, 789, 987, 654, 321, 147, 258, 369, 741, 852,
# 963, 159, 951, 753, 357, 756, 8462, 4875]
c1, c2, c30, c40, c3, c4, length_scale, lam = 3.1286, 2.568, 0, 0, 0.79, 0, 1, 0.1
n_data, n_plot = 1, 1
g, k, samples, last_sample = 0, 0, 0, 0
sigma_best, mu_best = [0, 0], [0, 0]
part_dist, part_ant, distances, s_ant = np.zeros(8), np.zeros((GEN + 1, 8)), np.zeros(4), np.zeros(4)
benchmark_data, n, sigma_data, mu_data, MSE_data, it, mu_d, x_p, y_p, y_data, part_data, x_g, y_g, y_mult, fitness, x_h, \
y_h, part_array, sigma, mu = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), \
                             list(), list(), list(), list(), list()
ok = False
s_n = np.array([True, True, True, True])
post_array = [length_scale, length_scale, length_scale, length_scale]
start_time = time.time()

# PSO initialization

pso = PSO(GEN, grid_min, grid_max, secure, xs, ys, X_test, bench_function, df_bounds)
toolbox, pop, best, stats, logbook = pso.initPSO(seed)

# Gaussian process initialization

gpr = Gaussian_Process(X_test)

# First iteration of PSO

for part in pop:
    ok, x_h, y_h, fitness, x_p, y_p, y_data, x_bench, y_bench, part, best, n_plot, s_n = pso.pso_fitness(g, s_n, n_data,
                                                                                                         part_ant,
                                                                                                         s_ant,
                                                                                                         part, ok, x_h,
                                                                                                         y_h, fitness,
                                                                                                         x_p, y_p,
                                                                                                         y_data,
                                                                                                         grid_min, x_g,
                                                                                                         y_g, n, n_plot,
                                                                                                         best,
                                                                                                         init=True)
    part_ant, distances = Utils().distance_part(g, n_data, part, part_ant, distances, init=True)

    n_data += 1
    if n_data > 4:
        n_data = 1

for part in pop:
    toolbox.update(g, c1, c2, c3, c4, part, best, sigma_best, mu_best)

dist_ant = 0

for g in range(GEN):
    lam, k, g, s_ant, x_g, y_g, n, last_sample, post_array, samples, MSE_data, it, s_n, n_data, part_ant, \
    ok, x_h, y_h, fitness, x_p, y_p, y_data, n_plot, best, distances, mu_data, sigma_data, sigma, mu, sigma_best, mu_best, dist_ant = pso.step(1, dist_ant, gpr, pso, c1, c2, c3, c4, lam, k,
                                                                                            pop, g, s_ant, x_g, y_g, n,
                                                                                            last_sample, post_array,
                                                                                            samples, MSE_data, it,
                                                                                            toolbox, stats, logbook, s_n, n_data, part_ant, ok, x_h, y_h, fitness, x_p, y_p, y_data,
             n_plot, best, distances, mu_data, sigma_data, sigma, mu, sigma_best, mu_best)

    # for part in pop:
    #
    #     ok, x_h, y_h, fitness, x_p, y_p, y_data, x_bench, y_bench, part, best, n_plot, s_n = pso.pso_fitness(g, s_n,
    #                                                                                                          n_data,
    #                                                                                                          part_ant,
    #                                                                                                          s_ant,
    #                                                                                                          part, ok,
    #                                                                                                          x_h,
    #                                                                                                          y_h,
    #                                                                                                          fitness,
    #                                                                                                          x_p, y_p,
    #                                                                                                          y_data,
    #                                                                                                          grid_min,
    #                                                                                                          x_g,
    #                                                                                                          y_g, n,
    #                                                                                                          n_plot,
    #                                                                                                          best,
    #                                                                                                          init=False)
    #
    #     part_ant, distances = Utils().distance_part(g, n_data, part, part_ant, distances, init=False)
    #
    #     n_data += 1
    #     if n_data > 4:
    #         n_data = 1
    #
    # if (np.mean(distances) - last_sample) >= (np.min(post_array) * lam):
    #     c3 = c3
    #     c4 = c4
    #     k += 1
    #     ok = True
    #     last_sample = np.mean(distances)
    #
    #     for part in pop:
    #
    #         ok, x_h, y_h, fitness, x_p, y_p, y_data, x_bench, y_bench, part, best, n_plot, s_n = pso.pso_fitness(g, s_n,
    #                                                                                                              n_data,
    #                                                                                                              part_ant,
    #                                                                                                              s_ant,
    #                                                                                                              part,
    #                                                                                                              ok,
    #                                                                                                              x_h,
    #                                                                                                              y_h,
    #                                                                                                              fitness,
    #                                                                                                              x_p,
    #                                                                                                              y_p,
    #                                                                                                              y_data,
    #                                                                                                              grid_min,
    #                                                                                                              x_g,
    #                                                                                                              y_g, n,
    #                                                                                                              n_plot,
    #                                                                                                              best,
    #                                                                                                              init=False)
    #
    #         sigma, mu, sigma_data, mu_data, post_array = gpr.gp_regression(x_bench, y_bench, x_h, y_h, fitness,
    #                                                                        post_array, n_data, mu_data, sigma_data)
    #
    #         samples += 1
    #
    #         n_data += 1
    #         if n_data > 4:
    #             n_data = 1
    #
    #     MSE_data, it = Utils().mse(g, fitness, mu_data, samples, MSE_data, it)
    #
    #     sigma_best, mu_best = gpr.sigma_max(sigma, mu)
    #     ok = False
    #
    # for part in pop:
    #     toolbox.update(g, c1, c2, c3, c4, part, best, sigma_best, mu_best)
    #
    # logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
    # print(logbook.stream)
    # mean_dist = np.mean(np.array(distances))
    # print(mean_dist)

data = {'Seed': seed, 'GEN': GEN, 'Time': time.time() - start_time, 'MSE_GEN': MSE_data[-1],
        'Avr_dist': np.mean(distances)}
plot = Plots(xs, ys, X_test, secure, bench_function, grid_min)
plot.gaussian(x_g, y_g, n, mu, sigma, part_ant)
plot.benchmark()
plot.error(MSE_data, it, GEN)
