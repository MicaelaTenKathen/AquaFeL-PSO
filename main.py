from Benchmark.benchmark_functions import Benchmark_function
from Environment.map import Map
from Environment.plot import Plots
import time
from PSO.pso_function import PSO_fun
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
n_data = 0
g = 0
benchmark_data, sigma, mu = list(), list(), list()
distances = np.zeros(4)
part_ant = np.zeros((GEN, 8))
post_array = [length_scale, length_scale, length_scale, length_scale]
start_time = time.time()

# PSO initialization

pso = PSO_fun(n_data, GEN, grid_min, grid_max, secure, xs, ys, X_test, bench_function, df_bounds)
best, pop, toolbox, stats, logbook = pso.initPSO(seed, n_data)
util = Utils()
method = 0

# Gaussian process initialization

gpr = Gaussian_Process(X_test)

# First iteration of PSO

n_data += 1

out, sigma_best, mu_best, post_array, last_sample, MSE_data, it, g, k, samples = pso.initcode(pop, pso, util, gpr,
                                                                                              toolbox,
                                                                                              g, c1, c2, c3, c4, lam,
                                                                                              best, post_array, method,
                                                                                              part_ant, distances)
ok = False
n_data = 1
f = 0


# Main part of the code

while g < GEN:
    g_ant = g

    out, sigma_best, mu_best, post_array, last_sample, MSE_data, it, g, k, f, samples, mu, sigma = pso.step(ok, pop, pso, util,
                                                                                                 gpr, toolbox, out, g,
                                                                                                 c1, c2, c3, c4, lam,
                                                                                                 best, post_array,
                                                                                                 last_sample, method,
                                                                                                 sigma_best, mu_best,
                                                                                                 part_ant, distances, k,
                                                                                                 f, samples, MSE_data,
                                                                                                 it)

    if g == g_ant:
        g += 1

    logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
    print(logbook.stream)
    mean_dist = np.mean(np.array(distances))
    print(mean_dist)


data = {'Seed': seed, 'GEN': GEN, 'Time': time.time() - start_time, 'MSE_GEN': MSE_data[-1],
        'Avr_dist': np.mean(distances)}
plot = Plots(xs, ys, X_test, secure, bench_function, grid_min)
x_g, y_g, n = pso.data_out()
plot.gaussian(x_g, y_g, n, mu, sigma, part_ant)
plot.benchmark()
plot.error(MSE_data, it, GEN)
