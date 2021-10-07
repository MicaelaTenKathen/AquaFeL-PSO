from Benchmark.benchmark_functions import Benchmark_function
from Environment.map import Map
from Environment.plot import Plots
import time
from PSO.pso_function import PSOEnvironment
import numpy as np

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

# Benchmark function

"""
n: number of the ground truth
bench_function: benchmark function values
X_test: coordinates of the points that are analyzed by the benchmark function
secure: map grid of the surface with security limits
df_bounds: data of the limits of the surface where the drone can travel
"""

# Variables initialization

seed = [20]  # , 95, 541, 65, 145, 156, 158, 12, 3, 89, 57, 123, 456, 789, 987, 654, 321, 147, 258, 369, 741, 852,
# 963, 159, 951, 753, 357, 756, 8462, 4875]
c1, c2, c30, c40, c3, c4, length_scale, lam = 3.1286, 2.568, 0, 0, 0.79, 0, 1, 0.1
start_time = time.time()

# PSO initialization

method = 0
pso = PSOEnvironment(resolution, GEN, ys, method)


# Gaussian process initialization


# First iteration of PSO

g = pso.iteration()
out = pso.reset()

# Main part of the code

while g < GEN:
    g_ant = pso.iteration()

    out = pso.ste(c1, c2, c3, c4)

    g = pso.iteration()

    if g == g_ant:
        g += 1

out = out[~np.all(out == 0, axis=1)]
# data = {'Seed': seed, 'GEN': GEN, 'Time': time.time() - start_time, 'MSE_GEN': MSE_data[-1],
#         'Avr_dist': np.mean(distances)}
x_g, y_g, n, X_test, secure, bench_function, grid_min, sigma, mu, MSE_data, it, part_ant = pso.data_out()
plot = Plots(xs, ys, X_test, secure, bench_function, grid_min)
plot.gaussian(x_g, y_g, n, mu, sigma, part_ant)
plot.benchmark()
plot.error(MSE_data, it, GEN)
