from Benchmark.benchmark_functions import Benchmark_function
from Environment.map import Map
from Environment.plot import Plots
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

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


action = np.array([3.1286, 2.568, 0.79, 0])
start_time = time.time()

# PSO initialization

method = 0
pso = PSOEnvironment(resolution, ys, method, reward_function='inc_mse')


# Gaussian process initialization


# First iteration of PSO
import matplotlib.pyplot as plt

mse_vec = []

for i in range(5):

    done = False
    state = pso.reset()
    R_vec = []

    # Main part of the code

    while not done:

        state, reward, done, dic = pso.step(action)

        R_vec.append(-reward)



    print('Time', time.time() - start_time)

    x_g, y_g, n, X_test, secure, bench_function, grid_min, sigma, mu, MSE_data, it, part_ant = pso.data_out()

    plt.plot(R_vec)

plt.grid()
plt.show()
"""
plot = Plots(xs, ys, X_test, secure, bench_function, grid_min)
plot.gaussian(x_g, y_g, n, mu, sigma, part_ant)
plot.benchmark()
plot.error(MSE_data, it)
"""