from Data.limits import Limits
from Environment.plot import Plots
from Environment.map import Map
from Benchmark.benchmark_functions import Benchmark_function
from Environment.bounds import Bounds
from Data.utils import Utils

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

import numpy as np
import random
import math
import gym

from deap import base
from deap import creator
from deap import tools

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""[https://deap.readthedocs.io/en/master/examples/pso_basic.html]"""


def createPart():
    """
    Creation of the objects "FitnessMax" and "Particle"
    """

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, smin=None, smax=None,
                   best=None)
    creator.create("BestGP", np.ndarray, fitness=creator.FitnessMax)


class PSOEnvironment(gym.Env):

    def __init__(self, resolution, ys, method, initial_seed, initial_position, reward_function='mse',
                 behavioral_method=0):
        self.f = None
        self.k = None
        self.population = 4
        self.resolution = resolution
        self.smin = 0
        self.smax = 3
        self.size = 2
        self.wmin = 0.4 / (15000 / ys)
        self.wmax = 0.9 / (15000 / ys)
        self.xs = int(10000 / (15000 / ys))
        self.ys = ys
        ker = RBF(length_scale=10, length_scale_bounds=(1e-1, 10))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1e-6)  # optimizer=None)
        self.x_h = []
        self.y_h = []
        self.x_p = []
        self.y_p = []
        self.fitness = []
        self.y_data = []
        self.mu_data = []
        self.sigma_data = []
        self.x_bench = None
        self.y_bench = None
        self.n_plot = float(1)
        self.s_n = np.array([True, True, True, True])
        self.s_ant = np.zeros(4)
        self.samples = None
        self.dist_ant = None
        self.sigma_best = []
        self.mu_best = []
        self.n_data = 1
        self.num = 0
        self.seed = initial_seed
        self.mu = []
        self.sigma = []
        self.g = 0
        self.post_array = np.array([1, 1, 1, 1])
        self.distances = np.zeros(4)
        self.lam = 0.3
        self.part_ant = np.zeros((1, 8))
        self.last_sample, self.k, self.f, self.samples, self.ok = 0, 0, 0, 0, False
        self.MSE_data = []
        self.MSE_data1 = []
        self.it = []
        self.method = method
        self.mse = []
        self.bench_array = []
        self.duplicate = False
        self.array_part = np.zeros((1, 8))
        self.reward_function = reward_function
        self.behavioral_method = behavioral_method
        self.initial_position = initial_position
        if self.method == 0:
            self.state = np.zeros(22, )
        else:
            self.state = np.zeros((6, self.xs, self.ys))

        self.grid_or = Map(self.xs, ys).black_white()

        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = 0, self.ys, self.xs, self.ys

        self.p = 1

        self.df_bounds, self.X_test = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()
        self.secure, self.df_bounds = Bounds(self.resolution, self.xs, self.ys).interest_area()
        # self.secure = navigation_map
        # print(self.secure)

        self.bench_function = None

        self.plot = Plots(self.xs, self.ys, self.X_test, self.secure, self.bench_function, self.grid_min)

        # self.action_space = gym.spaces.Box(low=0.0, high=4.0, shape=(4,))
        # self.state_space = gym.spaces.Box()

        self.util = Utils()

        createPart()

    def generatePart(self):

        """
        Generates a random position and a random speed for the particles (drones).
        """
        part = creator.Particle([self.initial_position[self.p, i] for i in range(self.size)])
        part.speed = np.array([random.uniform(self.smin, self.smax) for _ in range(self.size)])
        part.smin = self.smin
        part.smax = self.smax
        self.p += 1

        return part

    def updateParticle_n(self, c1, c2, c3, c4, part):

        """
        Calculates the speed and the position of the particles (drones).
        """

        if self.behavioral_method == 0:
            u1 = np.array([random.uniform(0, c1) for _ in range(len(part))])
            u2 = np.array([random.uniform(0, c2) for _ in range(len(part))])
            u3 = np.array([random.uniform(0, c3) for _ in range(len(part))])
            u4 = np.array([random.uniform(0, c4) for _ in range(len(part))])
        else:
            u1 = c1
            u2 = c2
            u3 = c3
            u4 = c4

        v_u1 = u1 * (part.best - part)
        v_u2 = u2 * (self.best - part)
        v_u3 = u3 * (self.sigma_best - part)
        v_u4 = u4 * (self.mu_best - part)
        w = 1
        part.speed = v_u1 + v_u2 + v_u3 + v_u4 + part.speed * w
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = part + part.speed

        return part

    def tool(self):

        """
        The operators are registered in the toolbox with their parameters.
        """
        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.generatePart)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle_n)

        return self.toolbox

    def swarm(self):

        """
        Creates a population.
        """
        toolbox = self.tool()
        self.pop = toolbox.population(n=self.population)
        self.best = self.pop[0]

        return self.best, self.pop

    def statistic(self):

        """
        Visualizes the stats of the code.
        """

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "evals"] + self.stats.fields

        return self.stats, self.logbook

    def reset(self):

        """
        Initialization of the pso.
        """
        self.reset_variables()
        self.bench_function, self.bench_array = Benchmark_function(self.grid_or, self.resolution, self.xs, self.ys,
                                                                   self.seed).create_new_map()
        self.generatePart()
        self.tool()
        random.seed(self.seed)
        self.swarm()
        self.statistic()
        self.state = self.first_values()
        return self.state

    def reset_variables(self):
        self.f = None
        self.k = None
        self.x_h = []
        self.y_h = []
        self.x_p = []
        self.y_p = []
        self.fitness = []
        self.y_data = []
        self.mu_data = []
        self.sigma_data = []
        self.x_bench = None
        self.y_bench = None
        self.n_plot = float(1)
        self.s_n = np.array([True, True, True, True])
        self.s_ant = np.zeros(4)
        self.samples = None
        self.dist_ant = None
        self.sigma_best = []
        self.mu_best = []
        self.n_data = 1
        self.mu = []
        self.p = 0
        self.sigma = []
        self.post_array = np.array([1, 1, 1, 1])
        self.distances = np.zeros(4)
        self.part_ant = np.zeros((1, 8))
        self.last_sample, self.k, self.f, self.samples, self.ok = 0, 0, 0, 0, False
        self.MSE_data = []
        self.it = []
        self.mse = []
        self.duplicate = False
        self.array_part = np.zeros((1, 8))
        self.seed += 1

        if self.method == 0:
            self.state = np.zeros(22, )
        else:
            self.state = np.zeros((6, self.xs, self.ys))

        self.num += 1
        self.g = 0
        self.distances = np.zeros(4)

    def pso_fitness(self, part):

        """
        Obtains the local best (part.best) of each particle (drone) and the global best (best) of the swarm (fleet).
        """

        part.fitness.values = self.new_fitness(part)

        if self.ok:
            self.check_duplicate(part)
        else:
            if part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if self.best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        return self.ok, part

    def new_fitness(self, part):
        part, self.s_n = Limits(self.secure, self.xs, self.ys).new_limit(self.g, part, self.s_n, self.n_data,
                                                                         self.s_ant, self.part_ant)
        self.x_bench = int(part[0])
        self.y_bench = int(part[1])

        new_fitness_value = [self.bench_function[self.x_bench][self.y_bench]]
        return new_fitness_value

    def gp_regression(self):

        """
        Fits the gaussian process.
        """

        x_a = np.array(self.x_h).reshape(-1, 1)
        y_a = np.array(self.y_h).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(self.fitness).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        self.mu, self.sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = np.min(np.exp(self.gpr.kernel_.theta[0]))
        r = self.n_data - 1
        self.post_array[r] = post_ls

        if not self.duplicate:
            for i in range(len(self.X_test)):
                di = self.X_test[i]
                dix = di[0]
                diy = di[1]
                if dix == self.x_bench and diy == self.y_bench:
                    self.mu_data.append(self.mu[i])
                    self.sigma_data.append(self.sigma[i])

        return self.post_array

    def sigma_max(self):

        """
        Returns the coordinates of the maximum uncertainty (sigma_best) and the maximum contamination (mu_best).
        """

        sigma_max = np.max(self.sigma)
        index_sigma = np.where(self.sigma == sigma_max)
        index_x1 = index_sigma[0]
        index_x2 = index_x1[0]
        index_x = int(self.X_test[index_x2][0])
        index_y = int(self.X_test[index_x2][1])

        mu_max = np.max(self.mu)
        index_mu = np.where(self.mu == mu_max)
        index_x1mu = index_mu[0]
        index_x2mu = index_x1mu[0]
        index_xmu = int(self.X_test[index_x2mu][0])
        index_ymu = int(self.X_test[index_x2mu][1])

        best_1 = [index_x, index_y]
        self.sigma_best = np.array(best_1)

        best_2 = [index_xmu, index_ymu]
        self.mu_best = np.array(best_2)

        return self.sigma_best, self.mu_best

    def calculate_reward(self):
        if self.reward_function == 'mse':
            reward = -self.MSE_data[-1]
        elif self.reward_function == 'inc_mse':
            reward = self.MSE_data[-2] - self.MSE_data[-1]
        return reward

    def check_duplicate(self, part):
        self.duplicate = False
        for i in range(len(self.x_h)):
            if self.x_h[i] == self.x_bench and self.y_h[i] == self.y_bench:
                self.duplicate = True
                break
            else:
                self.duplicate = False
        if self.duplicate:
            pass
        else:
            self.x_h.append(int(part[0]))
            self.y_h.append(int(part[1]))
            self.fitness.append(part.fitness.values)

    def first_values(self):

        """
        The output "out" of the method "initcode" is the positions of the particles (drones) after the first update of the
        gaussian process (initial state).
        method = 0 -> out = scalar vector
        out = [px_1, py_1, px_2, py_2, px_3, py_3, px_4, py_4, lbx_1, lby_1, lbx_2, lby_2, lbx_3, lby_3, lbx_4, lby_4, gbx, gby,
               sbx, sgy, mbx, mby]
               where:
               px: x coordinate of the drone position
               py: y coordinate of the drone position
               lbx: x coordinate of the local best
               lby: y coordinate of the local best
               gbx: x coordinate of the global best
               gby: y coordinate of the global best
               sbx: x coordinate of the sigma best (maximum uncertainty)
               sby: y coordinate of the sigma best (maximum uncertainty)
               mbx: x coordinate of the mean best (maximum contamination)
               mby: y coordinate of the mean best (maximum contamination)
        method = 1 -> out = images
        :param c1: weight that determinate the importance of the local best component
        :param c2: weight that determinate the importance of the global best component
        :param c3: weight that determinate the importance of the maximum uncertainty component
        :param c4: weight that determinate the importance of the maximum contamination component
        :param lam: ratio of one of the different length scales [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        :param post_array: refers to the posterior length scale of the surrogate model [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        """

        for part in self.pop:

            part.fitness.values = self.new_fitness(part)

            if self.n_plot > 4:
                self.n_plot = float(1)

            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values

            if self.best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                    self.distances, self.array_part, dfirst=True)

            self.check_duplicate(part)

            self.post_array = self.gp_regression()

            self.samples += 1

            self.n_data += 1
            if self.n_data > 4:
                self.n_data = 1

        # self.MSE_data1, self.it = self.util.mse(self.g, self.bench_array, self.mu)
        self.mse = mean_squared_error(y_true=self.bench_array, y_pred=self.mu)
        self.MSE_data.append(self.mse)
        self.it.append(self.g)

        self.sigma_best, self.mu_best = self.sigma_max()

        self.return_state()

        self.k = 4
        self.ok = False

        return self.state

    def step(self, action):

        """
        The output "out" of the method "step" is the positions of the particles (drones) after traveling 1000 m
        (scaled).

        method = 0 -> out = scalar vector
        out = [px_1, py_1, px_2, py_2, px_3, py_3, px_4, py_4, lbx_1, lby_1, lbx_2, lby_2, lbx_3, lby_3, lbx_4, lby_4,
               gbx, gby, sbx, sgy, mbx, mby]
               where:
               px: x coordinate of the drone position
               py: y coordinate of the drone position
               lbx: x coordinate of the local best
               lby: y coordinate of the local best
               gbx: x coordinate of the global best
               gby: y coordinate of the global best
               sbx: x coordinate of the sigma best (maximum uncertainty)
               sby: y coordinate of the sigma best (maximum uncertainty)
               mbx: x coordinate of the mean best (maximum contamination)
               mby: y coordinate of the mean best (maximum contamination)

        method = 1 -> out = images

        :param c1: weight that determinate the importance of the local best component
        :param c2: weight that determinate the importance of the global best component
        :param c3: weight that determinate the importance of the maximum uncertainty component
        :param c4: weight that determinate the importance of the maximum contamination component
        :param lam: ratio of one of the different length scales [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        :param post_array: refers to the posterior length scale of the surrogate model [Equation 7
        (https://doi.org/10.3390/electronics10131605)]
        """
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.n_data = 1
        self.f += 1

        while dis_steps < 10:

            for part in self.pop:
                self.toolbox.update(action[0], action[1], action[2], action[3], part)

            for part in self.pop:
                self.ok, part = self.pso_fitness(part)
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > 4:
                    self.n_data = 1

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.k += 1
                self.ok = True
                self.last_sample = np.mean(self.distances)

                for part in self.pop:
                    self.ok, part = self.pso_fitness(part)

                    self.post_array = self.gp_regression()

                    self.samples += 1

                    self.n_data += 1
                    if self.n_data > 4:
                        self.n_data = 1

                # self.MSE_data1, self.it = self.util.mse(self.g, self.bench_array, self.mu)
                self.mse = mean_squared_error(y_true=self.bench_array, y_pred=self.mu)
                self.MSE_data.append(self.mse)
                self.it.append(self.g)

                self.sigma_best, self.mu_best = self.sigma_max()

                self.ok = False

            dis_steps = np.mean(self.distances) - dist_ant

            self.g += 1

        self.return_state()

        reward = self.calculate_reward()

        self.logbook.record(gen=self.g, evals=len(self.pop), **self.stats.compile(self.pop))
        # print(self.logbook.stream)
        if ((self.distances) >= 150).any():
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def return_state(self):
        z = 0
        for part in self.pop:
            self.state = self.state
            if self.method == 0:
                self.state[z] = part[0]
                z += 1
                self.state[z] = part[1]
                z += 1
                self.state[z + 6] = part.best[0]
                self.state[z + 7] = part.best[1]
                if self.n_data == 4:
                    self.state[16] = self.best[0]
                    self.state[17] = self.best[1]
                    self.state[18] = self.sigma_best[0]
                    self.state[19] = self.sigma_best[1]
                    self.state[20] = self.mu_best[0]
                    self.state[21] = self.mu_best[1]
            else:
                posx = 2 * z
                posy = (2 * z) + 1
                self.state = self.plot.part_position(self.part_ant[:, posx], self.part_ant[:, posy], self.state, z)
                z += 1
                if self.n_data == 4:
                    self.state = self.plot.state_sigma_mu(self.mu, self.sigma, self.state)
            self.n_data += 1
            if self.n_data > 4:
                self.n_data = 1

    def data_out(self):

        """
        Return the first and the last position of the particles (drones).
        """

        return self.X_test, self.secure, self.bench_function, self.grid_min, self.sigma, \
               self.mu, self.MSE_data, self.it, self.part_ant, self.bench_array

    def MSE_value(self):
        return self.MSE_data

    def distances_data(self):
        return self.distances
