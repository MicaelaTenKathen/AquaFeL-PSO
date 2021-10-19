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

"""[https://deap.readthedocs.io/en/master/examples/pso_basic.html]"""


class PSOEnvironment(gym.Env):

    def __init__(self, resolution, ys, method, reward_function='mse', initial_seed=1000):
        self.f = int()
        self.k = int()
        self.population = 4
        self.resolution = resolution
        self.smin = 0
        self.smax = 2 / (15000 / ys)
        self.size = 2
        self.wmin = 0.4 / (15000 / ys)
        self.wmax = 0.9 / (15000 / ys)
        self.xs = int(10000 / (15000 / ys))
        self.ys = ys
        ker = RBF(length_scale=10, length_scale_bounds=(1e-1, 10))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1 ** 2)
        self.x_h = list()
        self.y_h = list()
        self.x_p = list()
        self.y_p = list()
        self.fitness = list()
        self.y_data = list()
        self.mu_data = list()
        self.sigma_data = list()
        self.x_bench = int()
        self.y_bench = int()
        self.n_plot = float(1)
        self.s_n = np.array([True, True, True, True])
        self.s_ant = np.zeros(4)
        self.x_g = list()
        self.y_g = list()
        self.ngp = list()
        self.n = list()
        self.samples = int()
        self.dist_ant = float()
        self.sigma_best = [0, 0]
        self.mu_best = [0, 0]
        self.n_data = 1
        self.num = 0
        self.initial_seed = initial_seed
        self.seed = self.initial_seed
        self.mu = []
        self.sigma = []
        self.g = 0
        self.post_array = np.array([1, 1, 1, 1])
        self.distances = np.zeros(4)
        self.lam = 0.6
        self.part_ant = np.zeros((1, 8))
        self.last_sample, self.k, self.f, self.samples, self.ok = 0, 0, 0, 0, False
        self.MSE_data = list()
        self.it = list()
        self.method = method
        self.mse = float()
        self.duplicate = False
        self.array_part = np.zeros((1, 8))
        self.reward_function = reward_function

        if self.method == 0:
            self.state = np.zeros(22,)
        else:
            self.state = np.zeros((6, self.xs, self.ys))

        self.grid_or = Map(self.xs, ys).black_white()
        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = Map(self.xs, ys).map_values()
        self.pmin = self.grid_min
        self.pmax = self.grid_max

        self.df_bounds, self.X_test = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()
        self.secure, self.df_bounds = Bounds(self.resolution, self.xs, self.ys).interest_area()

        self.bench_function = []

        self.plot = Plots(self.xs, self.ys, self.X_test, self.secure, self.bench_function, self.grid_min)

        # self.action_space = gym.spaces.Box(low=0.0, high=4.0, shape=(4,))
        # self.state_space = gym.spaces.Box()

        self.util = Utils()

    def createPart(self):

        """
        Creation of the objects "FitnessMax" and "Particle"
        """

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, smin=None, smax=None,
                       best=None)
        creator.create("BestGP", np.ndarray, fitness=creator.FitnessMax)

    def generatePart(self):

        """
        Generates a random position and a random speed for the particles (drones).
        """

        part = creator.Particle([random.uniform(self.pmin, self.pmax) for _ in range(self.size)])
        part.speed = np.array([random.uniform(self.smin, self.smax) for _ in range(self.size)])
        part.smin = self.smin
        part.smax = self.smax

        return part

    def updateParticle_n(self, c1, c2, c3, c4, part):

        """
        Calculates the speed and the position of the particles (drones).
        """

        u1 = np.array([random.uniform(0, c1) for _ in range(len(part))])
        u2 = np.array([random.uniform(0, c2) for _ in range(len(part))])
        u3 = np.array([random.uniform(0, c3) for _ in range(len(part))])
        u4 = np.array([random.uniform(0, c4) for _ in range(len(part))])
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
        self.num += 1
        self.bench_function = Benchmark_function('./GroundTruth/shww' + str(self.num) + '.npy'.format(0), self.grid_or,
                                                 self.resolution, self.xs, self.ys,
                                                 w_ostacles=False, obstacles_on=False, randomize_shekel=True,
                                                 sensor="", no_maxima=10,
                                                 load_from_db=False, file=0).create_map(self.num)
        self.createPart()
        self.generatePart()
        self.tool()
        self.seed = [20]
        # np.random.seed(self.seed)
        random.seed(self.seed[0])
        self.swarm()
        self.statistic()
        action = [3.1286, 2.568, 0.79, 0]
        self.g = 0
        self.distances = np.zeros(4)
        self.initcode(action)
        return self.state

    def pso_fitness(self, part, first=False):

        """
        Obtains the local best (part.best) of each particle (drone) and the global best (best) of the swarm (fleet).
        """

        part, self.s_n = Limits(self.secure, self.xs, self.ys).new_limit(self.g, part, self.s_n, self.n_data, self.s_ant, self.part_ant)
        self.x_bench = int(part[0])
        self.y_bench = int(part[1])

        for i in range(len(self.X_test)):
            if self.X_test[i][0] == self.x_bench and self.X_test[i][1] == self.y_bench:
                part.fitness.values = [self.bench_function[i]]
                break
        if self.ok:
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
                self.ngp.append(self.n_data)
                self.fitness.append(part.fitness.values)
        else:
            self.x_p.append(part[0])
            self.y_p.append(part[1])
            self.y_data.append(part.fitness.values)
            if first:
                x_gap = int(part[0]) + abs(self.grid_min)
                y_gap = int(part[1]) + abs(self.grid_min)
                self.x_g.append(x_gap)
                self.y_g.append(y_gap)
                self.n.append(self.n_data)
                if self.n_plot > 4:
                    self.n_plot = float(1)
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            else:
                if np.mean(self.distances) >= 249:
                    x_gap = int(part[0]) + abs(self.grid_min)
                    y_gap = int(part[1]) + abs(self.grid_min)
                    self.x_g.append(x_gap)
                    self.y_g.append(y_gap)
                    self.n.append(self.n_data)
                    self.n_plot += float(1)
                    if self.n_plot > 4:
                        self.n_plot = float(1)
                if part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
            if self.best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        return self.ok, part

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
        self.mse = mean_squared_error(y_true=self.fitness, y_pred=self.mu_data)
        if self.reward_function == 'mse':
            reward = -self.MSE_data[-1]
        elif self.reward_function == 'inc_mse':
            reward = self.MSE_data[-2] - self.MSE_data[-1]
        return reward

    def initcode(self, action):

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
            self.ok, part = self.pso_fitness(part, first=True)

            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant, self.distances, self.array_part, dfirst=True)

            self.n_data += 1
            if self.n_data > 4:
                self.n_data = 1

        for part in self.pop:
            self.toolbox.update(action[0], action[1], action[2], action[3], part)

        while self.k == 0:
            for part in self.pop:

                self.ok, part = self.pso_fitness(part, first=False)

                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant, self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > 4:
                    self.n_data = 1

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.ok = True
                self.last_sample = np.mean(self.distances)

                for part in self.pop:

                    self.ok, part = self.pso_fitness(part, first=False)

                    self.post_array = self.gp_regression()

                    self.samples += 1

                    self.n_data += 1
                    if self.n_data > 4:
                        self.n_data = 1

                self.MSE_data, self.it = self.util.mse(self.g, self.fitness, self.mu_data, self.samples)

                self.sigma_best, self.mu_best = self.sigma_max()

            z = 0

            for part in self.pop:
                self.toolbox.update(action[0], action[1], action[2], action[3], part)
                if self.ok:
                    self.state = np.array(self.state)
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
                        self.state = self.plot.part_position(self.part_ant[:, posx], self.part_ant[:, posy], self.state,
                                                             z)
                        z += 1
                        if self.n_data == 4:
                            self.state = self.plot.state_sigma_mu(self.mu, self.sigma, self.state)
                    self.n_data += 1
                    if self.n_data > 4:
                        self.n_data = 1
            self.g += 1
            if self.ok:
                self.k += 1
                self.ok = False

        reward = self.calculate_reward()

        if np.mean(self.distances) >= 250:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

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
                self.ok, part = self.pso_fitness(part, first=False)
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, part, self.part_ant, self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > 4:
                    self.n_data = 1

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.k += 1
                self.ok = True
                self.last_sample = np.mean(self.distances)

                for part in self.pop:
                    self.ok, part = self.pso_fitness(part, first=False)

                    self.post_array = self.gp_regression()

                    self.samples += 1

                    self.n_data += 1
                    if self.n_data > 4:
                        self.n_data = 1

                self.MSE_data, self.it = self.util.mse(self.g, self.fitness, self.mu_data, self.samples)
                self.mse = mean_squared_error(y_true=self.fitness, y_pred=self.mu_data)

                self.sigma_best, self.mu_best = self.sigma_max()

                self.ok = False

            for part in self.pop:
                self.toolbox.update(action[0], action[1], action[2], action[3], part)

            dis_steps = np.mean(self.distances) - dist_ant

            self.g += 1

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

        reward = self.calculate_reward()

        self.logbook.record(gen=self.g, evals=len(self.pop), **self.stats.compile(self.pop))
        print(self.logbook.stream)
        if np.mean(self.distances) >= 250:
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def iteration(self):
        return self.g

    def data_out(self):

        """
        Return the first and the last position of the particles (drones).
        """

        return self.x_g, self.y_g, self.n, self.X_test, self.secure, self.bench_function, self.grid_min, self.sigma, \
               self.mu, self.MSE_data, self.it, self.part_ant
