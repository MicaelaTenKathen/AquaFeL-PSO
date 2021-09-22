from Data.limits import Limits
from Environment.plot import Plots

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import numpy as np
import random
import math

from deap import base
from deap import creator
from deap import tools
"""[https://deap.readthedocs.io/en/master/examples/pso_basic.html]"""


class PSO_fun:
    def __init__(self, n_data, GEN, grid_min, grid_max, secure, xs, ys, X_test, bench_function, df_bounds):
        self.f = int()
        self.k = int()
        self.GEN = GEN
        self.population = 4
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.pmin = grid_min
        self.pmax = grid_max
        self.smin = 0
        self.smax = 0.02
        self.size = 2
        self.wmin = 0.004
        self.wmax = 0.009
        self.secure = secure
        self.xs = xs
        self.ys = ys
        self.X_test = X_test
        self.bench_function = bench_function
        self.df_bounds = df_bounds
        ker = RBF(length_scale=0.7, length_scale_bounds=(1e-1, 10))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1 ** 2)
        self.plot = Plots(xs, ys, X_test, secure, bench_function, grid_min)
        if n_data == 0:
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
            self.n = list()
            self.samples = int()
            self.dist_ant = float()

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

    def updateParticle_n(self, g, c1, c2, c3, c4, part, best, sigma_best, mu_best):

        """
        Calculates the speed and the position of the particles (drones).
        """

        u1 = np.array([random.uniform(0, c1) for _ in range(len(part))])
        u2 = np.array([random.uniform(0, c2) for _ in range(len(part))])
        u3 = np.array([random.uniform(0, c3) for _ in range(len(part))])
        u4 = np.array([random.uniform(0, c4) for _ in range(len(part))])
        v_u1 = u1 * (part.best - part)
        v_u2 = u2 * (best - part)
        v_u3 = u3 * (sigma_best - part)
        v_u4 = u4 * (mu_best - part)
        w = self.wmax - ((self.wmax - self.wmin) / self.GEN) * g
        part.speed = v_u1 + v_u2 + v_u3 + v_u4 + part.speed * w
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = part + part.speed

        return part

    def tool_n(self, n_data):

        """
        The operators are registered in the toolbox with their parameters.
        """

        toolbox = base.Toolbox()
        gen = PSO_fun(n_data, self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test,
                      self.bench_function, self.df_bounds)
        toolbox.register("particle", gen.generatePart)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", gen.updateParticle_n)

        return toolbox

    def swarm(self, n_data):

        """
        Creates a population.
        """

        pop = PSO_fun(n_data, self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test,
                      self.bench_function, self.df_bounds).tool_n(n_data).population(
            n=self.population)
        best = pop[0]

        return pop, best

    def statistic(self):

        """
        Visualizes the stats of the code.
        """

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        return stats, logbook

    def initPSO(self, seed, n_data):

        """
        Initialization of the pso.
        """

        pso = PSO_fun(n_data, self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test,
                      self.bench_function, self.df_bounds)
        pso.createPart()
        pso.generatePart()
        toolbox = pso.tool_n(n_data)
        random.seed(seed[0])
        pop, best = pso.swarm(n_data)
        stats, logbook = pso.statistic()

        return best, pop, toolbox, stats, logbook

    def pso_fitness(self, g, part_ant, part, ok, best, n_data, first=False):

        """
        Obtains the local best (part.best) of each particle (drone) and the global best (best) of the swarm (fleet).
        """

        part, s_n = Limits(self.secure, self.xs, self.ys).new_limit(g, part, self.s_n, n_data, self.s_ant, part_ant)
        self.x_bench = int(part[0])
        self.y_bench = int(part[1])

        for i in range(len(self.X_test)):
            if self.X_test[i][0] == self.x_bench and self.X_test[i][1] == self.y_bench:
                part.fitness.values = [self.bench_function[i]]
                break
        if ok:
            self.x_h.append(int(part[0]))
            self.y_h.append(int(part[1]))
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
                self.n.append(n_data)
                if self.n_plot > 4:
                    self.n_plot = float(1)
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            else:
                if g == self.GEN - 1:
                    x_gap = int(part[0]) + abs(self.grid_min)
                    y_gap = int(part[1]) + abs(self.grid_min)
                    self.x_g.append(x_gap)
                    self.y_g.append(y_gap)
                    self.n.append(n_data)
                    self.n_plot += float(1)
                    if self.n_plot > 4:
                        self.n_plot = float(1)
                if part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
            if best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        return ok, part, best

    def gp_regression(self, n_data, post_array):

        """
        Fits the gaussian process.
        """

        x_a = np.array(self.x_h).reshape(-1, 1)
        y_a = np.array(self.y_h).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(self.fitness).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        mu, sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = np.min(np.exp(self.gpr.kernel_.theta[0]))
        post_array[n_data - 1] = post_ls

        for i in range(len(self.X_test)):
            di = self.X_test[i]
            dix = di[0]
            diy = di[1]
            if dix == self.x_bench and diy == self.y_bench:
                self.mu_data.append(mu[i])
                self.sigma_data.append(sigma[i])

        return sigma, mu, post_array

    def sigma_max(self, sigma, mu):

        """
        Returns the coordinates of the maximum uncertainty (sigma_best) and the maximum contamination (mu_best).
        """

        sigma_max = np.max(sigma)
        index_sigma = np.where(sigma == sigma_max)
        index_x1 = index_sigma[0]
        index_x2 = index_x1[0]
        index_x = int(self.X_test[index_x2][0])
        index_y = int(self.X_test[index_x2][1])

        mu_max = np.max(mu)
        index_mu = np.where(mu == mu_max)
        index_x1mu = index_mu[0]
        index_x2mu = index_x1mu[0]
        index_xmu = int(self.X_test[index_x2mu][0])
        index_ymu = int(self.X_test[index_x2mu][1])

        best_1 = [index_x, index_y]
        sigma_best = np.array(best_1)

        best_2 = [index_xmu, index_ymu]
        mu_best = np.array(best_2)

        return sigma_best, mu_best

    def initcode(self, pop, pso, util, toolbox, g, c1, c2, c3, c4, lam, best, post_array, method, part_ant,
                 distances):

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

        global MSE_data, it

        sigma_best = [0, 0]
        mu_best = [0, 0]
        n_data = 1
        last_sample = 0
        samples = 0
        ok = False
        k = 0

        for part in pop:
            ok, part, best = pso.pso_fitness(g, part_ant, part, ok, best, n_data, first=True)

            part_ant, distances = util.distance_part(g, n_data, part, part_ant, distances, dfirst=True)

            n_data += 1
            if n_data > 4:
                n_data = 1

        for part in pop:
            toolbox.update(g, c1, c2, c3, c4, part, best, sigma_best, mu_best)

        while k == 0:
            for part in pop:

                ok, part, best = pso.pso_fitness(g, part_ant, part, ok, best, n_data, first=False)

                part_ant, distances = util.distance_part(g, n_data, part, part_ant, distances, dfirst=False)

                n_data += 1
                if n_data > 4:
                    n_data = 1

            if (np.mean(distances) - last_sample) >= (np.min(post_array) * lam):
                ok = True
                last_sample = np.mean(distances)

                for part in pop:

                    ok, part, best = pso.pso_fitness(g, part_ant, part, ok, best, n_data, first=False)

                    sigma, mu, post_array = pso.gp_regression(n_data, post_array)

                    samples += 1

                    n_data += 1
                    if n_data > 4:
                        n_data = 1

                MSE_data, it = util.mse(g, self.fitness, self.mu_data, samples)

                sigma_best, mu_best = pso.sigma_max(sigma, mu)

            z = 0

            for part in pop:
                toolbox.update(g, c1, c2, c3, c4, part, best, sigma_best, mu_best)
                if ok:
                    if method == 0:
                        if z == 0:
                            out = np.zeros((self.GEN, 22))
                        out[0, z] = part[0]
                        z += 1
                        out[0, z] = part[1]
                        z += 1
                        out[0, z + 6] = part.best[0]
                        out[0, z + 7] = part.best[1]
                        if n_data == 4:
                            out[0, 16] = best[0]
                            out[0, 17] = best[1]
                            out[0, 18] = sigma_best[0]
                            out[0, 19] = sigma_best[1]
                            out[0, 20] = mu_best[0]
                            out[0, 21] = mu_best[1]
                    else:
                        out = np.zeros((self.GEN, 6))
                        out[z] = self.plot.part_position(part_ant[:, 2 * z], part_ant[:, 2 * z + 1])
                        z += 1
                        if n_data == 4:
                            z_un, z_mean = self.plot.Z_var_mean(mu, sigma)
                            out[4] = z_un
                            out[5] = z_mean
                    n_data += 1
                    if n_data > 4:
                        n_data = 1
            g += 1
            if ok:
                k += 1
                ok = False

        return out, sigma_best, mu_best, post_array, last_sample, MSE_data, it, g, k, samples

    def step(self, ok, pop, pso, util, toolbox, out, g, c1, c2, c3, c4, lam, best, post_array, last_sample, method,
             sigma_best, mu_best, part_ant, distances, k, f, samples, MSE_data, it):

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
        global mu, sigma
        dist_ant = np.mean(distances)
        n_data = 1
        f += 1

        while dis_steps < 10 and g < self.GEN - 1:

            for part in pop:
                ok, part, best = pso.pso_fitness(g, part_ant, part, ok, best, n_data, first=False)
                part_ant, distances = util.distance_part(g, n_data, part, part_ant, distances, dfirst=False)

                n_data += 1
                if n_data > 4:
                    n_data = 1

            if (np.mean(distances) - last_sample) >= (np.min(post_array) * lam):
                c3 = c3
                c4 = c4
                k += 1
                ok = True
                last_sample = np.mean(distances)

                for part in pop:
                    ok, part, best = pso.pso_fitness(g, part_ant, part, ok, best, n_data, first=False)

                    sigma, mu, post_array = pso.gp_regression(n_data, post_array)

                    samples += 1

                    n_data += 1
                    if n_data > 4:
                        n_data = 1

                MSE_data, it = util.mse(g, self.fitness, self.mu_data, samples)

                sigma_best, mu_best = pso.sigma_max(sigma, mu)
                ok = False

            for part in pop:
                toolbox.update(g, c1, c2, c3, c4, part, best, sigma_best, mu_best)

            dis_steps = np.mean(distances) - dist_ant

            g += 1

        if g == self.GEN - 1:
            pass
        else:
            z = 0
            for part in pop:
                if method == 0:
                    out[f, z] = part[0]
                    z += 1
                    out[f, z] = part[1]
                    z += 1
                    out[f, z + 6] = part.best[0]
                    out[f, z + 7] = part.best[1]
                    if n_data == 4:
                        out[f, 16] = best[0]
                        out[f, 17] = best[1]
                        out[f, 18] = sigma_best[0]
                        out[f, 19] = sigma_best[1]
                        out[f, 20] = mu_best[0]
                        out[f, 21] = mu_best[1]
                else:
                    out[z] = self.plot.part_position(part_ant[:, 2 * z], part_ant[:, 2 * z + 1])
                    z += 1
                    if n_data == 4:
                        z_un, z_mean = self.plot.Z_var_mean(mu, sigma)
                        out[4] = z_un
                        out[5] = z_mean

                n_data += 1
                if n_data > 4:
                    n_data = 1

        return out, sigma_best, mu_best, post_array, last_sample, MSE_data, it, g, k, f, samples, mu, sigma

    def data_out(self):

        """
        Return the first and the last position of the particles (drones).
        """

        return self.x_g, self.y_g, self.n
