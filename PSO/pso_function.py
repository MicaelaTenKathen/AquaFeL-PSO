from Data.utils import Utils
from Data.limits import Limits
from GaussianProcess.gaussianp import Gaussian_Process
import numpy as np
import random
import math
from deap import base
from deap import creator
from deap import tools


class PSO:
    def __init__(self, GEN, grid_min, grid_max, secure, xs, ys, X_test, bench_function, df_bounds):
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
        return

    def createPart(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, smin=None, smax=None,
                       best=None)
        creator.create("BestGP", np.ndarray, fitness=creator.FitnessMax)

    def generatePart(self):
        self.part = creator.Particle([random.uniform(self.pmin, self.pmax) for _ in range(self.size)])
        self.part.speed = np.array([random.uniform(self.smin, self.smax) for _ in range(self.size)])
        self.part.smin = self.smin
        self.part.smax = self.smax
        return self.part

    def updateParticle_n(self, g, c1, c2, c3, c4, part, best, sigma_best, mu_best):
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

    def tool_n(self):
        toolbox = base.Toolbox()
        gen = PSO(self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test, self.bench_function, self.df_bounds)
        toolbox.register("particle", gen.generatePart)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", gen.updateParticle_n)
        return toolbox

    def swarm(self):
        pop = PSO(self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test, self.bench_function, self.df_bounds).tool_n().population(
            n=self.population)
        best = pop[0]
        return pop, best

    def statistic(self):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields
        return stats, logbook

    def initPSO(self, seed):
        pso = PSO(self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test, self.bench_function, self.df_bounds)
        pso.createPart()
        pso.generatePart()
        toolbox = pso.tool_n()
        random.seed(seed[0])
        pop, best = pso.swarm()
        stats, logbook = pso.statistic()
        return toolbox, pop, best, stats, logbook

    def pso_fitness(self, g, s_n, n_data, part_ant, s_ant, part, ok, x_h, y_h, fitness, x_p, y_p, y_data, grid_min, x_g, y_g, n, n_plot, best, init=False):

        def __ini__():


        part, s_n = Limits(self.secure, self.xs, self.ys).new_limit(g, part, s_n, n_data, s_ant, part_ant)
        x_bench = int(part[0])
        y_bench = int(part[1])

        for i in range(len(self.X_test)):
            if self.X_test[i][0] == x_bench and self.X_test[i][1] == y_bench:
                part.fitness.values = [self.bench_function[i]]
                break
        if ok:
            x_h.append(int(part[0]))
            y_h.append(int(part[1]))
            fitness.append(part.fitness.values)
        else:
            x_p.append(part[0])
            y_p.append(part[1])
            y_data.append(part.fitness.values)
            if init:
                x_gap = int(part[0]) + abs(grid_min)
                y_gap = int(part[1]) + abs(grid_min)
                x_g.append(x_gap)
                y_g.append(y_gap)
                n.append(n_data)
                if n_plot > 4:
                    n_plot = float(1)
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            else:
                if g == self.GEN - 1:
                    x_gap = int(part[0]) + abs(grid_min)
                    y_gap = int(part[1]) + abs(grid_min)
                    x_g.append(x_gap)
                    y_g.append(y_gap)
                    n.append(n_data)
                    n_plot += float(1)
                    if n_plot > 4:
                        n_plot = float(1)
                if part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
            if best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        return ok, x_h, y_h, fitness, x_p, y_p, y_data, x_bench, y_bench, part, best, n_plot, s_n

    def step(self, method, dist_ant, gpr, pso, c1, c2, c3, c4, lam, k, pop, g, s_ant, x_g, y_g, n, last_sample, post_array, samples,
             MSE_data, it, toolbox, stats, logbook, s_n, n_data, part_ant, ok, x_h, y_h, fitness, x_p, y_p, y_data,
             n_plot, best, distances, mu_data, sigma_data, sigma, mu, sigma_best, mu_best):
        dis_steps = 0
        while dis_steps < 10:
            for part in pop:
                ok, x_h, y_h, fitness, x_p, y_p, y_data, x_bench, y_bench, part, best, n_plot, s_n = pso.pso_fitness(g,
                                                                                                                     s_n,
                                                                                                                     n_data,
                                                                                                                     part_ant,
                                                                                                                     s_ant,
                                                                                                                     part,
                                                                                                                     ok,
                                                                                                                     x_h,
                                                                                                                     y_h,
                                                                                                                     fitness,
                                                                                                                     x_p,
                                                                                                                     y_p,
                                                                                                                     y_data,
                                                                                                                     self.grid_min,
                                                                                                                     x_g,
                                                                                                                     y_g,
                                                                                                                     n,
                                                                                                                     n_plot,
                                                                                                                     best,
                                                                                                                     init=False)
                part_ant, distances = Utils().distance_part(g, n_data, part, part_ant, distances, init=True)

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
                    ok, x_h, y_h, fitness, x_p, y_p, y_data, x_bench, y_bench, part, best, n_plot, s_n = pso.pso_fitness(g,
                                                                                                                     s_n,
                                                                                                                     n_data,
                                                                                                                     part_ant,
                                                                                                                     s_ant,
                                                                                                                     part,
                                                                                                                     ok,
                                                                                                                     x_h,
                                                                                                                     y_h,
                                                                                                                     fitness,
                                                                                                                     x_p,
                                                                                                                     y_p,
                                                                                                                     y_data,
                                                                                                                     self.grid_min,
                                                                                                                     x_g,
                                                                                                                     y_g,
                                                                                                                     n,
                                                                                                                     n_plot,
                                                                                                                     best,
                                                                                                                     init=False)

                    sigma, mu, sigma_data, mu_data, post_array = gpr.gp_regression(int(part[0]), int(part[1]), x_h, y_h,
                                                                                   fitness, post_array, n_data, mu_data,
                                                                                   sigma_data)

                    samples += 1

                    n_data += 1
                    if n_data > 4:
                        n_data = 1

                MSE_data, it = Utils().mse(g, fitness, mu_data, samples, MSE_data, it)

                sigma_best, mu_best = gpr.sigma_max(sigma, mu)
                ok = False

            for part in pop:
                # toolbox.update(part, best, sigma_best, mu_best, g, GEN, c1, c2, c3, c4)
                toolbox.update(g, c1, c2, c3, c4, part, best, sigma_best, mu_best)

            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)
            mean_dist = np.mean(np.array(distances))
            print(mean_dist)
            dis_steps = np.mean(distances) - dist_ant
            dist_ant = np.mean(distances)

        # if method == 0:
        #     out = [np.append] # vector escalar
        # elif method == 1:
        #     out = [plot] # image
        return lam, k, g, s_ant, x_g, y_g, n, last_sample, post_array, samples, MSE_data, it, s_n, n_data, part_ant, \
               ok, x_h, y_h, fitness, x_p, y_p, y_data, n_plot, best, distances, mu_data, sigma_data, sigma, mu, sigma_best, mu_best, dist_ant

