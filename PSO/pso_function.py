import numpy as np
from sys import path
from Data.limits import Limits
import numpy as np
import random
import math
from deap import base
from deap import creator
from deap import tools


class PSO_init:
    def __init__(self, g, GEN, c1, c2, c3, c4, pmin, pmax, part, best, sigma_best, mu_best, grid_min, grid_max, population=4, smin=0, smax=0.02, size=2, wmin=0.004, wmax=0.009):
        self.GEN = GEN
        self.g = g
        self.population = population
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.pmin = pmin
        self.pmax = pmax
        self.part = part
        self.best = best
        self.sigma_best = sigma_best
        self.mu_best = mu_best
        self.smin = smin
        self.smax = smax
        self.size = size
        self.wmin = wmin
        self.wmax = wmax
        return

    def initPSO(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, smin=None, smax=None, best=None)
        creator.create("BestGP", np.ndarray, fitness=creator.FitnessMax)

    def generate(self):
        part = creator.Particle([random.uniform(self.pmin, self.pmax) for _ in range(self.size)])
        part.speed = np.array([random.uniform(self.smin, self.smax) for _ in range(self.size)])
        part.smin = self.smin
        part.smax = self.smax

    def updateParticle_n(self):
        u1 = np.array([random.uniform(0, self.c1) for _ in range(len(self.part))])
        u2 = np.array([random.uniform(0, self.c2) for _ in range(len(self.part))])
        u3 = np.array([random.uniform(0, self.c3) for _ in range(len(self.part))])
        u4 = np.array([random.uniform(0, self.c4) for _ in range(len(self.part))])
        v_u1 = u1 * (self.part.best - self.part)
        v_u2 = u2 * (self.best - self.part)
        v_u3 = u3 * (self.sigma_best - self.part)
        v_u4 = u4 * (self.mu_best - self.part)
        w = self.wmax - ((self.wmax - self.wmin) / self.GEN) * self.g
        self.part.speed = v_u1 + v_u2 + v_u3 + v_u4 + self.part.speed * w
        for i, speed in enumerate(self.part.speed):
            if abs(speed) < self.part.smin:
                self.part.speed[i] = math.copysign(self.part.smin, speed)
            elif abs(speed) > self.part.smax:
                self.part.speed[i] = math.copysign(self.part.smax, speed)
        self.part[:] = self.part + self.part.speed

    def tool_n(self):
        toolbox = base.Toolbox()
        toolbox.register("particle", PSO_init(self.g, self.GEN, self.c1, self.c2, self.c3, self.c4, self.pmin, self.pmax,
                                              self.part, self.best, self.sigma_best, self.mu_best, self.grid_min, self.grid_max).generate())
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", PSO_init(self.g, self.GEN, self.c1, self.c2, self.c3, self.c4, self.pmin, self.pmax,
                                              self.part, self.best, self.sigma_best, self.mu_best, self.grid_min, self.grid_max).updateParticle_n())

    def swarm(self):
        pop = PSO_init(self.g, self.GEN, self.c1, self.c2, self.c3, self.c4, self.pmin, self.pmax, self.part,
                 self.best, self.sigma_best, self.mu_best, self.grid_min, self.grid_max).tool_n().population(n=self.population)
        best = pop[0]
        return pop, best


class PSO_fitness:
    def __init__(self):

        return
    def pso_fitness(self, s_n, n_data, part_ant, s_ant, part, ok, x_h, y_h, fitness, x_p, y_p, y_data, init, grid_min, x_g, y_g, n, n_plot, best):
        part, s_n = Limits(self.g, part, s_n, self.secure, self.xs, self.ys, n_data, part_ant, s_ant,
                           file=0).new_limit()
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
                if self.g == self.GEN - 1:
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

        return
