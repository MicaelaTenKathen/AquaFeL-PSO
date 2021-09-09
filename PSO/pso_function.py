import numpy as np
from sys import path
from Data.limits import Limits
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

    def initPSO(self):
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
        gen = PSO(self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test, self.bench_function).generatePart
        toolbox.register("particle", gen)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update",
                         PSO(self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test, self.bench_function).updateParticle_n)
        return toolbox

    def swarm(self):
        pop = PSO(self.GEN, self.grid_min, self.grid_max, self.secure, self.xs, self.ys, self.X_test, self.bench_function).tool_n().population(
            n=self.population)
        best = pop[0]
        return pop, best

    def pso_fitness(self, g, s_n, n_data, part_ant, s_ant, part, ok, x_h, y_h, fitness, x_p, y_p, y_data, init, grid_min, x_g, y_g, n, n_plot, best):

        part, s_n = Limits(g, part, s_n, self.secure, self.xs, self.ys, n_data, part_ant, s_ant,
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
