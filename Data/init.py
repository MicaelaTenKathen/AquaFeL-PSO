import numpy as np
import time


class Init:
    def __init__(self, GEN, seed=None, sigma_best=None, mu_best=None, c1=None, c2=None, c3=None, c4=None, c31=None, c41=None,
                 length_scale=None, lam=None, part_dist=None, part_ant=None, distances=None, n_data=None, n_plot=None,
                 benchmark_data=None, n=None, sigma_data=None, mu_data=None, MSE_data=None, it=None, mu_d=None,
                 g=None, k=None, samples=None, last_sample=None, x_p=None, y_p=None, y_data=None, part_data=None,
                 x_g=None, y_g=None, y_mult=None, fitness=None, x_h=None, y_h=None, part_array=None, s_ant=None,
                 s_n=None, ok=False, post_array=None, start_time=None):
        # if seed is None:
        seed = [20]  # , 95, 541, 65, 145, 156, 158, 12, 3, 89, 57, 123, 456, 789, 987, 654, 321, 147, 258,
        #     # 369, 741, 852, 963, 159, 951, 753, 357, 756, 8462, 4875]
        # if sigma_best is None:
        sigma_best = [0, 0]
        # if mu_best is None:
        mu_best = [0, 0]
        # if c1 is None:
        c1 = 3.1286
        # if c2 is None:
        c2 = 2.568
        # if c3 is None:
        c3 = 0
        # if c4 is None:
        c4 = 0
        # if c31 is None:
        c31 = 0.49
        # if c41 is None:
        c41 = 0
        # if length_scale is None:
        length_scale = 1
        # if lam is None:
        lam = 0.1
        # if part_dist is None:
        part_dist = np.zeros(8)
        # if part_ant is None:
        part_ant = np.zeros((GEN + 1, 8))
        # if distances is None:
        distances = np.zeros(4)
        # if n_data is None:
        n_data = 1
        # if n_plot is None:
        n_plot = 1
        # if benchmark_data is None:
        benchmark_data = list()
        # if n is None:
        self.n = list()
        # if sigma_data is None:
        sigma_data = list()
        # if mu_data is None:
        mu_data = list()
        # if MSE_data is None:
        MSE_data = list()
        # if it is None:
        it = list()
        # if mu_d is None:
        mu_d = list()
        # if g is None:
        g = 0
        # if k is None:
        self.k = 0
        # if samples is None:

        self.samples = 0
        # if last_sample is None:
        last_sample = 0
        # if x_p is None:
        #     self.x_p = list()
        # if y_p is None:
        #     self.y_p = list()
        # if y_data is None:
        #     self.y_data = list()
        # if part_data is None:
        #     self.part_data = list()
        # if x_g is None:
        #     self.x_g = list()
        # if y_g is None:
        #     self.y_g = list()
        # if y_mult is None:
        #     self.y_mult = list()
        # if x_h is None:
        #     self.x_h = list()
        # if y_h is None:
        #     self.y_h = list()
        # if fitness is None:
        #     self.fitness = list()
        # if part_array is None:
        #     self.part_array = list()
        # if s_ant is None:
        #     self.s_ant = np.zeros(4)
        if s_n is None:
            self.s_n = np.array([True, True, True, True])
        # if post_array is None:
        #     self.post_array = [self.length_scale, self.length_scale, self.length_scale, self.length_scale]
        # if start_time is None:
        #     self.start_time = time.time()
        # self.ok = ok



