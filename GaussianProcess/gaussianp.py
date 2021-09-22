from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from PSO.pso_function import PSO_fun


class Gaussian_Process:
    def __init__(self, X_test):
        self.X_test = X_test

        ker = RBF(length_scale=0.7, length_scale_bounds=(1e-1, 10))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1 ** 2)

    def gp_regression(self, x_bench, y_bench, fitness, post_array, n_data, mu_data, sigma_data, GEN, grid_min, grid_max, secure, xs, ys, bench_function, df_bounds):
        PSO_fun.__init__(self, GEN, grid_min, grid_max, secure, xs, ys, self.X_test, bench_function, df_bounds)
        x_a = np.array(self.x_h).reshape(-1, 1)
        y_a = np.array(self.y_h).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(fitness).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        mu, sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = np.min(np.exp(self.gpr.kernel_.theta[0]))
        post_array[n_data - 1] = post_ls

        for i in range(len(self.X_test)):
            di = self.X_test[i]
            dix = di[0]
            diy = di[1]
            if dix == x_bench and diy == y_bench:
                mu_data.append(mu[i])
                sigma_data.append(sigma[i])

        return sigma, mu, sigma_data, mu_data, post_array

    def sigma_max(self, sigma, mu):
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
        gp_best = np.array(best_1)

        best_2 = [index_xmu, index_ymu]
        mu_best = np.array(best_2)

        return gp_best, mu_best