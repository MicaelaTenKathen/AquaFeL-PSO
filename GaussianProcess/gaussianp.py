from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np


class Gaussian_Process:
    def __init__(self, x_p, y_p, y_data, n_data, X_test, post_array, x_bench, y_bench, mu_data, sigma_data):
        self.x_p = x_p
        self.y_p = y_p
        self.y_data = y_data
        self.n_data = n_data
        self.X_test = X_test
        self.post_array = post_array
        self.x_bench = x_bench
        self.y_bench = y_bench
        self.mu_data = mu_data
        self.sigma_data = sigma_data

        self.ker = RBF(length_scale=0.7, length_scale_bounds=(1e-1, 10))
        self.gpr = GaussianProcessRegressor(kernel=self.ker, alpha=1 ** 2)

        return

    def data(self):
        x_a = np.array(self.x_p).reshape(-1, 1)
        y_a = np.array(self.y_p).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(self.y_data).reshape(-1, 1)

        return x_a, y_a, x_train, y_train

    def gp_regression(self):
        x_a, y_a, x_train, y_train = Gaussian_Process(self.x_p, self.y_p, self.y_data, self.n_data, self.X_test,
                                                      self.post_array, self.x_bench, self.y_bench, self.mu_data,
                                                      self.sigma_data).data()
        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        mu, sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = np.min(np.exp(self.gpr.kernel_.theta[0]))
        self.post_array[self.n_data - 1] = post_ls

        return sigma, mu, self.post_array

    def gpr_value(self):
        mu, sigma, p_array = Gaussian_Process(self.x_p, self.y_p, self.y_data, self.n_data, self.X_test,
                                              self.post_array, self.x_bench, self.y_bench, self.mu_data,
                                              self.sigma_data).gp_regression()
        for i in range(len(self.X_test)):
            di = self.X_test[i]
            dix = di[0]
            diy = di[1]
            if dix == self.x_bench and diy == self.y_bench:
                self.mu_data.append(mu[i])
                self.sigma_data.append(sigma[i])

        return self.sigma_data, self.mu_data

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
