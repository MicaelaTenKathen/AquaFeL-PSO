import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


class Plots():
    def __init__(self, xs, ys, X_test, grid, bench_function, grid_min):
        self.xs = xs
        self.ys = ys
        self.X_test = X_test
        self.grid = grid
        self.bench_function = bench_function
        self.grid_min = grid_min

    def Z_var_mean(self, mu, sigma):
        Z_un = np.zeros([self.grid.shape[0], self.grid.shape[1]])
        Z_mean = np.zeros([self.grid.shape[0], self.grid.shape[1]])
        for i in range(len(self.X_test)):
            Z_un[self.X_test[i][0], self.X_test[i][1]] = sigma[i]
            Z_mean[self.X_test[i][0], self.X_test[i][1]] = mu[i]
        with open('./Position/uncertainty.npy', 'wb') as g:
            np.save(g, Z_un)
        with open('./Position/mean.npy', 'wb') as o:
            np.save(o, Z_mean)
        Z_un[Z_un == 0] = np.nan
        Z_mean[Z_mean == 0] = np.nan
        return Z_un, Z_mean

    def state_sigma_mu(self, mu, sigma, state):
        for i in range(len(self.X_test)):
            state[self.X_test[i][0], self.X_test[i][1], 4] = sigma[i]
            state[self.X_test[i][0], self.X_test[i][1], 5] = mu[i]
        # with open('./Position/uncertainty.npy', 'wb') as g:
        #     np.save(g, state)
        # with open('./Position/mean.npy', 'wb') as o:
        #     np.save(o, Z_mean)
        return state

    def part_position(self, array_position_x, array_position_y, state, z):
        for i in range(len(array_position_x)):
            state[int(array_position_x[i]), int(array_position_y[i]), z] = 1
        # with open('./Position/position' + str(n_data) + '.npy', 'wb') as g:
        #     np.save(g, position)
        return state

    def evolucion(self, log):
        gen = log.select("gen")
        fit_mins = log.select("min")
        fit_maxs = log.select("max")
        fit_ave = log.select("avg")

        fig, ax1 = plt.subplots()
        ax1.plot(gen, fit_mins, "b")
        ax1.plot(gen, fit_maxs, "r")
        ax1.plot(gen, fit_ave, "--k")
        ax1.fill_between(gen, fit_mins, fit_maxs,
                         where=fit_maxs >= fit_mins,
                         facecolor="g", alpha=0.2)
        ax1.set_xlabel("Generación")
        ax1.set_ylabel("Fitness")
        ax1.legend(["Min", "Max", "Avg"])
        plt.grid(True)

    def bench_plot(self):
        plot = np.zeros([self.xs, self.ys])
        for i in range(len(self.X_test)):
            plot[self.X_test[i][0], self.X_test[i][1]] = self.bench_function[i]
        plot[self.grid == 0] = np.nan
        benchma_plot = plot.T
        return plot, benchma_plot

    def gaussian(self, x_ga, y_ga, n, mu, sigma, part_ant):
        Z_var, Z_mean = Plots(self.xs, self.ys, self.X_test, self.grid, self.bench_function, self.grid_min).Z_var_mean(mu, sigma)

        fig, axs = plt.subplots(2, 1, figsize=(5, 10))

        im1 = axs[0].scatter(x_ga, y_ga, c=n, cmap="gist_rainbow", marker='.')
        p1x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 0]))
        p1y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 1]))
        axs[0].plot(p1x, p1y, 'r')
        p2x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 2]))
        p2y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 3]))
        axs[0].plot(p2x, p2y, 'w')
        p3x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 4]))
        p3y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 5]))
        axs[0].plot(p3x, p3y, 'c')
        p4x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 6]))
        p4y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 7]))
        axs[0].plot(p4x, p4y, 'k')

        im2 = axs[0].imshow(Z_var.T, interpolation='bilinear', origin='lower', cmap="viridis")
        #plt.colorbar(im2, ax=axs[0], label='σ', shrink=1.0)
        # axs[0].set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]")
        axs[0].set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
        axs[0].set_xticks([0, 50, 100])
        axs[0].set_aspect('equal')
        axs[0].set_ylim([self.ys, 0])
        axs[0].grid(True)
        # ticks_x = ticker.FuncFormatter()
        # print(ticks_x)
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        # print(ticks_x2)
        axs[0].xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs[0].yaxis.set_major_formatter(ticks_y)

        im3 = axs[1].imshow(Z_mean.T, interpolation='bilinear', origin='lower', cmap="jet")
        #plt.colorbar(im3, ax=axs[1], label='µ', shrink=1.0)
        axs[1].set_xlabel("x [m]")
        axs[1].set_ylabel("y [m]")
        axs[1].set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
        axs[1].set_xticks([0, 50, 100])
        axs[1].set_ylim([self.ys, 0])
        axs[1].set_aspect('equal')
        axs[1].grid(True)
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs[1].xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs[1].yaxis.set_major_formatter(ticks_y)

        #plt.savefig("Image/plot.pdf")
        plt.show()

    def benchmark(self):
        plot, benchmark_plot = Plots(self.xs, self.ys, self.X_test, self.grid, self.bench_function, self.grid_min).bench_plot()

        fig = plt.figure(2)
        ax1 = fig.add_subplot(121)
        im4 = ax1.imshow(benchmark_plot, interpolation='bilinear', origin='lower', cmap="jet")
        plt.colorbar(im4, label='µ', shrink=0.74)
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_ylim([self.ys, 0])
        ax1.set_aspect('equal')
        ax1.grid(True)
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        ax1.xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        ax1.yaxis.set_major_formatter(ticks_y)

        plt.savefig("Image/plot9.pdf")

        return plot

    def error(self, MSE_data, it):
        plt.figure(3)
        plt.plot(it, MSE_data, '-')
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.title("Mean Square Error")
        # plt.savefig("MSE", dpi=200)
        plt.show()