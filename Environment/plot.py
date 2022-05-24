import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy import interpolate
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from Environment.map import Map
from matplotlib.colors import LinearSegmentedColormap


class Plots():
    def __init__(self, xs, ys, X_test, grid, bench_function, grid_min, grid_or):
        self.xs = xs
        self.ys = ys
        self.grid_or = grid_or
        self.X_test = X_test
        self.grid = grid
        self.bench_function = bench_function
        self.grid_min = grid_min
        self.grid_or = Map(self.xs, ys).black_white()
        self.X1 = np.arange(0, self.grid.shape[1], 1)
        self.Y1 = np.arange(0, self.grid.shape[0], 1)
        self.cmap = LinearSegmentedColormap.from_list('name', ['green', 'yellow', 'red'])

    def Z_var_mean(self, mu, sigma):
        Z_un = np.zeros([self.grid.shape[0], self.grid.shape[1]])
        Z_mean = np.zeros([self.grid.shape[0], self.grid.shape[1]])
        for i in range(len(self.X_test)):
            Z_un[self.X_test[i][0], self.X_test[i][1]] = sigma[i]
            Z_mean[self.X_test[i][0], self.X_test[i][1]] = mu[i]
        Z_un[self.grid_or == 0] = np.nan
        Z_mean[self.grid_or == 0] = np.nan
        return Z_un, Z_mean

    def state_sigma_mu(self, mu, sigma, state):
        for i in range(len(self.X_test)):
            state[4, self.X_test[i][0], self.X_test[i][1]] = sigma[i]
            state[5, self.X_test[i][0], self.X_test[i][1]] = mu[i]
        # with open('./Position/uncertainty.npy', 'wb') as g:
        #     np.save(g, state)
        # with open('./Position/mean.npy', 'wb') as o:
        #     np.save(o, Z_mean)
        return state

    @staticmethod
    def part_position(array_position_x, array_position_y, state, z):
        for i in range(len(array_position_x)):
            state[z, int(array_position_x[i]), int(array_position_y[i])] = 1
        # with open('./Position/position' + str(n_data) + '.npy', 'wb') as g:
        #     np.save(g, position)
        return state

    @staticmethod
    def evolucion(log):
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

    def gaussian(self, mu, sigma, part_ant):
        Z_var, Z_mean = self.Z_var_mean(mu, sigma)

        fig, axs = plt.subplots(2, 1, figsize=(5, 10))

        self.plot_trajectory(axs[0], part_ant[:, 0], part_ant[:, 1], z=None, colormap='winter', num_of_points=(int((part_ant[:,0].shape)[0])*10))
        self.plot_trajectory(axs[0], part_ant[:, 2], part_ant[:, 3], z=None, colormap='Wistia', num_of_points=(int((part_ant[:,0].shape)[0])*10))
        self.plot_trajectory(axs[0], part_ant[:, 4], part_ant[:, 5], z=None, colormap='Purples', num_of_points=(int((part_ant[:,0].shape)[0])*10))
        self.plot_trajectory(axs[0], part_ant[:, 6], part_ant[:, 7], z=None, colormap='Reds', num_of_points=(int((part_ant[:,0].shape)[0])*10))
        #im1 = axs[0].scatter(x_ga, y_ga, c=n, cmap="gist_rainbow", marker='.')
        #p1x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 0]))
        #p1y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 1]))
        #axs[0].plot(p1x, p1y, 'r')
        #p2x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 2]))
        #p2y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 3]))
        #axs[0].plot(p2x, p2y, 'w')
        #p3x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 4]))
        #p3y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 5]))
        #axs[0].plot(p3x, p3y, 'c')
        #p4x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 6]))
        #p4y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 7]))
        #axs[0].plot(p4x, p4y, 'k')

        im2 = axs[0].imshow(Z_var.T, interpolation='bilinear', origin='lower', cmap="gist_yarg")
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

        # plt.savefig("../Image/GT3/Tabla_4.png")
        plt.show()

    def benchmark(self):
        plot_bench = np.copy(self.bench_function)
        plot_bench[self.grid_or == 0] = np.nan
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        im4 = ax1.imshow(plot_bench.T, interpolation='bilinear', origin='lower', cmap="jet")
        plt.colorbar(im4, label='µ', shrink=1)
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_ylim([self.ys, 0])
        ax1.set_aspect('equal')
        ax1.grid(True)
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        ax1.xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        ax1.yaxis.set_major_formatter(ticks_y)
        # plt.savefig("../Image/Contamination/GT3/Ground3.png")
        plt.show()

    def plot_classic(self, mu, sigma, part_ant):
        Z_var, Z_mean = self.Z_var_mean(mu, sigma)
        fig, axs = plt.subplots(2, 1, figsize=(5, 10))
        initial_x = list()
        initial_y = list()
        final_x = list()
        final_y = list()
        for i in range(part_ant.shape[1]):
            if i % 2 == 0:
                initial_x.append(part_ant[0, i])
                final_x.append(part_ant[-1, i])
            else:
                initial_y.append(part_ant[0, i])
                final_y.append(part_ant[-1, i])
        self.plot_trajectory_classic(axs[0], part_ant[:, 0], part_ant[:, 1], colormap='winter')
        self.plot_trajectory_classic(axs[0], part_ant[:, 2], part_ant[:, 3], colormap='Wistia')
        self.plot_trajectory_classic(axs[0], part_ant[:, 4], part_ant[:, 5], colormap='Purples')
        self.plot_trajectory_classic(axs[0], part_ant[:, 6], part_ant[:, 7], colormap='Reds')
        axs[0].plot(initial_x, initial_y, 'o', color='black', markersize=3, label='ASVs initial positions')
        axs[0].plot(final_x, final_y, 'X', color='red', markersize=3, label='ASVs final positions')
        axs[0].legend(loc=3, fontsize=6)
        # im1 = axs[0].scatter(x_ga, y_ga, c=n, cmap="gist_rainbow", marker='.')
        # p1x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 0]))
        # p1y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 1]))
        # axs[0].plot(p1x, p1y, 'r')
        # p2x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 2]))
        # p2y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 3]))
        # axs[0].plot(p2x, p2y, 'w')
        # p3x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 4]))
        # p3y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 5]))
        # axs[0].plot(p3x, p3y, 'c')
        # p4x = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 6]))
        # p4y = list(map(lambda x: x + abs(self.grid_min), part_ant[:, 7]))
        # axs[0].plot(p4x, p4y, 'k')

        im2 = axs[0].imshow(Z_var.T, interpolation='bilinear', origin='lower', cmap="gist_yarg")
        plt.colorbar(im2, ax=axs[0], label='σ', shrink=1.0)
        # axs[0].set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]")
        axs[0].set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
        axs[0].set_xticks([0, 50, 100])
        axs[0].set_aspect('equal')
        axs[0].set_ylim([self.ys, 0])
        axs[0].grid(True)
        # ticks_x = ticker.FuncFormatter()
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs[0].xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs[0].yaxis.set_major_formatter(ticks_y)

        im3 = axs[1].imshow(Z_mean.T, interpolation='bilinear', origin='lower', cmap="jet")
        plt.colorbar(im3, ax=axs[1], label='µ', shrink=1.0)
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

        plt.savefig("../Image/Contamination/GT3/Tabla_3.png")
        plt.show()

    @staticmethod
    def error(MSE_data, it):
        plt.figure()
        plt.plot(it, MSE_data, '-')
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.title("Mean Square Error")
        # plt.savefig("MSE", dpi=200)
        plt.show()

    @staticmethod
    def plot_trajectory(ax, x, y, z=None, colormap='jet', num_of_points=None, linewidth=1, k=5, plot_waypoints=True,
                        markersize=0.5):

        if z is None:
            tck, u = interpolate.splprep([x, y], s=0.0, k=k)
            x_i, y_i = interpolate.splev(np.linspace(0, 1, num_of_points), tck)
            points = np.array([x_i, y_i]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
            lc = LineCollection(segments, norm=plt.Normalize(0, 1), cmap=plt.get_cmap(colormap), linewidth=linewidth)
            lc.set_array(np.linspace(0, 1, len(x_i)))
            ax.add_collection(lc)
            if plot_waypoints:
                ax.plot(x, y, '.', color='black', markersize=markersize)
        else:
            tck, u = interpolate.splprep([x, y, z], s=0.0)
            x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, num_of_points), tck)
            points = np.array([x_i, y_i, z_i]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
            lc = Line3DCollection(segments, norm=plt.Normalize(0, 1), cmap=plt.get_cmap(colormap), linewidth=linewidth)
            lc.set_array(np.linspace(0, 1, len(x_i)))
            ax.add_collection(lc)
            ax.scatter(x, y, z, 'k')
            if plot_waypoints:
                ax.plot(x, y, 'kx')

    @staticmethod
    def plot_trajectory_classic(ax, x, y, z=None, colormap='jet', linewidth=1, plot_waypoints=True,
                        markersize=0.5):
        if z is None:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
            lc = LineCollection(segments, norm=plt.Normalize(0, 1), cmap=plt.get_cmap(colormap), linewidth=linewidth)
            lc.set_array(np.linspace(0, 1, len(x)))
            ax.add_collection(lc)
            if plot_waypoints:
                ax.plot(x, y, '.', color='black', markersize=markersize)
        else:
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
            lc = Line3DCollection(segments, norm=plt.Normalize(0, 1), cmap=plt.get_cmap(colormap), linewidth=linewidth)
            lc.set_array(np.linspace(0, 1, len(x)))
            ax.add_collection(lc)
            ax.scatter(x, y, z, 'k')
            if plot_waypoints:
                ax.plot(x, y, 'kx')

    def detection_areas(self, mu, sigma):
        Z_var, Z_mean = self.Z_var_mean(mu, sigma)

        fig, axs = plt.subplots()

        im3 = axs.imshow(Z_mean.T, interpolation='none', origin='lower', cmap=self.cmap)
        plt.colorbar(im3, ax=axs, label='µ', shrink=1.0)
        axs.set_xlabel("x [m]")
        axs.set_ylabel("y [m]")
        axs.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
        axs.set_xticks([0, 50, 100])
        axs.set_ylim([self.ys, 0])
        axs.set_aspect('equal')
        axs.grid(True)
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs.xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs.yaxis.set_major_formatter(ticks_y)

        plt.show()

    def action_areas(self, dict_coord_, dict_impo_, k):
        action_zone = np.zeros([self.grid.shape[0], self.grid.shape[1]])
        j = 0
        while j < k:
            j += 1
            action_coord = list(dict_coord_["action_zone%s" % j])
            action_impo = list(dict_impo_["action_zone%s" % j])
            for i in range(len(action_coord)):
                coord = action_coord[i]
                x = coord[0]
                y = coord[1]
                action_zone[x, y] = action_impo[i]
        action_zone[self.grid_or == 0] = np.nan
        fig, axs = plt.subplots()
        im2 = axs.imshow(action_zone.T, interpolation='none', origin='lower', cmap=self.cmap)
        plt.colorbar(im2, ax=axs, label='Importance', shrink=1.0)
        axs.set_xlabel("x [m]")
        axs.set_ylabel("y [m]")
        axs.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
        axs.set_xticks([0, 50, 100])
        axs.set_aspect('equal')
        axs.set_ylim([self.ys, 0])
        axs.grid(True)
        # ticks_x = ticker.FuncFormatter()
        ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs.xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
        axs.yaxis.set_major_formatter(ticks_y)
