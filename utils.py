from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def plot_trajectory(ax, x, y, z=None, colormap = 'jet', num_of_points = None, linewidth = 1, k = 3, plot_waypoints=False, markersize = 0.5):

    if z is None:
        tck,u = interpolate.splprep([x,y],s=0.0, k=k)
        x_i,y_i= interpolate.splev(np.linspace(0,1,num_of_points),tck)
        points = np.array([x_i,y_i]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
        lc = LineCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
        lc.set_array(np.linspace(0,1,len(x_i)))
        ax.add_collection(lc)
        if plot_waypoints:
            ax.plot(x,y,'.', color = 'black', markersize = markersize)
    else:
        tck,u =interpolate.splprep([x,y,z],s=0.0)
        x_i,y_i,z_i= interpolate.splev(np.linspace(0,1,num_of_points), tck)
        points = np.array([x_i,y_i,z_i]).T.reshape(-1,1,3)
        segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
        lc = Line3DCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
        lc.set_array(np.linspace(0,1,len(x_i)))
        ax.add_collection(lc)
        ax.scatter(x,y,z,'k')
        if plot_waypoints:
            ax.plot(x,y,'kx')