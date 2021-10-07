import numpy as np
import matplotlib.pyplot as plt
from Environment.EnvironmentUtils import GroundTruth
import gym

import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.colors import LinearSegmentedColormap

class JumpingPatrollingEnvironment(gym.Env):

    def __init__(self,
                 navigation_map,
                 initial_position,
                 initial_extra_measurements=None,
                 is_model=True,
                 max_distance=250,
                 max_meas_distance=100,
                 min_meas_distance=5,
                 initial_seed=10,
                 reward_function='regret'):

        """ Navigation map. 1 is navigable. """

        self.navigation_map = navigation_map
        assert self.navigation_map[initial_position[0], initial_position[1]] == 1, "Invalid initial position"

        self.navigable_positions = np.copy(np.where(self.navigation_map != 0)).T  # Only water positions
        self.possible_positions = np.copy(np.where(self.navigation_map != 2)).T  # All positions for evaluation

        """ Action and observation space """

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-50.0, high=50.0,
                                                shape=(4, self.navigation_map.shape[0], self.navigation_map.shape[1]),
                                                dtype=np.float64)

        """Initial deploy position """
        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.trajectory = np.atleast_2d(np.copy(self.initial_position))
        self.initial_extra_measurements = initial_extra_measurements  # Initial measurements points #

        """ Is this scenario a Model or Ground Truth?"""
        self.is_model = is_model
        self.max_distance = max_distance
        self.max_meas_distance = max_meas_distance
        self.min_meas_distance = min_meas_distance
        self.step_number = 0
        self.maximum_peak_location = np.copy(initial_position)
        self.num_of_collisions = 0
        self.distance = 0
        self.reward_function = reward_function
        self.accumulated_reward = 0
        self.previous_mse = 0

        """ Generate Benchmark Function """
        np.random.seed(initial_seed)
        self.GroundTruth = GroundTruth(self.navigation_map, function_type='shekel', initial_seed=initial_seed)
        self.GroundTruth_field = self.GroundTruth.sample_gt()

        """ Gaussian Process attributes """
        kernel = ConstantKernel() * RBF() * DotProduct()
        self.GaussianProcess = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

        # First Measuring and updating the gaussian process #
        self.measured_locations = None
        self.measured_values = None
        self.measure_and_update_gaussian_process()

    def seed(self, seed):
        np.random.seed(seed)

    def measure_and_update_gaussian_process(self, update = True):

        # Add element to X
        if self.measured_locations is None:
            self.measured_locations = np.atleast_2d(np.array(self.position[np.newaxis]))

            if self.initial_extra_measurements is not None:
                self.measured_locations = np.row_stack((self.measured_locations, self.initial_extra_measurements))

            self.measured_values = np.atleast_2d(self.GroundTruth_field[self.position[0], self.position[1]])

            if self.initial_extra_measurements is not None:
                for pos in self.initial_extra_measurements:
                    self.measured_values = np.atleast_2d(np.vstack((self.measured_values, self.GroundTruth_field[pos[0], pos[1]])))

        else:

            already_visited = any(np.equal(self.measured_locations, self.position).all(1))

            if already_visited:
                return
            else:
                self.measured_locations = np.atleast_2d(np.vstack((self.measured_locations, self.position)))
                # Add element to y
                measured_val = self.GroundTruth_field[self.position[0], self.position[1]]
                self.measured_values = np.atleast_2d(np.vstack((self.measured_values, measured_val)))

        # Fit and update the gaussian process #
        if update:
            self.GaussianProcess.fit(self.measured_locations[-self.max_distance:,:], self.measured_values[-self.max_distance:,:]) # Max 50 samples #
            self.estimated_model_mean, self.estimated_model_std = self.GaussianProcess.predict(self.possible_positions,
                                                                                               return_std=True)

    def reset(self):

        if self.is_model:  # If we are training with model, get a new value #

            self.GroundTruth.reset_gt()
            self.GroundTruth_field = self.GroundTruth.sample_gt()

        # Reset the initial position of the robot #
        self.position = np.copy(self.initial_position)
        self.trajectory = np.atleast_2d(np.copy(self.initial_position))
        self.distance = 0
        self.step_number = 0
        self.num_of_collisions = 0
        self.accumulated_reward = 0

        # Reupdate gaussian process
        self.measured_locations = None
        self.measured_values = None
        self.measure_and_update_gaussian_process()
        self.previous_mse = self.compute_mse()

        return self.compute_new_state()

    def check_collision(self, action):

        vector = self.action2vector(action)

        intended_position = self.position + vector

        if intended_position[0] >= self.navigation_map.shape[0] or intended_position[0] < 0 or intended_position[1] >= self.navigation_map.shape[1] or intended_position[1] < 0:
            return True

        if self.navigation_map[intended_position[0], intended_position[1]] == 0:
            return True
        else:
            return False

    def step(self, action):

        """ Step function - Process the action and produce a new state & a reward. """

        """ 0 -> -l, 1 -> l for every axis x,y"""

        done = False

        action = np.clip(action, 0.0, 1.0)

        assert action.shape == (2,), "Error in the action shape!"

        # Increase the step number #
        self.step_number += 1


        movement = np.array([(self.max_meas_distance * action[0] + self.min_meas_distance) * np.cos(2*np.pi*action[1]),
                             (self.max_meas_distance * action[0] + self.min_meas_distance) * np.sin(2*np.pi*action[1])])



        next_position = np.floor(self.position + movement).astype(int)

        next_intended_position = np.clip(next_position, a_min = (0,0), a_max = np.array(self.navigation_map.shape)-1)

        path_distance = np.linalg.norm(self.position-next_intended_position)

        self.distance += path_distance


        if self.distance >= self.max_distance:

            limited_distance = np.abs(self.distance-self.max_distance)

            final_position = np.floor(limited_distance * (next_intended_position - self.position)/path_distance).astype(int) + self.position

            done = True

        else:

            final_position = next_intended_position

        self.position = final_position


        # New sample
        self.measure_and_update_gaussian_process(True)
        self.maximum_peak_location = self.measured_locations[np.argmax(self.measured_values)]

        # compute state
        state = self.compute_new_state()

        # Compute reward #
        reward = self.compute_reward(False)

        # Update the MSE for posterior calculation
        self.previous_mse = self.compute_mse()

        # Accumulate reward #
        self.accumulated_reward += reward

        done = self.step_number >= 50 or done

        return state, reward, done, {}

    def compute_reward(self, collision):

        """ Reward function calculation """

        if not collision:
            # Maximum expected regret #
            if self.reward_function == 'regret':
                reward = (- np.max(self.estimated_model_mean) + self.measured_values[-1])[0]
            # Maxmin regret
            elif self.reward_function == 'minimax_regret':
                samples = self.GaussianProcess.sample_y(self.navigable_positions, 10) # Draw 10 samples of the GP
                reward = (- np.max(samples) + self.measured_values[-1])[0]
            # Raw MSE metric
            elif self.reward_function == 'mse':
                reward = -self.compute_mse()
            # Incremental MSE metric
            elif self.reward_function == 'inc_mse':
                reward = np.clip(self.previous_mse - self.compute_mse(), a_min=-1, a_max=1)

        # Collision penalization
        else:
            reward = - self.penalization

        return reward

    def compute_new_state(self):

        state = np.zeros((4,self.navigation_map.shape[0], self.navigation_map.shape[1]), dtype=np.float32)

        # Position #
        for i, pos in enumerate(self.trajectory):
            state[0, pos[0], pos[1]] = i/len(self.trajectory)

        # Navigation Map #
        state[1] = np.copy(self.navigation_map)

        # Gain #
        state[2] = np.copy(self.estimated_model_mean.reshape(self.navigation_map.shape))

        # Uncertainty #
        state[3] = np.copy(self.estimated_model_std.reshape(self.navigation_map.shape))

        return state

    def compute_weighted_mse(self):

        wmse = mean_squared_error(y_true=self.GroundTruth_field[self.navigation_map == 1],
                                  y_pred=self.estimated_model_mean.reshape(self.navigation_map.shape)[self.navigation_map == 1],
                                  sample_weight=self.GroundTruth_field[self.navigation_map == 1]/np.linalg.norm(self.GroundTruth_field[self.navigation_map == 1]))

        return wmse

    def compute_mse(self):

        mse = mean_squared_error(y_true=self.GroundTruth_field[self.navigation_map == 1],
                                 y_pred=self.estimated_model_mean.reshape(self.navigation_map.shape)[self.navigation_map == 1])

        return mse

    def compute_r2(self):

        r2 = r2_score(y_true=self.GroundTruth_field[self.navigation_map == 1],
                      y_pred=self.estimated_model_mean.reshape(self.navigation_map.shape)[self.navigation_map == 1])

        return r2

    def compute_gathered_information(self):

        total_information = np.nansum(self.GroundTruth_field)
        information = np.sum(self.measured_values)/total_information

        return information

    def compute_safe_action(self):
        """ Compute a random safe action """

        # No collisions #
        possible_actions = []

        for a in range(8):
            if not self.check_collision(a):
                possible_actions.append(a)

        return possible_actions

    def compute_real_regret(self):

        real_regret = (- np.nanmax(self.GroundTruth_field) + self.measured_values[-1])[0]

        return real_regret

    def return_accumulated_reward(self, path):

        self.reset()
        done = False
        i = 0
        R = 0
        while done is not True:

            _, r, d, _ = self.step(path[i])
            R += r
            i += 1

        return R,


if __name__ == '__main__':

    navigation_map = np.ones((100,100))
    initial_position = np.array([50, 50])


    plot = True

    env = JumpingPatrollingEnvironment(navigation_map=navigation_map, initial_position=initial_position,
                                       initial_seed=1000001, reward_function='inc_mse', max_meas_distance=40,
                                       min_meas_distance=5)

    s = env.reset()
    R = 0
    Rt = 0
    d = False
    r_vec = [0]
    rt_vec = [0]

    if plot:
        fig, axs = plt.subplots(1, 4)
        colors = [(0.0, 0.3, 0.7), (0.0, 0.6, 0.1)]
        cm = LinearSegmentedColormap.from_list('contamination', colors, N=100)
        axs[1].plot(env.position[1], env.position[0], 'Xr')
        mean = np.copy(s[2])
        mean[env.navigation_map == 0] = np.nan
        axs[0].imshow(mean, cmap=cm)
        axs[1].imshow(s[3], cmap=cm)
        axs[2].imshow(env.GroundTruth_field, cmap=cm)
        axs[3].imshow(s[0], cmap='viridis')
        plt.pause(0.0002)  # In interactive mode, need a small delay to get the plot to appear
        plt.draw()

    while not d:

        s, r, d, _ = env.step(np.random.rand(2))

        r_vec.append(r)

        rt_vec.append(env.compute_mse())

        print(f"Reward: {r}")

        if plot:

            axs[1].plot(env.position[1], env.position[0], 'Xr')
            mean = np.copy(s[2])
            mean[env.navigation_map == 0] = np.nan
            axs[0].imshow(mean,cmap=cm, interpolation = 'None')
            axs[1].imshow(s[3],cmap=cm, interpolation = 'None')
            axs[2].imshow(env.GroundTruth_field,cmap=cm, interpolation = 'bilinear')

            plt.pause(3)  # In interactive mode, need a small delay to get the plot to appear
            plt.draw()

    print("Total Reward: ", R)
    print("R2 score: ", env.compute_r2())
    print("MES: ", env.compute_mse())
    print("Weighted MES: ", env.compute_weighted_mse())
    print("Information gathered: ", env.compute_gathered_information())

    if plot:
        axs[0].plot(env.maximum_peak_location[1]-1, env.maximum_peak_location[0]-1, 'ro')
        index = np.unravel_index(np.nanargmax(env.GroundTruth_field, axis=None), env.GroundTruth_field.shape)
        axs[0].plot(index[1],index[0], 'bs')

        fig2, axs2 = plt.subplots(1, 1)
        axs2.plot(r_vec,'b-o')
        axs2.plot(rt_vec, 'r-o')
        axs2.legend(['Inc. MSE', 'MSE'])

        plt.grid()

        plt.show()

