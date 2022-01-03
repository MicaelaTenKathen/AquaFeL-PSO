from bayes_opt import BayesianOptimization
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
import matplotlib.pyplot as plt


from PSO.pso_function import PSOEnvironment
import numpy as np

number = 0

pbounds = {'c1': (0, 4), 'c2': (0, 4), 'c3': (0, 4), 'c4': (0, 4)}

resolution = 1
xs = 100
ys = 150
# navigation_map = np.genfromtxt('Image/ypacarai_map_bigger.csv')


def model_psogp(c1, c2, c3, c4):
    action = np.array([c1, c2, c3, c4])

    initial_position = np.array([[0, 0],
                                 [8, 56],
                                 [37, 16],
                                 [78, 81],
                                 [74, 124]])

    method = 0
    pso = PSOEnvironment(resolution, ys, method, initial_seed=1, initial_position=initial_position,
                         reward_function='inc_mse')

    last_mse = []

    for i in range(200):

        done = False
        state = pso.reset()
        R_vec = []

        # Main part of the code

        while not done:
            state, reward, done, dic = pso.step(action)

            R_vec.append(-reward)

        MSE_data = np.array(pso.MSE_value())
        last_mse.append(MSE_data[-1])

    mean_total = np.mean(np.array(last_mse))
    MSE = mean_total * -1

    return MSE


optimizer = BayesianOptimization(
            f=model_psogp,
            pbounds=pbounds,
            random_state=1)

optimizer.maximize(init_points=10, n_iter=20, acq='ei')

print(optimizer.max)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
