import random

import gymnasium as gym
import numpy as np


class ReimbursementMDP(gym.Env):
    def __init__(self):
        # Define states and actions 0'start', 1'deny', 2'approve', 3'require_more_evidence', 4'submit_for_review'
        self.states = [0, 1, 2, 3, 4]
        # 0'submit_for_review', 1'deny', 2'approve', 3'request_more_evidence'
        self.actions = [0, 1, 2, 3]

        self.num_states = 5
        self.num_actions = 4
        self.T = np.zeros((self.num_states, self.num_states, self.num_actions))
        self.R = np.zeros((self.num_states, self.num_states, self.num_actions))
        # next curr act
        self.T[1, 0, 0] = self.T[2, 0, 0] = 0.4
        self.T[3, 0, 0] = self.T[4, 0, 0] = 0.1
        self.T[1, 3, 3] = self.T[2, 3, 3] = 0.5
        self.T[1, 4, 3] = self.T[2, 4, 3] = 0.5

        self.R[3, 0, 0] = self.R[4, 0, 0] = -100
        self.R[2, 0, 0] = 10
        self.R[2, 3, 3] = 10
        self.R[2, 4, 3] = 10
        self.P_0 = np.array([1, 0, 0, 0, 0])
        # env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=P_0, p=T, r=R, disable_env_checker=True)
        # observation, info = env.reset()

    def iterate(self):
        env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=self.P_0, p=self.T, r=self.R, disable_env_checker=True)
        observation, info = env.reset()

        terminated = False
        while not terminated:
            init = []

            # choose random valid action from state
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    if self.T[s][observation][a] > 0:
                        init.append(a)
            # choose action from curr state
            if len(init) == 1:
                act = init[0]
            elif len(init) == 0:
                break
            else:
                act = random.choice(init)

            observation, reward, terminated, truncated, info = env.step(act)

            if terminated:
                return observation, reward


mdp = ReimbursementMDP()

print(mdp.iterate())
