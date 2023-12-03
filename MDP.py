import random
import matrix_mdp
import gymnasium as gym
import numpy as np
from rewardCalc import reward_calculation


class ReimbursementMDP(gym.Env):
    def __init__(self, x, r, a, if_travel):
        # Define states and actions 0'start', 1'deny', 2'approve',
        # 3'require_more_evidence', 4'submit_for_review'
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

        self.R[3, 0, 0] = reward_calculation(x, r, a, 0, 3, if_travel)
        self.R[4, 0, 0] = reward_calculation(x, r, a, 0, 4, if_travel)
        self.R[2, 0, 0] = reward_calculation(x, r, a, 0, 2, if_travel)
        self.R[2, 3, 3] = reward_calculation(x, r, a, 3, 2, if_travel)
        self.R[2, 4, 3] = reward_calculation(x, r, a, 4, 2, if_travel)

        self.P_0 = np.array([1, 0, 0, 0, 0])
        # env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=P_0, p=T, r=R, disable_env_checker=True)
        # observation, info = env.reset()
        self.env= gym.make('matrix_mdp/MatrixMDP-v0', p_0=self.P_0, p=self.T, r=self.R, disable_env_checker=True)
        self.observation, self.info = self.env.reset()
        self.terminated = False
        self.act=None
    def iterate(self):
        if not self.terminated:
            init = []

            # choose random valid action from state
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    if self.T[s][self.observation][a] > 0:
                        init.append(a)
            # choose action from curr state
            if len(init) == 1:
                self.act = init[0]
            elif len(init) == 0:
                self.act=None
            else:
                self.act = random.choice(init)
            self.observation, reward, self.terminated, truncated, self.info = self.env.step(self.act)

            if self.terminated:
                return self.observation
            else:
                return self.act


mdp = ReimbursementMDP(0.2, 0.1, 0.2, True)

print(mdp.iterate())
print(mdp.iterate())
print(mdp.iterate())
print(mdp.iterate())
print(mdp.iterate())
