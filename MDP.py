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
        # 0'review', 1'deny', 2'approve', 3'evidence'
        self.actions = [0, 1, 2, 3]

        self.num_states = 5
        self.num_actions = 4
        self.T = np.zeros((self.num_states, self.num_states, self.num_actions))
        self.R = np.zeros((self.num_states, self.num_states, self.num_actions))
        # next curr act
        self.T[1, 0, 1] = self.T[2, 0, 2] = 1
        self.T[3, 0, 3] = self.T[4, 0, 0] = 1
        self.T[1, 3, 1] = self.T[2, 3, 2] = self.T[4, 3, 0] = 1
        self.T[1, 4, 1] = self.T[2, 4, 2] = 1

        self.R[3, 0, 3] = reward_calculation(x, r, a, 0, 3, if_travel)
        self.R[4, 0, 0] = reward_calculation(x, r, a, 0, 4, if_travel)
        self.R[2, 0, 2] = reward_calculation(x, r, a, 0, 2, if_travel)
        self.R[2, 3, 2] = reward_calculation(x, r, a, 3, 2, if_travel)
        self.R[4, 3, 0] = reward_calculation(x, r, a, 3, 4, if_travel)
        self.R[1, 3, 1] = reward_calculation(x, r, a, 3, 1, if_travel)
        self.R[2, 4, 2] = reward_calculation(x, r, a, 4, 2, if_travel)
        self.R[1, 4, 1] = reward_calculation(x, r, a, 4, 1, if_travel)

        self.P_0 = np.array([1, 0, 0, 0, 0])
        # env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=P_0, p=T, r=R, disable_env_checker=True)
        # observation, info = env.reset()
        self.env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=self.P_0, p=self.T, r=self.R, disable_env_checker=True)
        self.observation, self.info = self.env.reset()
        self.terminated = False
        self.act = None
        self.path = []

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
                self.act = None
            else:
                self.act = random.choice(init)

            self.observation, reward, self.terminated, truncated, self.info = self.env.step(self.act)
            self.path.append(self.observation)

            if self.terminated:
                return self.path
            else:
                return self.path

    def q_learning(self):
        env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=self.P_0, p=self.T, r=self.R, disable_env_checker=True)
        observation, info = env.reset()
        if_terminated = False
        count_episode = 0
        Q = np.zeros((self.num_states, self.num_actions))

        gamma = 0.9
        epsilon = 0.9
        eta = 1
        policy = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1}
        for i in range(10000):
            while not if_terminated:
                ngh = self.valid_neighbors(observation)
                if round(epsilon, 1) == 0.5:
                    action = min(ngh.values())

                elif random.random() < epsilon:
                    action = random.choice(list(ngh.values()))
                else:
                    max_q_value = -float('inf')
                    max_action = -1

                    for j, v in ngh.items():
                        q_value = Q[observation, v]
                        if q_value > max_q_value:
                            max_q_value = q_value
                            max_action = v
                    action = max_action
                last = observation
                observation, reward, terminated, truncated, info = env.step(action)

                if reward == None:
                    reward = 0
                else:
                    reward = float(reward)

                # update Q, use Q formula
                Q[last, action] = ((1 - eta) * Q[last, action]
                                   + eta * (reward + gamma * np.max(Q[observation])))

                if_terminated = terminated
                if if_terminated:
                    observation, info = env.reset()
                    count_episode += 1
                    eta = 1 / (1 + count_episode)
                    epsilon *= 0.9999

        for state in range(5):
            max_value= np.argmax(Q[state, :])
            if max_value==0:
                policy[state] = random.choice([1, 2])
            else:
                policy[state]=max_value
        policy[1]=-1
        policy[2]=-1
        # print(self.T)
        # print(self.R)
        # print(policy)
        # print(Q)
        # print(self.generate_optimal_path(policy))
        return self.generate_optimal_path(policy)

    def valid_neighbors(self, curr):
        ngh = {}

        for i in range(5):
            for k in range(4):
                if self.T[i, curr, k] != 0.0 and i != curr:
                    ngh[i] = k
        return ngh

    def generate_optimal_path(self, policy):
        current_state = 0
        optimal_path = [current_state]

        while policy[current_state] !=-1:
            current_state = policy[current_state]
            optimal_path.append(current_state)

        return optimal_path
        # print(optimal_path)

mdp = ReimbursementMDP(0.2, 0.1, 0.2, True)

# print(mdp.iterate())
# print(mdp.iterate())
# print(mdp.iterate())
print(mdp.q_learning())
