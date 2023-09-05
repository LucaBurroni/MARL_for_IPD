import gym
from gym import spaces
import numpy as np

class IPDEnv(gym.Env):
    def __init__(self, max_history_length=16, max_rounds=24, payoff_matrix = np.array([[[3, 3], [0, 5]], [[5, 0], [1, 1]]])):
        super(IPDEnv, self).__init__()

        # define action space (Cooperate or Defect)
        self.action_space = spaces.Discrete(2)

        # define observation space (action history)
        self.max_history_length = max_history_length
        self.observation_space = spaces.Box(low=0, high=1, shape=[max_history_length * 2])  # observation space with shape [max_history_length * 2]
        self.payoff_matrix = payoff_matrix

        # initialize game parameters
        self.rounds_played = 0
        self.max_rounds = max_rounds
        self.players_actions = [[], []]  # history of all actions for both players
        self.scores = [0, 0]  # cumulative scores for both players

    def reset(self):
        # reset game parameters
        self.rounds_played = 0
        self.players_actions = [[], []]
        self.scores = [0, 0]

        # return initial observation (empty history)
        return self.encode_observation(), {}

    def step(self, actions):
        # check if the number of actions matches the number of players
        assert len(actions) == 2, "There should be two actions, one for each player."

        # record players' actions
        self.players_actions[0].append(actions[0])
        self.players_actions[1].append(actions[1])

        # update scores based on the payoff matrix 
        self.scores[0] += self.payoff_matrix[actions[0]][actions[1]][0]
        self.scores[1] += self.payoff_matrix[actions[0]][actions[1]][1]

	# calculate the individual rewards for each agent
        reward_player1 = self.payoff_matrix[actions[0]][actions[1]][0]
        reward_player2 = self.payoff_matrix[actions[0]][actions[1]][1]

        # encode the rewards into a single value
        combined_reward = reward_player1 + reward_player2

        # increment round counter
        self.rounds_played += 1

        # check if the game is over
        truncated = self.rounds_played >= self.max_rounds

        # put reward_player2 in info to unpack the two values once received
        info = {
        	'reward_player2': reward_player2
        }

        # return the next observation, reward, terminated and truncated flags, and additional info
        return self.encode_observation(), combined_reward, False, truncated, info

    def encode_observation(self):
    	# encode the history of actions
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        for i in range(self.max_history_length):
            if i < len(self.players_actions[0]) and i < len(self.players_actions[1]):
                player1_action = self.players_actions[0][-1 - i]
                player2_action = self.players_actions[1][-1 - i]
                obs[i * 2] = player1_action
                obs[i * 2 + 1] = player2_action
            else:
                break

        return obs


    def render(self, mode='human'):
        pass

    def close(self):
        pass