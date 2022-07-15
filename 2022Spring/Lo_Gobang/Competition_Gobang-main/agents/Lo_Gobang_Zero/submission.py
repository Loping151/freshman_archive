# -*- coding:utf-8  -*-
# Model: Lo_151_Coach
# Time  : 2022/7/7
# Author: Loping151
# Algorithm: DQN with tf2


from Logo_net import Lo_DQN
import numpy as np

agent = Lo_DQN()
agent.load_model(load_name='Lo_zero_base.h5')


def my_controller(observation, action_space, is_act_continuous=False):
    board = np.array(observation['state_map']).reshape((15, 15))
    if observation['chess_player_idx'] == 2:
        board = board_switch(board)
    action = agent.act(board, explore=False)
    agent_action = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    agent_action[0][action // 15] = 1
    agent_action[1][action % 15] = 1
    return agent_action


def board_switch(b):
    for i in range(15):
        for j in range(15):
            if b[i, j]:  # not 0
                b[i, j] = 3 - b[i, j]  # switch 1, 2
    return b
