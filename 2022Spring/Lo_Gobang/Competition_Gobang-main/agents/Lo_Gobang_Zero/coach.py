# -*- coding:utf-8  -*-
# Model: Lo_151_Coach
# Time  : 2022/7
# Author: Loping151
# Algorithm: maybe 151 v4? I don't remember

import random
import numpy as np
from gym.spaces import Discrete


class Coach:
    def __init__(self):
        self.action_space = (Discrete(15), Discrete(15))  # No use

    def act(self, board):
        observation = {'state_map': board.tolist(), 'chess_player_idx': 1}
        # select trainer
        action = my_controller(observation, self.action_space)
        x = action[0].index(1)
        y = action[1].index(1)
        return 15 * x + y


def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    c_map = np.array(observation['state_map']).reshape(15, 15)
    if not np.sum(np.array(c_map)):  # empty map
        return [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    myself = observation['chess_player_idx']
    if myself == 2:  # normalize, I'm 1 and opponent is 2
        for i in range(15):
            for j in range(15):
                if c_map[i, j]:  # not 0
                    c_map[i, j] = 3 - c_map[i, j]  # switch 1, 2
    solution = decision(c_map)
    agent_action = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    agent_action[0][solution[0]] = 1
    agent_action[1][solution[1]] = 1
    return agent_action


def down(p, m):
    return p[0] + m, p[1]


def right(p, m):
    return p[0], p[1] + m


def dr(p, m):
    return p[0] + m, p[1] + m


def dl(p, m):
    return p[0] + m, p[1] - m


# mode 1 for beginning, and 2 for latter
def scan_v2(c_map, mode=1):
    # construct full map with boundary walls
    f_map = np.zeros((19, 19))
    f_map[[0, 1, 17, 18]] = np.ones((4, 19)) * -1
    f_map[:, [0, 1, 17, 18]] = np.ones((19, 4)) * -1
    f_map[2:17, 2:17] = c_map
    # score map
    score = np.zeros((19, 19))
    # scan four directions
    # down
    for i in range(13):
        for j in range(2, 17):
            index = f_map[i:i + 7, j]
            judge_v2(loc=(i, j), sec=index, score=score, move=down, mode=mode)

    # right
    for i in range(2, 17):
        for j in range(13):
            index = f_map[i, j:j + 7]
            judge_v2(loc=(i, j), sec=index, score=score, move=right, mode=mode)

    # down right
    for i in range(-10, 11):
        line = np.diagonal(f_map, offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            if i < 0:
                judge_v2(loc=(j - i, j), sec=index, score=score, move=dr, mode=mode)
            else:
                judge_v2(loc=(j, i + j), sec=index, score=score, move=dr, mode=mode)

    # up right
    for i in range(-10, 11):
        line = np.diagonal(np.fliplr(f_map), offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            if i < 0:
                judge_v2(loc=(j - i, 18 - j), sec=index, score=score, move=dl, mode=mode)
            else:
                judge_v2(loc=(j, 18 - i - j), sec=index, score=score, move=dl, mode=mode)

    # in case all zero score matrix
    if mode == 2:
        for i in range(2, 17):
            for j in range(2, 17):
                if not score[i, j] and not f_map[i, j]:
                    score[i, j] = 0.5

    return score


# best position
def decision(c_map):
    score = scan_v2(c_map)
    if np.sum(score) == 0:
        score = scan_v2(c_map, mode=2)
    max_s = np.max(score)
    index = []
    for i in range(19):
        for j in range(19):
            if score[i, j] == max_s:
                index.append((i - 2, j - 2))
    random.shuffle(index)
    return index[0]


# judge_v2_v2_v2 block for each player
def block(pos, player):
    if player == 'my':
        return pos != 0 and pos != 1
    if player == 'op':
        return pos != 0 and pos != 2


# current location, input sequence, score matrix, move type
def judge_v2(loc, sec, score, move, mode=1):
    sec = list(sec)  # should be list to compare
    # it would be better to sort these ifs according to frequency, which could make it faster. but I don't have time

    # mode 2 for score is 0 when mode is 1
    if mode == 2:
        if not block(sec[1], 'op') and not block(sec[2], 'op') and not block(sec[3], 'op') \
                and not block(sec[4], 'op') and not block(sec[5], 'op'):
            score[move(loc, 1)] += 2
            score[move(loc, 2)] += 2
            score[move(loc, 3)] += 2
            score[move(loc, 4)] += 2
            score[move(loc, 5)] += 2
            return
        if sec[1:-2] == [1, 1, 0, 0] and block(sec[0], 'my'):
            score[move(loc, 3)] += 2
            score[move(loc, 4)] += 2
            return
        if sec[1:-2] == [0, 0, 1, 1] and block(sec[5], 'my'):
            score[move(loc, 1)] += 2
            score[move(loc, 2)] += 2
            return
        if sec[1:-2] == [2, 2, 0, 0] and block(sec[0], 'op'):
            score[move(loc, 3)] += 3
            score[move(loc, 4)] += 3
            return
        if sec[1:-2] == [0, 0, 2, 2] and block(sec[5], 'op'):
            score[move(loc, 1)] += 3
            score[move(loc, 2)] += 3
            return
        if sec[1:-1] == [1, 0, 0, 0, 0] and block(sec[0], 'my'):
            score[move(loc, 3)] += 2
            score[move(loc, 4)] += 2
            return
        if sec[1:-1] == [0, 0, 0, 0, 1] and block(sec[6], 'my'):
            score[move(loc, 1)] += 2
            score[move(loc, 2)] += 2
            return
        if sec[1:-1] == [2, 0, 0, 0, 0] and block(sec[0], 'op'):
            score[move(loc, 3)] += 3
            score[move(loc, 4)] += 3
            return
        if sec[1:-1] == [0, 0, 0, 0, 2] and block(sec[6], 'op'):
            score[move(loc, 1)] += 3
            score[move(loc, 2)] += 3
            return
        if sec[2:-2] == [0, 1, 0]:
            score[move(loc, 2)] += 1.1
            score[move(loc, 4)] += 1.1
            return
        if sec[2:-2] == [0, 2, 0]:
            score[move(loc, 2)] += 2
            score[move(loc, 4)] += 2
            return
        if sec[-4:-1] == [0, 0, 1] and block(sec[4], 'my'):
            score[move(loc, 1)] += 1.5
            score[move(loc, 2)] += 1.5
            return
        if sec[-4:-1] == [0, 0, 2] and block(sec[4], 'op'):
            score[move(loc, 1)] += 2
            score[move(loc, 2)] += 2
            return
        if sec[1:4] == [1, 0, 0] and block(sec[0], 'my'):
            score[move(loc, 2)] += 1.5
            score[move(loc, 3)] += 1.5
            return
        if sec[1:4] == [2, 0, 0] and block(sec[0], 'op'):
            score[move(loc, 2)] += 2
            score[move(loc, 3)] += 2
            return
        return

    # simple cases
    if sec[1:-1] == [0, 0, 0, 0, 0]:
        return
    if sec[:-1] == [0, 0, 1, 1, 0, 0]:
        score[move(loc, 1)] += 20
        score[move(loc, 4)] += 20
        score[move(loc, 0)] += 17
        score[move(loc, 5)] += 17
        return
    if sec[:-1] == [0, 0, 2, 2, 0, 0]:
        score[move(loc, 1)] += 20
        score[move(loc, 4)] += 20
        score[move(loc, 0)] += 17
        score[move(loc, 5)] += 17
        return
    if sec[1:-1] == [0, 1, 0, 1, 0]:
        score[move(loc, 3)] += 20
        score[move(loc, 1)] += 10
        score[move(loc, 5)] += 10
        return
    if sec[1:-1] == [0, 2, 0, 2, 0]:
        score[move(loc, 3)] += 20
        score[move(loc, 1)] += 5
        score[move(loc, 5)] += 5
        return
    if sec[1:-1] == [0, 0, 1, 0, 0]:
        score[move(loc, 1)] += 5
        score[move(loc, 2)] += 3
        score[move(loc, 4)] += 3
        score[move(loc, 5)] += 5
        return
    if sec[1:-1] == [0, 0, 2, 0, 0]:
        score[move(loc, 2)] += 5
        score[move(loc, 4)] += 5
        return

    # link 3 group
    if sec == [0, 0, 1, 1, 1, 0, 0]:  # all live 3 my
        score[move(loc, 1)] += 100
        score[move(loc, 5)] += 100
        return
    if sec == [0, 0, 2, 2, 2, 0, 0]:  # all live 3 op
        score[move(loc, 1)] += 100
        score[move(loc, 5)] += 100
        score[move(loc, 0)] += 20
        score[move(loc, 6)] += 20
        return
    if block(sec[0], 'my') and sec[1:] == [0, 1, 1, 1, 0, 0]:  # single live 3 my
        score[move(loc, 5)] += 100
        score[move(loc, 6)] += 85
        return
    if block(sec[-1], 'my') and sec[:-1] == [0, 0, 1, 1, 1, 0]:  # single live 3 my r
        score[move(loc, 1)] += 100
        score[move(loc, 0)] += 85
        return
    if block(sec[0], 'op') and sec[1:] == [0, 2, 2, 2, 0, 0]:  # single live 3 op
        score[move(loc, 5)] += 100
        score[move(loc, 1)] += 20
        return
    if block(sec[-1], 'op') and sec[:-1] == [0, 0, 2, 2, 2, 0]:  # single live 3 op r
        score[move(loc, 1)] += 100
        score[move(loc, 5)] += 20
        return
    if block(sec[0], 'my') and block(sec[-1], 'my') and sec[1: -1] == [0, 1, 1, 1, 0]:  # double live 3 my
        score[move(loc, 1)] += 10
        score[move(loc, 5)] += 10
        return
    if block(sec[0], 'op') and block(sec[-1], 'op') and sec[1: -1] == [0, 2, 2, 2, 0]:  # double live 3 op
        score[move(loc, 1)] += 20
        score[move(loc, 5)] += 20
        return
    if block(sec[1], 'my') and sec[2:] == [1, 1, 1, 0, 0]:  # block 3 my
        score[move(loc, 5)] += 25
        score[move(loc, 6)] += 20
        return
    if block(sec[-2], 'my') and sec[:-2] == [0, 0, 1, 1, 1]:  # block 3 my r
        score[move(loc, 0)] += 20
        score[move(loc, 1)] += 25
        return
    if block(sec[1], 'op') and sec[2:] == [2, 2, 2, 0, 0]:  # block 3 op
        score[move(loc, 5)] += 15
        score[move(loc, 6)] += 25
        return
    if block(sec[-2], 'op') and sec[:-2] == [0, 0, 2, 2, 2]:  # block 3 op r
        score[move(loc, 0)] += 25
        score[move(loc, 1)] += 15
        return
    if sec[1:-1] == [1, 1, 1, 0, 1]:  # jump 3 my
        score[move(loc, 4)] = 5000
        return
    if sec[1:-1] == [1, 0, 1, 1, 1]:  # jump 3 my r
        score[move(loc, 2)] += 5000
        return
    if sec[1:-1] == [2, 2, 2, 0, 2]:  # jump 3 op
        score[move(loc, 4)] += 1000
        return
    if sec[1:-1] == [2, 0, 2, 2, 2]:  # jump 3 op r
        score[move(loc, 2)] += 1000
        return

    # link 4 group
    if sec[1:-1] == [1, 1, 1, 1, 0]:
        score[move(loc, 5)] += 5000
        return
    if sec[1:-1] == [0, 1, 1, 1, 1]:
        score[move(loc, 1)] += 5000
        return
    if sec[1:-1] == [2, 2, 2, 2, 0]:
        score[move(loc, 5)] += 1000
        return
    if sec[1:-1] == [0, 2, 2, 2, 2]:
        score[move(loc, 1)] += 1000
        return
    if sec[1:-1] == [1, 1, 0, 1, 1]:
        score[move(loc, 3)] += 5000
        return
    if sec[1:-1] == [2, 2, 0, 2, 2]:
        score[move(loc, 3)] += 1000
        return

    # dis 3 group
    if sec[:-1] == [0, 1, 1, 0, 1, 0]:  # all live d my
        score[move(loc, 3)] += 100
        return
    if sec[:-1] == [0, 1, 0, 1, 1, 0]:  # all live d my r
        score[move(loc, 2)] += 100
        return
    if sec[:-1] == [0, 2, 2, 0, 2, 0]:  # all live d op
        score[move(loc, 3)] += 100
        score[move(loc, 0)] += 20
        score[move(loc, 5)] += 20
        return
    if sec[:-1] == [0, 2, 0, 2, 2, 0]:  # all live d op r
        score[move(loc, 2)] += 100
        score[move(loc, 0)] += 20
        score[move(loc, 5)] += 20
        return
    if block(sec[0], 'my') and sec[1:-1] == [1, 1, 0, 1, 0]:  # block d my
        score[move(loc, 3)] += 10
        score[move(loc, 5)] += 10
        return
    if block(sec[6], 'my') and sec[1:-1] == [0, 1, 0, 1, 1]:  # block d my r
        score[move(loc, 1)] += 10
        score[move(loc, 3)] += 10
        return
    if block(sec[0], 'op') and sec[1:-1] == [2, 2, 0, 2, 0]:  # block d op
        score[move(loc, 3)] += 10
        score[move(loc, 5)] += 15
        return
    if block(sec[6], 'op') and sec[1:-1] == [0, 2, 0, 2, 2]:  # block d op r
        score[move(loc, 1)] += 15
        score[move(loc, 3)] += 10
        return
    if block(sec[6], 'my') and sec[1:-1] == [0, 1, 1, 0, 1]:  # block d2 my
        score[move(loc, 1)] += 10
        score[move(loc, 4)] += 10
        return
    if block(sec[0], 'my') and sec[1:-1] == [1, 0, 1, 1, 0]:  # block d2 my r
        score[move(loc, 2)] += 10
        score[move(loc, 5)] += 10
        return
    if block(sec[6], 'op') and sec[1:-1] == [0, 2, 2, 0, 2]:  # block d2 op
        score[move(loc, 1)] += 15
        score[move(loc, 4)] += 10
        return
    if block(sec[0], 'op') and sec[1:-1] == [2, 0, 2, 2, 0]:  # block d2 op r
        score[move(loc, 2)] += 10
        score[move(loc, 5)] += 15
        return
