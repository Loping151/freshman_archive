# -*- coding:utf-8  -*-
# Model: Lo_151_alpha
# Time  : 2022/7/9
# Author: Loping151
# Algorithm: based on 151_v4, added search


import numpy as np
import random


def board_switch(b):
    for i in range(15):
        for j in range(15):
            if b[i, j]:  # not 0
                b[i, j] = 3 - b[i, j]  # switch 1, 2
    return b


def my_controller(observation, action_space, is_act_continuous=False):
    c_map = np.array(observation['state_map']).reshape(15, 15)
    if not np.sum(np.array(c_map)):  # empty map
        return [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    myself = observation['chess_player_idx']
    if myself == 2:  # normalize, I'm 1 and opponent is 2
        c_map = board_switch(c_map)
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
def scan_v2(c_map, mode=1, add=True):
    # construct full map with boundary walls
    f_map = np.zeros((19, 19))
    f_map[[0, 1, 17, 18]] = np.ones((4, 19)) * -1
    f_map[:, [0, 1, 17, 18]] = np.ones((19, 4)) * -1
    f_map[2:17, 2:17] = c_map
    my_score = np.zeros((19, 19))
    if add:
        op_score = my_score
    else:
        op_score = my_score.copy()
    # scan four directions
    # down
    for i in range(13):
        for j in range(2, 17):
            index = f_map[i:i + 7, j]
            judge_v2(loc=(i, j), sec=index, my_score=my_score, op_score=op_score, move=down, mode=mode)

    # right
    for i in range(2, 17):
        for j in range(13):
            index = f_map[i, j:j + 7]
            judge_v2(loc=(i, j), sec=index, my_score=my_score, op_score=op_score, move=right, mode=mode)

    # down right
    for i in range(-10, 11):
        line = np.diagonal(f_map, offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            if i < 0:
                judge_v2(loc=(j - i, j), sec=index, my_score=my_score, op_score=op_score, move=dr, mode=mode)
            else:
                judge_v2(loc=(j, i + j), sec=index, my_score=my_score, op_score=op_score, move=dr, mode=mode)

    # up right
    for i in range(-10, 11):
        line = np.diagonal(np.fliplr(f_map), offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            if i < 0:
                judge_v2(loc=(j - i, 18 - j), sec=index, my_score=my_score, op_score=op_score, move=dl, mode=mode)
            else:
                judge_v2(loc=(j, 18 - i - j), sec=index, my_score=my_score, op_score=op_score, move=dl, mode=mode)

    # in case all zero score matrix
    if mode == 2:
        for i in range(2, 17):
            for j in range(2, 17):
                if not my_score[i, j] and not f_map[i, j]:
                    my_score[i, j] = 0.01

    if add:
        return my_score
    else:
        return my_score, op_score


# best position
def decision(c_map):
    return search(c_map,
                  depth=4)  # you can change depth here. depth is round, not step. n depth sees 2n+1 steps in the future


def search(c_map, depth=3):
    c_map, score = _search([[c_map, [None]]], init=True)
    act = np.argmax(score)
    for _ in range(depth):
        c_map = update_op(c_map)  # This is optional, choose only max score for opponent.
        c_map = _search(c_map)
    m_s = 0
    best_a = 0
    if not len(c_map):
        return [act // 19 - 2, act % 19 - 2]
    for i in range(len(c_map)):
        m = c_map[i][0]
        score = np.max(scan_v2(board_switch(m.copy())))
        if score > m_s:
            best_a = i
            m_s = score
    return c_map[best_a][1]


def _search(c_maps, init=False):
    maps = []
    score = None
    for m in c_maps:
        ia = m[1]
        m = m[0]
        score = scan_v2(m)
        if np.sum(score) == 0:
            score = scan_v2(m, mode=2)
        n_act = np.argpartition(-score.flatten(), 1)[:2]
        for a in n_act:
            n_m = m.copy()
            n_m[a // 19 - 2, a % 19 - 2] = 1
            if init:
                maps.append([board_switch(n_m), [a // 19 - 2, a % 19 - 2]])
            else:
                maps.append([board_switch(n_m), ia])
    if init:
        return maps, score
    return maps


def update_op(c_maps):
    maps = []
    for _ in range(len(c_maps)):
        m = c_maps[_][0]
        scores = scan_v2(m, add=False)
        if np.sum(scores) == 0:
            scores = scan_v2(m, mode=2, add=False)
        max_s = (np.max(scores[0]), np.max(scores[1]))
        if max_s[0] >= 100 and max_s[1] <= max_s[0]:
            continue
        index = []
        score = scores[0] + scores[1]
        max_s = np.max(score)
        for i in range(19):
            for j in range(19):
                if score[i, j] == max_s:
                    index.append((i - 2, j - 2))
        for c in range(len(index)):
            if c > 2:
                break
            n_m = m.copy()
            n_m[index[c]] = 1
            maps.append([board_switch(n_m), c_maps[_][1]])
    return maps


# judge_v2_v2_v2 block for each player
def block(pos, player):
    if player == 'my':
        return pos != 0 and pos != 1
    if player == 'op':
        return pos != 0 and pos != 2


# current location, input sequence, score matrix, move type
def judge_v2(loc, sec, my_score, op_score, move, mode=1):
    sec = sec.tolist()  # should be list to compare
    # it would be better to sort these ifs according to frequency, which could make it faster. but I don't have time

    # mode 2 for score is 0 when mode is 1
    if mode == 2:
        if not block(sec[1], 'op') and not block(sec[2], 'op') and not block(sec[3], 'op') \
                and not block(sec[4], 'op') and not block(sec[5], 'op'):
            op_score[move(loc, 1)] += 1
            op_score[move(loc, 2)] += 1
            op_score[move(loc, 3)] += 1
            op_score[move(loc, 4)] += 1
            op_score[move(loc, 5)] += 1
            return
        if sec[1:-2] == [1, 1, 0, 0] and block(sec[0], 'my'):
            my_score[move(loc, 3)] += 2
            my_score[move(loc, 4)] += 2
            return
        if sec[1:-2] == [0, 0, 1, 1] and block(sec[5], 'my'):
            my_score[move(loc, 1)] += 2
            my_score[move(loc, 2)] += 2
            return
        if sec[1:-2] == [2, 2, 0, 0] and block(sec[0], 'op'):
            op_score[move(loc, 3)] += 3
            op_score[move(loc, 4)] += 5.5
            return
        if sec[1:-2] == [0, 0, 2, 2] and block(sec[5], 'op'):
            op_score[move(loc, 1)] += 5.5
            op_score[move(loc, 2)] += 3
            return
        if sec[1:-1] == [1, 0, 0, 0, 0] and block(sec[0], 'my'):
            my_score[move(loc, 5)] += 2
            my_score[move(loc, 2)] += 1
            my_score[move(loc, 3)] += 1
            my_score[move(loc, 4)] += 2
            return
        if sec[1:-1] == [0, 0, 0, 0, 1] and block(sec[6], 'my'):
            my_score[move(loc, 1)] += 1
            my_score[move(loc, 2)] += 1
            my_score[move(loc, 3)] += 2
            my_score[move(loc, 4)] += 2
            return
        if sec[1:-1] == [2, 0, 0, 0, 0] and block(sec[0], 'op'):
            op_score[move(loc, 3)] += 3
            op_score[move(loc, 4)] += 3
            return
        if sec[1:-1] == [0, 0, 0, 0, 2] and block(sec[6], 'op'):
            op_score[move(loc, 1)] += 2.5
            op_score[move(loc, 2)] += 2.5
            return
        if sec[2:-2] == [0, 1, 0]:
            my_score[move(loc, 2)] += 1
            my_score[move(loc, 4)] += 1
            return
        if sec[2:-2] == [0, 2, 0]:
            op_score[move(loc, 2)] += 2.5
            op_score[move(loc, 4)] += 2.5
            return
        if sec[-4:-1] == [0, 0, 1] and block(sec[6], 'my'):
            my_score[move(loc, 3)] += 1
            my_score[move(loc, 4)] += 1
            return
        if sec[-4:-1] == [0, 0, 2] and block(sec[6], 'op'):
            op_score[move(loc, 3)] += 2.5
            op_score[move(loc, 4)] += 2.5
            return
        if sec[1:4] == [1, 0, 0] and block(sec[0], 'my'):
            my_score[move(loc, 2)] += 1
            my_score[move(loc, 3)] += 1
            return
        if sec[1:4] == [2, 0, 0] and block(sec[0], 'op'):
            op_score[move(loc, 2)] += 2.5
            op_score[move(loc, 3)] += 2.5
            return
        return

    # simple cases
    if sec[1:-1] == [0, 0, 0, 0, 0]:
        return
    if sec[1:-1] == [1, 1, 1, 1, 1]:
        my_score[9, 9] += 1e6
        return
    if sec[1:-1] == [2, 2, 2, 2, 2]:
        op_score[9, 9] += 1e6
        return
    if sec[:-1] == [0, 0, 1, 1, 0, 0]:
        my_score[move(loc, 1)] += 20
        my_score[move(loc, 4)] += 20
        my_score[move(loc, 0)] += 17
        my_score[move(loc, 5)] += 17
        return
    if sec[:-1] == [0, 0, 2, 2, 0, 0]:
        op_score[move(loc, 1)] += 20
        op_score[move(loc, 4)] += 20
        op_score[move(loc, 0)] += 17
        op_score[move(loc, 5)] += 17
        return
    if sec[1:-1] == [0, 1, 0, 1, 0]:
        my_score[move(loc, 3)] += 20
        my_score[move(loc, 1)] += 10
        my_score[move(loc, 5)] += 10
        return
    if sec[1:-1] == [0, 2, 0, 2, 0]:
        op_score[move(loc, 3)] += 20
        op_score[move(loc, 1)] += 10
        op_score[move(loc, 5)] += 10
        return
    if sec[1:-1] == [0, 0, 1, 0, 0]:
        my_score[move(loc, 1)] += 5
        my_score[move(loc, 2)] += 3
        my_score[move(loc, 4)] += 3
        my_score[move(loc, 5)] += 5
        return
    if sec[1:-1] == [0, 0, 2, 0, 0]:
        op_score[move(loc, 2)] += 5
        op_score[move(loc, 4)] += 5
        return

    # link 3 group
    if sec == [0, 0, 1, 1, 1, 0, 0]:  # all live 3 my
        my_score[move(loc, 1)] += 100
        my_score[move(loc, 5)] += 100
        return
    if sec == [0, 0, 2, 2, 2, 0, 0]:  # all live 3 op
        op_score[move(loc, 1)] += 100
        op_score[move(loc, 5)] += 100
        op_score[move(loc, 0)] += 20
        op_score[move(loc, 6)] += 20
        return
    if block(sec[0], 'my') and sec[1:] == [0, 1, 1, 1, 0, 0]:  # single live 3 my
        my_score[move(loc, 5)] += 100
        my_score[move(loc, 6)] += 85
        return
    if block(sec[-1], 'my') and sec[:-1] == [0, 0, 1, 1, 1, 0]:  # single live 3 my r
        my_score[move(loc, 1)] += 100
        my_score[move(loc, 0)] += 85
        return
    if block(sec[0], 'op') and sec[1:] == [0, 2, 2, 2, 0, 0]:  # single live 3 op
        op_score[move(loc, 5)] += 100
        op_score[move(loc, 1)] += 20
        return
    if block(sec[-1], 'op') and sec[:-1] == [0, 0, 2, 2, 2, 0]:  # single live 3 op r
        op_score[move(loc, 1)] += 100
        op_score[move(loc, 5)] += 20
        return
    if block(sec[0], 'my') and block(sec[-1], 'my') and sec[1: -1] == [0, 1, 1, 1, 0]:  # double live 3 my
        my_score[move(loc, 1)] += 10
        my_score[move(loc, 5)] += 10
        return
    if block(sec[0], 'op') and block(sec[-1], 'op') and sec[1: -1] == [0, 2, 2, 2, 0]:  # double live 3 op
        op_score[move(loc, 1)] += 20
        op_score[move(loc, 5)] += 20
        return
    if block(sec[1], 'my') and sec[2:] == [1, 1, 1, 0, 0]:  # block 3 my
        my_score[move(loc, 5)] += 25
        my_score[move(loc, 6)] += 20
        return
    if block(sec[-2], 'my') and sec[:-2] == [0, 0, 1, 1, 1]:  # block 3 my r
        my_score[move(loc, 0)] += 20
        my_score[move(loc, 1)] += 25
        return
    if block(sec[1], 'op') and sec[2:] == [2, 2, 2, 0, 0]:  # block 3 op
        op_score[move(loc, 5)] += 15
        op_score[move(loc, 6)] += 25
        return
    if block(sec[-2], 'op') and sec[:-2] == [0, 0, 2, 2, 2]:  # block 3 op r
        op_score[move(loc, 0)] += 25
        op_score[move(loc, 1)] += 15
        return
    if sec[1:-1] == [1, 1, 1, 0, 1]:  # jump 3 my
        my_score[move(loc, 4)] += 5000
        return
    if sec[1:-1] == [1, 0, 1, 1, 1]:  # jump 3 my r
        my_score[move(loc, 2)] += 5000
        return
    if sec[1:-1] == [2, 2, 2, 0, 2]:  # jump 3 op
        op_score[move(loc, 4)] += 1000
        return
    if sec[1:-1] == [2, 0, 2, 2, 2]:  # jump 3 op r
        op_score[move(loc, 2)] += 1000
        return

    # link 4 group
    if sec[1:-1] == [1, 1, 1, 1, 0]:
        my_score[move(loc, 5)] += 5000
        return
    if sec[1:-1] == [0, 1, 1, 1, 1]:
        my_score[move(loc, 1)] += 5000
        return
    if sec[1:-1] == [2, 2, 2, 2, 0]:
        op_score[move(loc, 5)] += 1000
        return
    if sec[1:-1] == [0, 2, 2, 2, 2]:
        op_score[move(loc, 1)] += 1000
        return
    if sec[1:-1] == [1, 1, 0, 1, 1]:
        my_score[move(loc, 3)] += 5000
        return
    if sec[1:-1] == [2, 2, 0, 2, 2]:
        op_score[move(loc, 3)] += 1000
        return

    # dis 3 group
    if sec[:-1] == [0, 1, 1, 0, 1, 0]:  # all live d my
        my_score[move(loc, 3)] += 100
        return
    if sec[:-1] == [0, 1, 0, 1, 1, 0]:  # all live d my r
        my_score[move(loc, 2)] += 100
        return
    if sec[:-1] == [0, 2, 2, 0, 2, 0]:  # all live d op
        op_score[move(loc, 3)] += 100
        op_score[move(loc, 0)] += 20
        op_score[move(loc, 5)] += 20
        return
    if sec[:-1] == [0, 2, 0, 2, 2, 0]:  # all live d op r
        op_score[move(loc, 2)] += 100
        op_score[move(loc, 0)] += 20
        op_score[move(loc, 5)] += 20
        return
    if block(sec[0], 'my') and sec[1:-1] == [1, 1, 0, 1, 0]:  # block d my
        my_score[move(loc, 3)] += 10
        my_score[move(loc, 5)] += 10
        return
    if block(sec[6], 'my') and sec[1:-1] == [0, 1, 0, 1, 1]:  # block d my r
        my_score[move(loc, 1)] += 10
        my_score[move(loc, 3)] += 10
        return
    if block(sec[0], 'op') and sec[1:-1] == [2, 2, 0, 2, 0]:  # block d op
        op_score[move(loc, 3)] += 15
        op_score[move(loc, 5)] += 20
        return
    if block(sec[6], 'op') and sec[1:-1] == [0, 2, 0, 2, 2]:  # block d op r
        op_score[move(loc, 1)] += 20
        op_score[move(loc, 3)] += 15
        return
    if block(sec[6], 'my') and sec[1:-1] == [0, 1, 1, 0, 1]:  # block d2 my
        my_score[move(loc, 1)] += 10
        my_score[move(loc, 4)] += 10
        return
    if block(sec[0], 'my') and sec[1:-1] == [1, 0, 1, 1, 0]:  # block d2 my r
        my_score[move(loc, 2)] += 10
        my_score[move(loc, 5)] += 10
        return
    if block(sec[6], 'op') and sec[1:-1] == [0, 2, 2, 0, 2]:  # block d2 op
        op_score[move(loc, 1)] += 20
        op_score[move(loc, 4)] += 15
        return
    if block(sec[0], 'op') and sec[1:-1] == [2, 0, 2, 2, 0]:  # block d2 op r
        op_score[move(loc, 2)] += 15
        op_score[move(loc, 5)] += 20
        return

    # addition
    if sec[1:-1] == [2, 2, 0, 0, 0] and block(sec[0], 'op'):
        op_score[move(loc, 3)] += 10
        op_score[move(loc, 4)] += 10
        return
    if sec[1:-1] == [0, 0, 0, 2, 2] and block(sec[5], 'op'):
        op_score[move(loc, 3)] += 10
        op_score[move(loc, 2)] += 10
        return
