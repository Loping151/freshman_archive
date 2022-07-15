# -*- coding:utf-8  -*-
# Time  : 2022/7
# Author: Loping151

from gym.spaces import Discrete
import Lo_Gobang_151.submission as p151
import Lo_Gobang_151_v2.submission as p151v2
import Lo_Gobang_151_v3.submission as p151v3
import Lo_Gobang_151_v4.submission as p151v4
import Lo_alpha.submission as p151va
import Lo_another.submission as p151another
import numpy as np

from judgement import *
from time import sleep
import os

np.set_printoptions(linewidth=400)
firsthand = 1


# show mode will show board and pause mode will pause 0.5 second each step
def game(player1, player2, show=True, pause=False):
    # to test 残局 here
    _x = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    _x = np.array(_x)
    if firsthand == 1:
        if show:
            print('p1:1, p2:2')
        players = [player1, player2]
    else:
        if show:
            print('p1:2, p2:1')
        players = [player2, player1]
    turn = 0
    while True:
        act1 = players[turn].my_controller({'state_map': _x, 'chess_player_idx': 1}, [Discrete(15), Discrete(15)])
        x = act1[0].index(1)
        y = act1[1].index(1)
        _x[x, y] = 1
        if show:
            print(x, y)
            print(_x)
        if pause:
            sleep(0.5)
        s = check(_x)
        if s == firsthand or s + firsthand == 3:
            return firsthand
        if s == 3:
            return s
        turn = 1 - turn
        act2 = players[turn].my_controller({'state_map': _x, 'chess_player_idx': 2}, [Discrete(15), Discrete(15)])
        x = act2[0].index(1)
        y = act2[1].index(1)
        _x[x, y] = 2
        if show:
            print(x, y)
            print(_x)
        if pause:
            sleep(0.5)
        s = check(_x)
        if s == firsthand or s + firsthand == 3:
            return firsthand
        if s == 3:
            return s
        turn = 1 - turn


def compare(player1, player2, Round=100):
    global firsthand
    score = [0, 0, 0, 0]
    for _ in range(Round):
        print('\rRound', _, end=' ')
        score[game(player1, player2, show=False)] += 1
        firsthand = 3 - firsthand  # switch firsthand
    print('\rp1 wins {}%, loses {}%, draws {}%'.format(score[1] * 100 / Round, score[2] * 100 / Round,
                                                       score[3] * 100 / Round))
    # output p1 is the former and p2 is the latter player you give
    print('')


compare(p151, p151v4)
compare(p151, p151va)
compare(p151, p151)
compare(p151v4, p151v4)
compare(p151v4, p151va)
compare(p151va, p151va)
