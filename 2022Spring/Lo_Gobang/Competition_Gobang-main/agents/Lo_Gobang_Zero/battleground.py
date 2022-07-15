from gym.spaces import Discrete
import coach as p1
# import submission as p2
import random_agent as p3
import numpy as np
from judgement import *
from time import sleep
import os
from coach import scan_v2
np.set_printoptions(linewidth=400)
def game(player1, player2, show=True, pause=False):
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

    while True:
        act1 = player1.my_controller({'state_map': _x, 'chess_player_idx': 1}, [Discrete(15), Discrete(15)])
        x = act1[0].index(1)
        y = act1[1].index(1)
        _x[x, y] = 1
        if show:
            print(x, y)
            print(_x)
        if pause:
            sleep(0.5)
        s = check(_x)
        if s:
            return s
        act2 = player2.my_controller({'state_map': _x, 'chess_player_idx': 2}, [Discrete(15), Discrete(15)])
        x = act2[0].index(1)
        y = act2[1].index(1)
        _x[x, y] = 2
        if show:
            print(x, y)
            print(_x)
        if pause:
            sleep(0.5)
        s = check(_x)
        if s:
            return s


def compare(player1, player2, Round=1000):
    score = [0, 0, 0, 0]
    for _ in range(Round):
        print('\rRound', _, end=' ')
        score[game(player1, player2, show=False)] += 1
    print('\rp1 wins {}%, loses {}%, draws {}%'.format(score[1] * 100 / Round, score[2] * 100 / Round,
                                                       score[3] * 100 / Round))

