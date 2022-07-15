# -*- coding:utf-8  -*-
# Time  : 2022/7
# Author: Loping151

import numpy as np


def check(chess_board):
    # construct full map with boundary walls
    f_map = np.zeros((19, 19))
    f_map[[0, 1, 17, 18]] = np.ones((4, 19)) * -1
    f_map[:, [0, 1, 17, 18]] = np.ones((19, 4)) * -1
    f_map[2:17, 2:17] = chess_board
    score = [0, 0, 0]

    for i in range(13):
        for j in range(2, 17):
            index = f_map[i:i + 7, j]
            score[_judge(sec=index)] = 1

    # right
    for i in range(2, 17):
        for j in range(13):
            index = f_map[i, j:j + 7]
            score[_judge(sec=index)] = 1

    # down right
    for i in range(-10, 11):
        line = np.diagonal(f_map, offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            score[_judge(sec=index)] = 1

    # up right
    for i in range(-10, 11):
        line = np.diagonal(np.fliplr(f_map), offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            score[_judge(sec=index)] = 1

    if score[1] and score[2]:
        print('Invalid chess_board.')
        return -1
    if score[2]:
        return 2
    if score[1]:
        return 1
    if np.min(chess_board):  # Draw
        return 3
    return 0


def _judge(sec):
    sec = list(sec)  # should be list to compare
    if sec[1:-1] == [1, 1, 1, 1, 1]:
        return 1
    if sec[1:-1] == [2, 2, 2, 2, 2]:
        return 2
    return 0
