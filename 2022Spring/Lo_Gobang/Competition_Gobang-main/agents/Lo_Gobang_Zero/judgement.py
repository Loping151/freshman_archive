# -*- coding:utf-8  -*-
# Time  : 2022/7/6
# Author: Loping151
# in this file there is a Judge class used in training


import numpy as np


def board_switch(b):
    for i in range(15):
        for j in range(15):
            if b[i, j]:  # not 0
                b[i, j] = 3 - b[i, j]  # switch 1, 2
    return b


class Judge:
    def __init__(self, p1, p2, firsthand=1, board=np.zeros((15, 15))):
        self.board = np.array(board)
        self.next_board = self.board.copy()
        if firsthand == 1:
            self.agents = [p1, p2]
        else:
            self.agents = [p2, p1]
        self.winner = -1

    def game_finish(self, max_step=1e5):
        winner = check(self.next_board)
        if winner == 1 or winner == 2:
            self.winner = winner
            print('Winner:', type(self.agents[winner - 1]).__name__)
            return True
        if winner == 3:
            self.winner = winner
            print('Draw!')
            return True
        if np.sum(self.next_board) > max_step * 3:
            return True
        return False

    def first(self):
        act1 = self.agents[0].act(self.next_board)
        self.board = self.next_board.copy()
        self.next_board[act1 // 15, act1 % 15] = 1
        return self.board.copy(), act1, self.next_board.copy()

    def second(self):
        reverse_board = board_switch(self.next_board.copy())
        act2 = self.agents[1].act(reverse_board)
        self.board = self.next_board.copy()
        self.next_board[act2 // 15, act2 % 15] = 2
        next_reverse_board = reverse_board.copy()
        next_reverse_board[act2 // 15, act2 % 15] = 1
        return reverse_board, act2, next_reverse_board

    def print_board(self):
        print(self.next_board)


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
