# -*- coding:utf-8  -*-
# Time  : 2022/7/6
# Author: Loping151
# This is the first Gobang(Gomoku) DQN realized with tensorflow2.x. The author is a freshman, so bugs will exist.
# Email: wangkailing151@gmail.com

import tensorflow as tf
from tensorflow.python.keras import Sequential, layers
from keras import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from collections import deque
from coach import Coach
import numpy as np
import random
import time


class Lo_DQN(Model):
    def call(self, inputs, training=None, mask=None):
        pass

    def __init__(self):
        super(Lo_DQN, self).__init__()
        self.step = 0  # record current step(total step, beyond games)
        self.size = 15  # chessboard size, but hard to change because 151s are all 15-based, too much to change
        self.update_freq = 300  # after how many steps you update target_model
        self.replay_size = 50  # experience replay size
        self.learning_rate = 0.001
        self.epsilon_explore = 0.1 / 5000  # init explore rate/end explore step
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.loss = []

    def create_model(self):  # model architecture here, used relu
        model = Sequential([
            layers.Conv2D(5, kernel_size=2),
            layers.ReLU(),
            layers.Conv2D(10, kernel_size=2),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(300, activation='relu'),
            layers.Dense(225, activation='relu'),
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(self.learning_rate))
        return model

    # When applying this model, you should turn off explore
    def act(self, board, epsilon=0.1, explore=True):
        # explore happens when 1)you allow exploring 2)step lower than 5000 3) less than 50 chess pieces on board
        if explore and np.random.uniform() < epsilon - self.step * self.epsilon_explore and np.sum(board) < 150:
            # x = np.random.randint(0, 225)
            # while board[x // 15, x % 15] != 0:
            #     x = np.random.randint(0, 225)
            x = Coach().act(board)
            return x
        rank = np.array(self.model.predict(board.reshape((1, self.size, self.size, 1)), verbose=0))[0].argsort()
        index = -1
        while board[rank[index] // 15, rank[index] % 15] != 0:
            index -= 1
        return rank[index]

    # You should read the code if you really want to save and load
    def save_model(self, save_name, save_dir='', name_time=True, step=0):
        print('model saved')
        if name_time and step:
            self.model.save(save_dir + time.strftime('%m.%d.%H.%M_') + str(step) + '_' + save_name)
        elif name_time:
            self.model.save(save_dir + time.strftime('%m.%d.%H.%M_') + save_name)
        elif step:
            self.model.save(save_dir + str(step) + '_' + save_name)
        else:
            self.model.save(save_dir + save_name)

    def load_model(self, load_name, load_dir=''):
        model1 = tf.keras.models.load_model(load_dir + load_name)
        self.model = model1
        model2 = tf.keras.models.load_model(load_dir + load_name)
        self.target_model = model2

    def remember(self, board, action, next_board):
        reward = get_reward(next_board, rule='default') - get_reward(board, rule='default')
        _board = board.reshape((self.size, self.size, 1))
        _next_board = next_board.reshape((self.size, self.size, 1))
        self.replay_queue.append((_board, action, _next_board, reward))
        return reward

    def train(self, batch_size=50, lr=0, factor=0.5):
        if len(self.replay_queue) < self.replay_size:
            return
        if not lr:
            lr = self.learning_rate
        print('Training...')
        self.step += 1
        if not self.step % self.update_freq:
            self.target_model.set_weights(self.model.get_weights())
        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])
        Q = self.model.predict(s_batch, verbose=0)
        Q_next = self.target_model.predict(next_s_batch)
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward / 1e3 + factor * np.amax(Q_next[i]))
        history = self.model.fit(s_batch, Q, verbose=0)
        self.loss.append(history.history['loss'])
        self.replay_queue.clear()

    def get_loss(self):
        return self.loss


def get_reward(board, rule='default'):
    score = scan(board, rule)
    return np.sum(score)


def down(p, m):
    return p[0] + m, p[1]


def right(p, m):
    return p[0], p[1] + m


def dr(p, m):
    return p[0] + m, p[1] + m


def dl(p, m):
    return p[0] + m, p[1] - m


# mode 1 for beginning, and 2 for latter
def scan(c_map, rule='default'):
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
            judge(sec=index, score=score, rule=rule)

    # right
    for i in range(2, 17):
        for j in range(13):
            index = f_map[i, j:j + 7]
            judge(sec=index, score=score, rule=rule)

    # down right
    for i in range(-10, 11):
        line = np.diagonal(f_map, offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            judge(sec=index, score=score, rule=rule)

    # up right
    for i in range(-10, 11):
        line = np.diagonal(np.fliplr(f_map), offset=i)
        for j in range(len(line) - 6):
            index = line[j:j + 7]
            judge(sec=index, score=score, rule=rule)

    return score


# judge block for each player
def block(pos, player):
    if player == 'my':
        return pos != 0 and pos != 1
    if player == 'op':
        return pos != 0 and pos != 2


# current location, input sequence, score matrix, move type
def judge(sec, score, rule):
    sec = list(sec)

    if rule == 'simple':
        if sec[:2] == [1, 1]:
            score[0, 0] += 10
            return
        if sec[:2] == [1, 2]:
            score[0, 0] += 10
            return
        if sec[:2] == [2, 1]:
            score[0, 0] += 10
            return
        if sec[:5] == [1, 1, 1, 1, 1]:
            score[0, 0] += 100
            return
        return

    if rule == 'default':
        # simple cases
        if sec[1:-1] == [0, 0, 0, 0, 0]:
            return
        if sec[1:-1] == [1, 1, 1, 1, 1]:
            score[0, 0] += 200
            return
        if sec[:-1] == [0, 0, 1, 1, 0, 0]:
            score[0, 0] += 10
            return
        if sec[:-1] == [0, 0, 1, 2, 1, 0]:
            score[0, 0] += 20
            return
        if sec[:-1] == [0, 0, 1, 2, 0, 0]:
            score[0, 0] += 20
            return
        if sec[:-1] == [0, 0, 0, 2, 1, 0]:
            score[0, 0] += 20
            return
        if sec[:-1] == [0, 1, 2, 2, 1, 0]:
            score[0, 0] += 30
            return
        if sec[:-1] == [0, 0, 2, 2, 1, 0]:
            score[0, 0] += 20
            return
        if sec[:-1] == [0, 1, 2, 2, 0, 0]:
            score[0, 0] += 20
            return
        if sec[1:-1] == [0, 1, 0, 1, 0]:
            score[0, 0] += 20
            return
        if sec[1:-1] == [0, 2, 1, 2, 0]:
            score[0, 0] += 20
            return

        # link 3 group
        if sec == [0, 0, 1, 1, 1, 0, 0]:  # all live 3 my
            score[0, 0] += 50
            return
        if sec == [0, 1, 2, 2, 2, 1, 0]:  # all live 3 op
            score[0, 0] += 70
            return
        if sec == [0, 0, 2, 2, 2, 1, 0]:  # all live 3 op
            score[0, 0] += 50
            return
        if sec == [0, 1, 2, 2, 2, 0, 0]:  # all live 3 op
            score[0, 0] += 50
            return
        if block(sec[0], 'my') and sec[1:] == [0, 1, 1, 1, 0, 0]:  # single live 3 my
            score[0, 0] += 30
            return
        if block(sec[-1], 'my') and sec[:-1] == [0, 0, 1, 1, 1, 0]:  # single live 3 my r
            score[0, 0] += 30
            return
        if block(sec[0], 'op') and sec[1:] == [1, 2, 2, 2, 1, 0]:  # single live 3 op
            score[0, 0] += 30
            return
        if block(sec[0], 'op') and sec[1:] == [0, 2, 2, 2, 1, 0]:  # single live 3 op
            score[0, 0] += 50
            return
        if block(sec[-1], 'op') and sec[:-1] == [0, 1, 2, 2, 2, 1]:  # single live 3 op r
            score[0, 0] += 30
            return
        if block(sec[-1], 'op') and sec[:-1] == [0, 1, 2, 2, 2, 0]:  # single live 3 op r
            score[0, 0] += 50
            return
        if block(sec[0], 'op') and (sec[1:] == [1, 2, 2, 2, 0, 1] or sec[1:] == [1, 2, 2, 2, 0, 1]):  # single live 3 op
            score[0, 0] += 30
            return
        if block(sec[0], 'op') and (sec[1:] == [0, 2, 2, 2, 0, 1] or sec[1:] == [1, 2, 2, 2, 0, 0]):  # single live 3 op
            score[0, 0] += 20
            return
        if block(sec[-1], 'op') and (
                sec[:-1] == [1, 0, 2, 2, 2, 1] or sec[:-1] == [1, 0, 2, 2, 2, 1]):  # single live 3 op
            score[0, 0] += 30
            return
        if block(sec[-1], 'op') and (
                sec[:-1] == [1, 0, 2, 2, 2, 0] or sec[:-1] == [0, 0, 2, 2, 2, 1]):  # single live 3 op
            score[0, 0] += 20
            return
        if block(sec[0], 'my') and block(sec[-1], 'my') and sec[1: -1] == [0, 1, 1, 1, 0]:  # double live 3 my
            score[0, 0] += 10
            return
        if block(sec[0], 'op') and block(sec[0], 'op') and sec[1: -1] == [0, 2, 2, 2, 1]:  # double live 3 op
            score[0, 0] += 20
            return
        if block(sec[0], 'op') and block(sec[-1], 'op') and sec[1: -1] == [1, 2, 2, 2, 0]:  # double live 3 op
            score[0, 0] += 20
            return
        if block(sec[1], 'my') and sec[2:] == [1, 1, 1, 0, 0]:  # block 3 my
            score[0, 0] += 20
            return
        if block(sec[-2], 'my') and sec[:-2] == [0, 0, 1, 1, 1]:  # block 3 my r
            score[0, 0] += 20
            return
        if block(sec[1], 'op') and sec[2:] == [2, 2, 2, 0, 1]:  # block 3 op
            score[0, 0] += 25
        if block(sec[1], 'op') and sec[2:] == [2, 2, 2, 1, 0]:  # block 3 op
            score[0, 0] += 15
            return
        if block(sec[-2], 'op') and sec[:-2] == [1, 0, 2, 2, 2]:  # block 3 op r
            score[0, 0] += 25
            return
        if block(sec[-2], 'op') and sec[:-2] == [0, 1, 2, 2, 2]:  # block 3 op r
            score[0, 0] += 15
            return
        if sec[1:-1] == [1, 1, 1, 0, 1]:  # jump 3 my
            score[0, 0] += 100
            return
        if sec[1:-1] == [1, 0, 1, 1, 1]:  # jump 3 my r
            score[0, 0] += 100
            return
        if sec[1:-1] == [2, 2, 2, 1, 2]:  # jump 3 op
            score[0, 0] += 75
            return
        if sec[1:-1] == [2, 1, 2, 2, 2]:  # jump 3 op r
            score[0, 0] += 75
            return

        # link 4 group
        if sec[1:-1] == [1, 1, 1, 1, 0] or sec[0:-2] == [0, 1, 1, 1, 1]:
            score[0, 0] += 100
            return
        if sec[:-1] == [1, 2, 2, 2, 2, 1]:
            score[0, 0] += 100
            return
        if sec[1:-1] == [2, 2, 2, 2, 1]:
            score[0, 0] += 75
            return
        if sec[:-2] == [1, 2, 2, 2, 2]:
            score[0, 0] += 75
            return
        if sec[1:-1] == [1, 1, 0, 1, 1]:
            score[0, 0] += 100
            return
        if sec[1:-1] == [2, 2, 1, 2, 2]:
            score[0, 0] += 75
            return

        # dis 3 group
        if sec[:-1] == [0, 1, 1, 0, 1, 0]:  # all live d my
            score[0, 0] += 50
            return
        if sec[:-1] == [0, 1, 0, 1, 1, 0]:  # all live d my r
            score[0, 0] += 50
            return
        if sec[:-1] == [1, 2, 2, 1, 2, 0]:  # all live d op
            score[0, 0] += 50
            return
        if sec[:-1] == [0, 2, 2, 1, 2, 1]:  # all live d op
            score[0, 0] += 10
            return
        if sec[:-1] == [0, 2, 2, 1, 2, 0]:  # all live d op
            score[0, 0] += 50
            return
        if sec[:-1] == [0, 2, 1, 2, 2, 1]:  # all live d op r
            score[0, 0] += 50
            return
        if sec[:-1] == [1, 2, 1, 2, 2, 0]:  # all live d op r
            score[0, 0] += 10
            return
        if sec[:-1] == [0, 2, 1, 2, 2, 0]:  # all live d op r
            score[0, 0] += 50
            return
        if block(sec[0], 'my') and sec[1:-1] == [1, 1, 0, 1, 0]:  # block d my
            score[0, 0] += 10
            return
        if block(sec[6], 'my') and sec[1:-1] == [0, 1, 0, 1, 1]:  # block d my r
            score[0, 0] += 10
            return
        if block(sec[0], 'op') and sec[1:-1] == [2, 2, 1, 2, 1]:  # block d op
            score[0, 0] += 10
            return
        if block(sec[0], 'op') and sec[1:-1] == [2, 2, 1, 2, 0]:  # block d op
            score[0, 0] += 10
            return
        if block(sec[0], 'op') and sec[1:-1] == [2, 2, 0, 2, 1]:  # block d op
            score[0, 0] += 15
            return
        if block(sec[6], 'op') and sec[1:-1] == [1, 2, 1, 2, 2]:  # block d op r
            score[0, 0] += 10
            return
        if block(sec[6], 'op') and sec[1:-1] == [0, 2, 1, 2, 2]:  # block d op r
            score[0, 0] += 10
            return
        if block(sec[6], 'op') and sec[1:-1] == [1, 2, 0, 2, 2]:  # block d op r
            score[0, 0] += 15
            return
        if block(sec[6], 'my') and sec[1:-1] == [0, 1, 1, 0, 1]:  # block d2 my
            score[0, 0] += 10
            return
        if block(sec[0], 'my') and sec[1:-1] == [1, 0, 1, 1, 0]:  # block d2 my r
            score[0, 0] += 10
            return
        if block(sec[6], 'op') and sec[1:-1] == [1, 2, 2, 1, 2]:  # block d2 op
            score[0, 0] += 15
            return
        if block(sec[6], 'op') and sec[1:-1] == [1, 2, 2, 0, 2]:  # block d2 op
            score[0, 0] += 15
            return
        if block(sec[6], 'op') and sec[1:-1] == [0, 2, 2, 1, 2]:  # block d2 op
            score[0, 0] += 15
            return
        if block(sec[0], 'op') and sec[1:-1] == [2, 1, 2, 2, 1]:  # block d2 op r
            score[0, 0] += 10
            return
        if block(sec[0], 'op') and sec[1:-1] == [2, 0, 2, 2, 1]:  # block d2 op r
            score[0, 0] += 10
            return
        if block(sec[0], 'op') and sec[1:-1] == [2, 1, 2, 2, 0]:  # block d2 op r
            score[0, 0] += 10
            return

        # additional
        if sec[1:-2] == [1, 1, 0, 0] and block(sec[0], 'my'):
            score[0, 0] += 5
            return
        if sec[1:-2] == [0, 0, 1, 1] and block(sec[5], 'my'):
            score[0, 0] += 5
            return
        if (sec[1:-2] == [2, 2, 1, 0] or sec[1:-2] == [2, 2, 0, 1]) and block(sec[0], 'op'):
            score[0, 0] += 7
            return
        if (sec[1:-2] == [1, 0, 2, 2] or sec[1:-2] == [0, 1, 2, 2]) and block(sec[5], 'op'):
            score[0, 0] += 7
            return
        if sec[1:-1] == [2, 1, 0, 0, 0] and block(sec[0], 'op'):
            score[0, 0] += 5
            return
        if sec[1:-1] == [0, 0, 0, 1, 2] and block(sec[6], 'op'):
            score[0, 0] += 5
            return
        if sec[2:-2] == [1, 2, 0] or sec[2:-2] == [0, 2, 1]:
            score[0, 0] += 5
            return
        if (sec[-4:-1] == [0, 1, 2] or sec[-4:-1] == [1, 0, 2]) and block(sec[4], 'op'):
            score[0, 0] += 5
            return
        if (sec[1:4] == [2, 1, 0] or sec[1:4] == [2, 0, 1]) and block(sec[0], 'op'):
            score[0, 0] += 5
            return
