# -*- coding:utf-8  -*-
# Time  : 2022/7
# Author: Loping151
# you can use this file to train
# either agent train with coach or watch coaches play

from coach import Coach
from Logo_net import Lo_DQN
from judgement import Judge
import os

agent = Lo_DQN()
coach = Coach()

# make dictionary
if not os.path.exists('./train_coach'):
    os.makedirs('./train_coach')

# TODO basic training options
Round = 30000

# if to continue train on a given model
# TODO model name here
name = ''
if name:
    agent.load_model(load_dir='./train_coach/', load_name=name)

# train
try:
    fh = 1
    for _ in range(Round):
        print('Round', _, end=' ')
        agent.prepare()
        # mind this is an option
        # TODO to train with coach(agent vs coach)
        # game = None
        # print('firsthand:', end=' ')
        #
        # if fh == 1:
        #     print('agent')
        #     game = Judge(agent, coach, firsthand=1)
        # if fh == 2:
        #     print('coach')
        #     game = Judge(agent, coach, firsthand=2)

        # TODO to watch coaches play
        game = Judge(coach, coach)

        for counter in range(255):
            b, a, n_b = game.first()
            if fh == 2:  # learn only from coach
                agent.remember(b, a, n_b)
                agent.train()
            if game.game_finish():
                game.print_board()
                break
            b, a, n_b = game.second()
            if fh == 1:
                agent.remember(b, a, n_b)
                agent.train()
            if game.game_finish():
                game.print_board()
                break
        if _ > 0 and _ % 100 == 0:
            fh = 3 - fh
            agent.save_model(save_dir='./train_coach/', save_name='tmp.h5', step=_, name_time=False)
            if os.path.isfile('./train_coach/' + str(_ - 500) + '_tmp.h5'):
                os.remove('./train_coach/' + str(_ - 500) + '_tmp.h5')
    agent.save_model(save_dir='./train_coach/', save_name='model.h5')
except KeyboardInterrupt:
    print('You stopped, emergency save')
    agent.save_model(save_dir='./train_coach/', save_name='agent_interrupt.h5')
