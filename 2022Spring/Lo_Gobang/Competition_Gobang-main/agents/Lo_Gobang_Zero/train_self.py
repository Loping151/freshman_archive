# -*- coding:utf-8  -*-
# Time  : 2022/7
# Author: Loping151
# you can use this file to let two model play with each other and train
# one is coach and one learns. The role will switch after a few games

from Logo_net import Lo_DQN
from judgement import Judge
import os
import matplotlib.pyplot as plt

# initialize two agents
agent1 = Lo_DQN()
agent2 = Lo_DQN()

# make dictionary
if not os.path.exists('./train_self'):
    os.makedirs('./train_self')

# TODO basic training options
Round = 30000

# if to continue train on a given model, you should load them here. mind the dictionary(./train_self)
# TODO model name here
name1 = None  # 'Lo_zero_base.h5'
name2 = None  # 'Lo_zero_base.h5'
if name1:
    agent1.load_model(load_dir='./train_self/', load_name=name1)
if name2:
    agent2.load_model(load_dir='./train_self/', load_name=name2)

# train, you can simply stop by KeyboardInterrupt
try:
    trainee = 0  # who to be trained.
    fh = 1  # to be switched
    agents = [agent1, agent2]
    for _ in range(Round):
        print('Round', _, end=' ')
        game = None
        print('firsthand:', end=' ')
        if fh == 1:
            print('agent')
            game = Judge(agent1, agent2, firsthand=1)
        if fh == 2:
            print('coach')
            game = Judge(agent1, agent2, firsthand=2)
        for counter in range(255):
            b, a, n_b = game.first()
            if fh != trainee:
                agents[trainee].remember(b, a, n_b)
                agents[trainee].train()
            if game.game_finish():
                game.print_board()
                break
            b, a, n_b = game.second()
            if fh == trainee:
                agents[trainee].remember(b, a, n_b)
                agents[trainee].train()
            if game.game_finish():
                game.print_board()
                break
        if _ > 0 and _ % 50 == 0:  # switch trainee
            fh = 3 - fh
        if _ > 0 and _ % 50 == 0:  # save recent 10 tmp model
            trainee = 1 - trainee
            agent1.save_model(save_dir='./train_self/', save_name='tmp1.h5', step=_, name_time=False)
            if os.path.isfile('./train_self/' + str(_ - 300) + '_tmp1.h5'):
                os.remove('./train_self/' + str(_ - 300) + '_tmp1.h5')
            agent2.save_model(save_dir='./train_self/', save_name='tmp2.h5', step=_, name_time=False)
            if os.path.isfile('./train_self/' + str(_ - 300) + '_tmp2.h5'):
                os.remove('./train_self/' + str(_ - 300) + '_tmp2.h5')
    agent1.save_model(save_dir='./train_self/', save_name='model1.h5')
    agent2.save_model(save_dir='./train_self/', save_name='model2.h5')
except KeyboardInterrupt:
    print('You stopped, emergency save')
    # the code below is to plot loss
    # loss = agent1.get_loss()
    # plt.plot(range(len(loss)), loss)
    # plt.xlabel('replay')
    # plt.ylabel('loss')
    # plt.show()
    agent1.save_model(save_dir='./train_self/', save_name='agent1_interrupt.h5')
    agent2.save_model(save_dir='./train_self/', save_name='agent2_interrupt.h5')
