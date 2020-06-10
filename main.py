# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload


from keras.utils import plot_model

from game import Game, GameState
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

# If loading an existing neural network, copy the config file to root
if initialise.INITIAL_RUN_NUMBER != None:
    copyfile(run_archive_folder  + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py', './config.py')

import config

# create an untrained neural network objects from the config file
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) +  env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

best_player_version = 48
m_tmp = best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
best_NN.model.set_weights(m_tmp.get_weights())

current_player_version = 1
m_tmp2 = current_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, current_player_version)
current_NN.model.set_weights(m_tmp2.get_weights())


#copy the config file to the run folder
copyfile('./config.py', run_folder + 'config.py')
plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes = True)

print('\n')

######## CREATE THE PLAYERS ########
current_player = Agent('version_1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('version_48', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

iteration = 0
while 1:
    iteration += 1
    lg.logger_main.info('ITERATION NUMBER ' + str(iteration))
    
    lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
    lg.logger_main.info('CURRENT PLAYER VERSION: %d', current_player_version)
    print('BEST PLAYER VERSION ' + str(best_player_version))
    print('CURRENT PLAYER VERSION ' + str(current_player_version))

    ######## TOURNAMENT ########
    lg.logger_main.info('TOURNAMENT...')
    scores, _, points, sp_scores = playMatches(best_player, current_player, config.EVAL_EPISODES, lg.logger_tourney, turns_until_tau0 = 0, memory = None)
    lg.logger_main.info('\nSCORES')
    lg.logger_main.info(scores)
    lg.logger_main.info('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
    lg.logger_main.info(sp_scores)

    lg.logger_main.info('\n\n')