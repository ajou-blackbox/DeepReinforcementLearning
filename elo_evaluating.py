# -*- coding: utf-8 -*-
# %matplotlib inline
#%load_ext autoreload
#%autoreload 2

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
import config

import time
import os

import random
import operator

env = Game()


EVAL_COUNT = 1  # 몇 번 대전 수행할지 설정
EVAL_COUNT_ONETIME = 1  # 한 번 만나면 몇회전 할 지 설정
MCTS_SIMS = 50

MODEL_START = 1 # 몇 번 모델 버전부터 평가에 사용할지 결정
MODEL_SPACE = 10 # 평가할 모델 버전 간격
HIGHEST_VERSION = 500 # 가장 높은 버전

INIT_RATING = 500 # 초기 ELO Rating
ELO_CONST = 20  # ELO 계산에 사용하는 상수 K (프로 : 16, 일반 : 32)
ELO_CONST_NEW = 40  # 배치고사 시 사용하는 K
PLACEMENT_COUNT = 10 # 배치고사 판수

# elo[model][rating, 평가 경기 수]
# record[record_num][model1, model2, result, 사용여부]

def check_model():  # 사용할 모델 존재 체크
    using_model = []
    for version in range(MODEL_START, HIGHEST_VERSION+1, MODEL_SPACE):
        if os.path.exists('./run/models/version' + "{0:0>4}".format(version) + '.h5'):
            using_model.append(version)
    return using_model

def pick_model_random(using_model):    # 사용할 모델 중 랜덤하게 2개 고름
    model1 = random.choice(using_model)
    model2 = random.choice(using_model)
    while model2 == model1:
        model2 = random.choice(using_model)
    return model1, model2

def pick_model(using_model, eval_num):  # 각 모델간 플레이 횟수 동일해지도록 모델 2개 고름
    min_pair_key = min(eval_num.items(), key=operator.itemgetter(1))[0]
    border = min_pair_key.find('-')
    model1 = int(min_pair_key[0:border])
    model2 = int(min_pair_key[border+1:])
    return model1, model2
    

def load_record():  # 게임 기록 불러옴
    if not os.path.exists('./ratings/eval_record.pickle'):
        record = []
        with open('./ratings/eval_record.pickle', 'wb') as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return record
    else: 
        with open('./ratings/eval_record.pickle', 'rb') as handle:
            record = pickle.load(handle)
        if record == '':
            return []
        else:
            return record

def save_record(record, model1, model2, result):    # 게임 기록 저장
    record.append([model1, model2, result, 0]) # 출력값은 Nonetype임에 주의
    with open('./ratings/eval_record.pickle', 'wb') as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return record

def save_record_direct(record):
    with open('./ratings/eval_record.pickle', 'wb') as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_eval_number(record, using_model):   # 각 매칭 몇 번 있었는지 dict로 정리
    eval_num = {}
    for record_num in range(len(record)):
        if record[record_num][0] > record[record_num][1]:
            player1 = record[record_num][1]
            player2 = record[record_num][0]
        else:
            player1 = record[record_num][0]
            player2 = record[record_num][1]
        if str(player1) + '-' + str(player2) in eval_num :
            eval_num[str(player1) + '-' + str(player2)] += 1
        else:
            eval_num[str(player1) + '-' + str(player2)] = 1

        for i in range(len(using_model)-1):
            for j in range(i+1, len(using_model)):
                if str(using_model[i]) + '-' + str(using_model[j]) not in eval_num:
                    eval_num[str(using_model[i]) + '-' + str(using_model[j])] = 0

    return eval_num



def load_elo(): # elo 기록 불러옴
    if os.path.exists('./ratings/elo.pickle') == False:
        elo = []
        with open('./ratings/elo.pickle', 'wb') as handle:
            pickle.dump(elo, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return elo
    else: 
        with open('./ratings/elo.pickle', 'rb') as handle:
            elo = pickle.load(handle)
        if elo == '':
            return []
        else:
            return elo

def save_elo(elo):
    with open('./ratings/elo.pickle', 'rb') as handle:
        elo_old = pickle.load(handle)
    with open('./ratings/elo_past.pickle', 'wb') as handle:
        pickle.dump(elo_old, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./ratings/elo.pickle', 'wb') as handle:
        pickle.dump(elo, handle, protocol=pickle.HIGHEST_PROTOCOL)  # 새 elo 저장

def calc_elo(elo, record): # 첫 판이면 기본점수 부여, 배치 감안 큰 폭 조정
    for record_num in range(len(record)):
        if record[record_num][3] == 0:
            model1 = record[record_num][0]
            model2 = record[record_num][1]

            # 기록 없는 모델 elo 전적 생성
            elo_len = len(elo)
            if elo_len < model1:
                for model_num in range(elo_len, model1+1):   
                    elo.append([INIT_RATING, 0])
            if elo_len < model2:
                for model_num in range(elo_len, model2+1):
                    elo.append([INIT_RATING, 0])

            model1_rating = elo[model1][0]
            model2_rating = elo[model2][0]

            # ELO rating 계산
            p1 = (1.0 / (1.0 + pow(10, ( (model2_rating - model1_rating) / 400.0 ))))   # model1이 이길 확률
            p2 = (1.0 / (1.0 + pow(10, ( (model1_rating - model2_rating) / 400.0 ))))   # model2가 이길 확률

            if record[record_num][2] == 1:
                elo_s1 = 1.0
                elo_s2 = 0.0
            elif record[record_num][2] == -1:
                elo_s1 = 0.0
                elo_s2 = 1.0
            else:
                elo_s1 = 0.5
                elo_s2 = 0.5
            
            # 배치고사인 경우 더 큰 K constant 적용
            if elo[model1][1] < PLACEMENT_COUNT:
                model1_elo_const = ELO_CONST_NEW
            else:
                model1_elo_const = ELO_CONST
            
            if elo[model2][1] < PLACEMENT_COUNT:
                model2_elo_const = ELO_CONST_NEW
            else:
                model2_elo_const = ELO_CONST

            model1_rating = model1_rating + model1_elo_const * (elo_s1 - p1)
            model2_rating = model2_rating + model2_elo_const * (elo_s2 - p2)

            if model1_rating < 0:
                model1_rating = 0
            if model2_rating < 0:
                model2_rating = 0

            # elo 갱신
            elo[model1][0] = round(model1_rating)
            elo[model2][0] = round(model2_rating)
            elo[model1][1] += 1
            elo[model2][1] += 1

            # record 사용했다고 표시
            record[record_num][3] = 1

            with open('./ratings/elo.pickle', 'wb') as handle:
                pickle.dump(elo, handle, protocol=pickle.HIGHEST_PROTOCOL)  # 새 elo 저장

            with open('./ratings/eval_record.pickle', 'wb') as handle:
                pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)   # record 사용표시한 것 저장

    return elo, record

'''
elo = [[0,0],[1200, 0], [3000, 0]]
record = [[1,2,1,1],[1,2,1,0],[1,2,1,0],[1,2,1,0],[1,2,1,0]]
calc_elo(elo, record)
'''

def elo_tournament(record, model1, model2):
    NN_1 = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
    NN_2 = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) +  env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

    m_tmp = NN_1.read(env.name, model1, model1)
    NN_1.model.set_weights(m_tmp.get_weights())
    m_tmp = NN_2.read(env.name, model2, model2)
    NN_2.model.set_weights(m_tmp.get_weights())

    player1 = Agent('player1', env.state_size, env.action_size, MCTS_SIMS, config.CPUCT, NN_1)
    player2 = Agent('player2', env.state_size, env.action_size, MCTS_SIMS, config.CPUCT, NN_2)

    scores, _, points, sp_scores = playMatches(player1, player2, EVAL_COUNT_ONETIME, lg.logger_tourney, turns_until_tau0 = 0, memory = None)

    while scores['player1'] > 0:
        save_record(record, model1, model2, 1)
        scores['player1'] -= 1
    
    while scores['player2'] > 0:
        save_record(record, model1, model2, -1)
        scores['player2'] -= 1 

    while scores['drawn'] > 0:
        save_record(record, model1, model2, 0)
        scores['drawn'] -= 1 


