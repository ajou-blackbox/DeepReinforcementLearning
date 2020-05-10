# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl

from datetime import datetime


class User():
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		action = int(input('Enter your chosen action: '))
		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = 0
		NN_value = 0
		return (action, pi, value, NN_value)



class Agent():
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name

		self.state_size = state_size
		self.action_size = action_size

		self.cpuct = cpuct

		self.MCTSsimulations = mcts_simulations
		self.model = model

		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []

	
	def simulate(self):

		lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
		self.mcts.root.state.render(lg.logger_mcts)
		lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

		##### MOVE THE LEAF NODE
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		leaf.state.render(lg.logger_mcts)

		##### EVALUATE THE LEAF NODE
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

		##### BACKFILL THE VALUE THROUGH THE TREE
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau):

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		#### run the simulation
		for sim in range(self.MCTSsimulations):
			lg.logger_mcts.info('***************************')
			lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
			lg.logger_mcts.info('***************************')
			self.simulate()

		#### get action values
		pi, values = self.getAV(1)

		####pick the action
		action, value = self.chooseAction(pi, values, tau)

		nextState, _, _ = state.takeAction(action)


		if nextState.playerTurn == state.playerTurn:
			NN_value = self.get_preds(nextState)[0]
		else:
			NN_value = -self.get_preds(nextState)[0]
		

		lg.logger_mcts.info('ACTION VALUES...%s', pi)
		lg.logger_mcts.info('CHOSEN ACTION...%d', action)
		lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
		lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

		return (action, pi, value, NN_value)


	def get_preds(self, state):

		#predict the leaf
		lg.logger_debug.info('START...')
		convertedModelInput, frag_allowed_count = self.model.convertToModelInput(state)
		lg.logger_debug.info('A...')
		inputToModel = np.expand_dims(np.array(convertedModelInput[0]), axis = 0)
		lg.logger_debug.info('B...')
		for i in range (1,64): 
			inputToModel = np.append(inputToModel, np.expand_dims(np.array(convertedModelInput[i]), axis = 0))
			lg.logger_debug.info('C %d...',i)

		for i in range(64):

			preds = self.model.predict(inputToModel[i])
			lg.logger_debug.info('D %d...',i)
			value_array = preds[0]
			lg.logger_debug.info('E %d...',i)
			logits_array = preds[1]
			lg.logger_debug.info('F %d...',i)
			
			if i==0:
				value = np.expand_dims(value_array, axis = 0)
				lg.logger_debug.info('G %d...',i)
				logits = np.expand_dims(logits_array, axis = 0)
				lg.logger_debug.info('H %d...',i)

			else:
				value = np.append(value, np.expand_dims(value_array, axis = 0))
				lg.logger_debug.info('I %d...',i)
				logits = np.append(logits, np.expand_dims(logits_array, axis = 0))
				lg.logger_debug.info('J %d...',i)

		allowedActions = state.allowedActions
		lg.logger_debug.info('K...')
		# 학습시키는 기능 비활성화 필요

		# Integrator

		# Value 합산
		total_value = 0.0 # value : float
		vaild_value_count = 0
		for i in range(64):
			if frag_allowed_count[i] != 0:	# 남은 수 없는 fragment 배제
				total_value += value[i]
				vaild_value_count += 1
		total_value /= vaild_value_count

		# Policy 가중치
		frag_policy_weight = []
		for i in range(64):
			frag_policy_weight[i] = abs(value[i]) + 1
			frag_policy_weight[i] *= frag_allowed_count[i]

		# 19*19에 합산
		number_count = np.zeros(len(self.state_size), dtype=np.int)
		total_logits = np.zeros(len(self.action_size), dtype=np.int)
		for n in range(64):
			start_num = int(n/8)*19 + n%8
			for i in range(12):
				for j in range(12):
					now_num = start_num + i*19 + j
					total_logits[now_num] += logits[n][i*12 + j] * frag_policy_weight[n]
					number_count[now_num] += 1
		for i in range(len(self.action_size)):
			total_logits[i] /= number_count[i]

		# 둘 수 없는 곳 policy 낮게 만듬
		mask = np.ones(total_logits.shape,dtype=bool)
		mask[allowedActions] = False
		total_logits[mask] = -100		# AllowedAction 아닌 곳은 e^(-100) : 작은 수

		#SOFTMAX
		odds = np.exp(total_logits)
		probs = odds / np.sum(odds) ###put this just before the for?

		return ((total_value, probs, allowedActions))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		lg.logger_mcts.info('------EVALUATING LEAF------')

		if done == 0:
	
			value, probs, allowedActions = self.get_preds(leaf.state)
			lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

			probs = probs[allowedActions]

			for idx, action in enumerate(allowedActions):
				newState, _, _ = leaf.state.takeAction(action)
				if newState.id not in self.mcts.tree:
					node = mc.Node(newState)
					self.mcts.addNode(node)
					lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
				else:
					node = self.mcts.tree[newState.id]
					lg.logger_mcts.info('existing node...%s...', node.id)

				newEdge = mc.Edge(leaf, node, probs[idx], action)
				leaf.edges.append((action, newEdge))
				
		else:
			lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

		return ((value, breadcrumbs))


		
	def getAV(self, tau):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=np.integer)
		values = np.zeros(self.action_size, dtype=np.float32)
		
		for action, edge in edges:
			pi[action] = pow(edge.stats['N'], 1/tau)
			values[action] = edge.stats['Q']

		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def chooseAction(self, pi, values, tau):
		if tau == 0:
			actions = np.argwhere(pi == max(pi))
			action = random.choice(actions)[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]

		value = values[action]

		return action, value

	def replay(self, ltmemory):
		lg.logger_mcts.info('******RETRAINING MODEL******')


		for i in range(config.TRAINING_LOOPS):
			minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

			training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
			training_targets = {'value_head': np.array([row['value'] for row in minibatch])
								, 'policy_head': np.array([row['AV'] for row in minibatch])} 

			fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = 32)
			lg.logger_mcts.info('NEW LOSS %s', fit.history)

			self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1],4))
			self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1],4)) 
			self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1],4)) 

		plt.plot(self.train_overall_loss, 'k')
		plt.plot(self.train_value_loss, 'k:')
		plt.plot(self.train_policy_loss, 'k--')

		plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')
		plt.savefig('run/logs/loss_{}.svg'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))

		display.clear_output(wait=True)
		display.display(pl.gcf())
		pl.gcf().clear()
		time.sleep(1.0)

		print('\n')
		self.model.printWeightAverages()

	def predict(self, inputToModel):
		preds = self.model.predict(inputToModel)
		return preds

	def buildMCTS(self, state):
		lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		self.mcts.root = self.mcts.tree[state.id]