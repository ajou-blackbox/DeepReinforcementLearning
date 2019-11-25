import numpy as np
import logging

ROW = 19
COL = 19
INIT_BOARD = np.zeros(ROW * COL, dtype=np.int)
INIT_CURRENT_PLAYER = -1
INIT_BOARD[180] = -INIT_CURRENT_PLAYER	# 첫 수로 정중앙에 한 수를 놓음
WIN_COUNT = 6

class Game:

	def __init__(self):
		self.currentPlayer = INIT_CURRENT_PLAYER
		self.gameState = GameState(INIT_BOARD, INIT_CURRENT_PLAYER)
		self.actionSpace = INIT_BOARD
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.grid_shape = (ROW, COL)
		self.input_shape = (2, ROW, COL)
		self.name = 'connect6'
		self.state_size = len(self.gameState.binary)
		self.action_size = len(self.actionSpace)

	def reset(self):
		self.gameState = GameState(INIT_BOARD, INIT_CURRENT_PLAYER)
		self.currentPlayer = INIT_CURRENT_PLAYER
		return self.gameState

	def step(self, action):
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		self.currentPlayer = next_state.playerTurn
		info = None
		return ((next_state, value, done, info))

	def identities(self, state, actionValues):
		identities = [(state,actionValues)]

		currentBoard = state.board
		currentAV = actionValues

		identities.append((GameState(currentBoard, state.playerTurn), currentAV))

		return identities


class GameState():
	def __init__(self, board, playerTurn):
		self.board = board
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.playerTurn = playerTurn
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = self._allowedActions()
		self.score = (0, 0)

	def _allowedActions(self):
		allowed = []
		for i in range(len(self.board)):
			if self.board[i] == 0:
				allowed.append(i)
		return allowed

	def _binary(self):

		currentplayer_position = np.zeros(len(self.board), dtype=np.int)
		currentplayer_position[self.board==self.playerTurn] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-self.playerTurn] = 1

		position = np.append(currentplayer_position,other_position)

		return (position)

	def _convertStateToId(self):
		player1_position = np.zeros(len(self.board), dtype=np.int)
		player1_position[self.board==1] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-1] = 1

		position = np.append(player1_position,other_position)

		id = ''.join(map(str,position))

		return id

	def _get2DToAction(self, row, col):
		return self.board[int(row*COL + col)]

	def _checkForEndGame(self, action):
		stoneNum = np.count_nonzero(self.board)
		if stoneNum == len(self.board):
			return 1
		
		# 돌 12개 이전이면 무조건 return 0
		if stoneNum < 12:
			return 0
		
		recentAction = int(action)
		recentRow = int(recentAction / ROW)
		recentCol = int(recentAction % ROW)
		
		leftBound = recentCol - (WIN_COUNT - 1)
		if leftBound < 0:
			leftBound = 0

		rightBound = recentCol + (WIN_COUNT - 1)
		if rightBound > COL - 1:
			rightBound = COL - 1

		upBound = recentRow - (WIN_COUNT - 1)
		if upBound < 0:
			upBound = 0

		downBound = recentRow + (WIN_COUNT - 1)
		if downBound > ROW - 1:
			downBound = ROW - 1

		# horizon direction	
		for i in range(leftBound, rightBound - (WIN_COUNT - 1) + 1):
			sum = 0
			for j in range(i, i + WIN_COUNT):
				sum += self._get2DToAction(recentRow, j)
			if abs(sum) == 6:
				return 1

		# vertical direction
		for i in range(upBound, downBound - (WIN_COUNT - 1) + 1):
			sum = 0
			for j in range(i, i + WIN_COUNT):
				sum += self._get2DToAction(j, recentCol)
			if abs(sum) == 6:
				return 1

		# \ diagonal direction
		x = recentRow
		y = recentCol

		while((x != 0) and (y != 0)):
			x -= 1
			y -= 1
		startRow = x
		startCol = y

		x = recentRow
		y = recentCol

		while((x != ROW - 1) and (y != COL - 1) ):
			x += 1
			y += 1
		endRow = x
		endCol = y

		x = startRow
		y = startCol
		while((x <= endRow - WIN_COUNT + 1) and (y >= endCol - WIN_COUNT + 1)):
			sum = 0
			for j in range(0, WIN_COUNT):
				sum += self._get2DToAction(x + j, y + j)
			if abs(sum) == 6:
				return 1
			x += 1
			y += 1

		# / diagonal direction
		x = recentRow
		y = recentCol

		while((x != 0) and (y != COL - 1)):
			x -= 1
			y += 1
		startRow = x
		startCol = y

		x = recentRow
		y = recentCol

		while((x != ROW - 1) and (y != 0) ):
			x += 1
			y -= 1
		endRow = x
		endCol = y

		x = startRow
		y = startCol
		while((x <= endRow - WIN_COUNT + 1) and (y >= endCol + WIN_COUNT - 1)):
			sum = 0
			for j in range(0, WIN_COUNT):
				sum += self._get2DToAction(x + j, y - j)
			if abs(sum) == 6:
				return 1
			x += 1
			y -= 1

		return 0
		
		

	def _getValue(self, action):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose

		if self._checkForEndGame(action) == 1:
			beforeTurnPlayer = self.playerTurn * (np.count_nonzero(self.board) % 2 == 0 and 1 or -1)
			self.score = (beforeTurnPlayer, beforeTurnPlayer * -1)
			return (beforeTurnPlayer, beforeTurnPlayer, beforeTurnPlayer * -1)
		self.score = (0, 0)
		return (0, 0, 0)


	def takeAction(self, action):
		newBoard = np.array(self.board)
		newBoard[action]=self.playerTurn

		nextPlayerTurn = self.playerTurn * (np.count_nonzero(self.board) % 2 == 0 and -1 or 1)	# 2n 마다 턴 변경 (첫수는 Default로 깔아둠)
		newState = GameState(newBoard, nextPlayerTurn)

		value = 0
		done = 0

		if newState._checkForEndGame(action):
			value = newState._getValue(action)[0]
			done = 1

		return (newState, value, done)


	def render(self, logger):
		for r in range(ROW - 1):
			logger.info([self.pieces[str(x)] for x in self.board[COL*r : (COL*r + COL)]])
		logger.info('--------------')
