# 현재 TIME 검사 넣은 곳
# 메모리 저장 검사 끝난 후, 가장 마지막


import time
import sys

ITERATION_LIMIT = 1	# None이면 무한 반복
TIME_LIMIT = None		# None이면 무한 반복, 시간[h] 단위로 입력



class Limiter():
	def __init__(self):
		self.START_TIME = time.time()

	def check_time(self, auto):
		if TIME_LIMIT != None :
			if time.time() - self.START_TIME > TIME_LIMIT * 3600 :
				if auto == True:
					sys.exit()
					print('Program terminated by TIME LIMIT')
				else:
					return True
			else:
				return False
		else:
			return False

	def check_iteration(self, interation):
		if interation >= ITERATION_LIMIT:
			sys.exit()
			print('Program terminated by ITERATION LIMIT')