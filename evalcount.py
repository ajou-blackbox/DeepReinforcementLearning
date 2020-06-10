import config
import time
from agent import Agent, User

sleep_time = 30

# 0 : player1_score
# 1 : drawn_score
# 2 : player2 score
# 3 : eval_count
# 4 : best_player_version

def eval_end():
    #f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
    #player1_score = int(f.readline())
    #drawn_score = int(f.readline())
    #player2_score = int(f.readline())
    #eval_count = int(f.readline())
    #f.close

    evalcount = read_evalcount()

    if evalcount[3] < config.EVAL_EPISODES:
        evalcount[3] += 1
        write_evalcount(evalcount)
        return 0
    else:
        return 1

def evel_wait():
    while 1:
        evalcount = read_evalcount()
        
        if evalcount[3] >= config.EVAL_EPISODES:
            print('Wating for other evaluation...')
            time.sleep(sleep_time)
        else:
            break

def eval_wait_full():
    while 1:
        evalcount = read_evalcount()
        
        if evalcount[3] < config.EVAL_EPISODES:
            print('Wating for evaluations...')
            time.sleep(120)
        else:
            break

def eval_reset():    # current_player 승리 시 1, 아니면 0
    evalcount = read_evalcount()
    evalcount[0] = 0
    evalcount[1] = 0
    evalcount[2] = 0
    evalcount[3] = 0
    #if win == 1:
    #    evalcount[4] += 1
    write_evalcount(evalcount)

def init_best_player_version(version):
    evalcount = read_evalcount()
    evalcount[4] = version
    write_evalcount(evalcount)


def add_score(scores, player1, player2):
# game eval횟수만큼 채웠으면 더 이상 돌리지 않고 대기
# scores = {'current_player':0, "drawn": 0, 'best_player':1}
# scores = {player1.name:0, "drawn": 0, player2.name:0}

    evalcount = read_evalcount()

    if scores[player1.name] == 1:
        evalcount[0] += 1
    elif scores['drawn'] == 1:
        evalcount[1] += 1
    else:
        evalcount[2] += 1

    write_evalcount(evalcount)

    print({player1.name:evalcount[0], "drawn": evalcount[1], player2.name:evalcount[2]})
    
    return {player1.name : evalcount[0], "drawn": evalcount[1], player2.name : evalcount[2]}

def best_player_version():
    evalcount = read_evalcount()
    return evalcount[4]

def read_evalcount():
    while 1:
        f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
        lines = f.readlines()
        f.close

        if lines: # 비어있으면 다시 읽어옴
            break

    lines = list(map(int, lines))
    return lines

def write_evalcount(lines):
        f = open('evalcount.txt', mode = 'wt', encoding = 'utf-8')
        lines = list(map(str, lines))
        for i in range(len(lines)):
            f.writelines(lines[i] + '\n')
        f.close

        return 0


#while (player1 + drawn + player2) >= config.EVAL_EPISODES or eval_count >= config.EVAL_EPISODES:
#    time.sleep(sleep_time)
    
#    f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
#    player1 = int(f.readline())
#    drawn = int(f.readline())
#    player2 = int(f.readline())
#    eval_count = int(f.readline())
#    f.close


