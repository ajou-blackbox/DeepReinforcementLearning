import config
import time
from agent import Agent, User

sleep_time = 30

def eval_end():
    f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
    player1_score = int(f.readline())
    drawn_score = int(f.readline())
    player2_score = int(f.readline())
    eval_count = int(f.readline())
    f.close

    if eval_count < config.EVAL_EPISODES:
        eval_count += 1
        f = open('evalcount.txt', mode = 'wt', encoding = 'utf-8')
        f.write(str(player1_score)+'\n')
        f.write(str(drawn_score)+'\n')
        f.write(str(player2_score)+'\n')
        f.write(str(eval_count))
        f.close
        return 0
    else:
        return 1

def evel_wait():
    while 1:
        f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
        player1_score = int(f.readline())
        drawn_score = int(f.readline())
        player2_score = int(f.readline())
        eval_count = int(f.readline())
        f.close
        
        if eval_count >= config.EVAL_EPISODES:
            print('Wating for other evaluation...')
            time.sleep(sleep_time)
        else:
            break

def eval_wait_full():
    while 1:
        f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
        player1_score = int(f.readline())
        drawn_score = int(f.readline())
        player2_score = int(f.readline())
        eval_count = int(f.readline())
        f.close
        
        if eval_count < config.EVAL_EPISODES:
            print('Wating for evaluations...')
            time.sleep(120)
        else:
            break

def eval_reset():
    f = open('evalcount.txt', mode = 'wt', encoding = 'utf-8')
    f.write(str(0)+'\n')
    f.write(str(0)+'\n')
    f.write(str(0)+'\n')
    f.write(str(0))
    f.close()

def add_score(scores, player1, player2):
# game eval횟수만큼 채웠으면 더 이상 돌리지 않고 대기
# scores = {'current_player':0, "drawn": 0, 'best_player':1}
# scores = {player1.name:0, "drawn": 0, player2.name:0}
    f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
    player1_score = int(f.readline())
    drawn_score = int(f.readline())
    player2_score = int(f.readline())
    eval_count = int(f.readline())
    f.close

    if scores[player1.name] == 1:
        player1_score += 1
    elif scores['drawn'] == 1:
        drawn_score += 1
    else:
        player2_score += 1

    f = open('evalcount.txt', mode = 'wt', encoding = 'utf-8')
    f.write(str(player1_score)+'\n')
    f.write(str(drawn_score)+'\n')
    f.write(str(player2_score)+'\n')
    f.write(str(eval_count))
    f.close()

    print({player1.name:player1_score, "drawn": drawn_score, player2.name:player2_score})
    
    return {player1.name : player1_score, "drawn": drawn_score, player2.name : player2_score}

#while (player1 + drawn + player2) >= config.EVAL_EPISODES or eval_count >= config.EVAL_EPISODES:
#    time.sleep(sleep_time)
    
#    f = open('evalcount.txt', mode = 'rt', encoding = 'utf-8')
#    player1 = int(f.readline())
#    drawn = int(f.readline())
#    player2 = int(f.readline())
#    eval_count = int(f.readline())
#    f.close


