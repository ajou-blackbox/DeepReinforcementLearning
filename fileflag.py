import time

sleep_time = 10 # 초 단위

def fileFlag_on(n):  # 0 : memory_temp, 1 : training_model, 2 : model_temp, 3 : now_training

    
    while 1:
        f = open('fileflag.txt', mode = 'rt', encoding = 'utf-8')
        flag_training_model = list(f.read(4))
        f.close()

        # print(''.join(flag_training_model))

        if flag_training_model[n] == '0':
           break
        elif n==3 and flag_training_model[3] == '1':
            print('A model is training!')
            time.sleep(sleep_time)
        else:
            print('The file is busy')
            time.sleep(sleep_time)

    flag_training_model[n] = '1'
    f = open('fileflag.txt', mode = 'wt', encoding = 'utf-8')
    f.write(''.join(flag_training_model))
    f.close()

    print('File flag ' + str(n) + ' ON')

    # print(''.join(flag_training_model))

# print('작업(시간 소요)')
# time.sleep(sleep_time)

def fileFlag_off(n):

    f = open('fileflag.txt', mode = 'rt', encoding = 'utf-8')
    flag_training_model = list(f.read(4))
    f.close()

    # print(''.join(flag_training_model))

    flag_training_model[n] = '0'
    f = open('fileflag.txt', mode = 'wt', encoding = 'utf-8')
    f.write(''.join(flag_training_model))
    f.close()

    print('File flag ' + str(n) + ' OFF')
    # print(''.join(flag_training_model))

def get_fileFlag(n):
    f = open('fileflag.txt', mode = 'rt', encoding = 'utf-8')
    flag_training_model = list(f.read(4))
    f.close()

    return flag_training_model[n]