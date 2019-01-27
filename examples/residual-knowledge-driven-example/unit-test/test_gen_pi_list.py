import numpy as np

def gen_pi_list(gm_num, pi_decay_ratio):
    index_array = np.zeros(101) # [-50, -49, ..., 0, -1, -2, ..., -50]
    for i in range(51):
        index_array[i] = -50 + i
    for i in range(51):
        index_array[50 + i] = -i
    print('index_array: ', index_array)
    pi_list = []
    for i in range(gm_num):
        pi_list.append(index_array[(50-i):(50+gm_num-i)]) # [[0, -1, -2, -3], [-1, 0, -1, -2], [-2, -1, 0, -1], [-3, -2, -1, 0]]
    for i in range(gm_num):
        print ('pi_list[i]: ', pi_list[i])
    # decay and normalize 
    for i in range(gm_num):
        print ('pi_list[i] * pi_decay_ratio: ', pi_list[i] * pi_decay_ratio)
        pi_list[i] = np.exp(pi_list[i] * pi_decay_ratio)
        print ('exp: ', pi_list[i])
        print ('np.sum(pi_list[i]): ', np.sum(pi_list[i]))
        pi_list[i] = pi_list[i] / float(np.sum(pi_list[i]))
        print ('normalize: ', pi_list[i])
    print (pi_list)

if __name__ == '__main__':
    gen_pi_list(3, 1)
