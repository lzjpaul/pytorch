import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print Weight')
    parser.add_argument('-datapath', default='cifar-10-batches-py.csv')
    parser.add_argument('-savepath', default='save.png')
    args = parser.parse_args()

    data = np.genfromtxt(args.datapath, delimiter=',')
    print ('data shape: ')
    print (data.shape)
    plt.hist(data, bins=50, normed=1, color='g', alpha=0.75)
    plt.savefig(args.savepath.rstrip())
    # plt.show()
