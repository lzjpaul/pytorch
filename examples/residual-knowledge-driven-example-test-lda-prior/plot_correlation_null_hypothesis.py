import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print Null Hypothesis')
    parser.add_argument('-correlationavgpath', default='correlation avg path')
    parser.add_argument('-correlationvaravgpath', default='correlation var avg path')
    parser.add_argument('-minibatchnum', type=int, help='minibatch number')
    # parser.add_argument('-savepath', default='savepath.png')
    parser.add_argument('-criticalvalue', type=float, help='critical value')
    args = parser.parse_args()
    
    print ('args.correlationavgpath: ', args.correlationavgpath)
    avg_matrix = np.genfromtxt(args.correlationavgpath, delimiter=',')
    print ('args.correlationvaravgpath: ', args.correlationvaravgpath)
    var_avg_matrix = np.genfromtxt(args.correlationvaravgpath, delimiter=',')

    t_matrix = avg_matrix/np.sqrt(var_avg_matrix/args.minibatchnum)
    print ('t_matrix shape: ')
    print (t_matrix.shape)
    print ('t_matrix size: ')
    print (t_matrix.size)
    print ('t_matrix: ')
    print (t_matrix)
    print ((np.abs(t_matrix)>args.criticalvalue).sum())
    print ((np.abs(t_matrix)>args.criticalvalue))
    print (float((np.abs(t_matrix)>args.criticalvalue).sum()/t_matrix.size))
    # plt.hist(t_matrix.reshape(-1), bins=50, normed=1, color='g', alpha=0.75)
    # plt.savefig(args.savepath)

# python plot_correlation_null_hypothesis.py -correlationavgpath mnist_verify_correlation_avg_matrixregmlp199 -correlationvaravgpath mnist_verify_correlation_var_avg_matrixregmlp199 -minibatchnum 937 -criticalvalue 1.96249899
# python plot_correlation_null_hypothesis.py -correlationavgpath movie_review_verify_correlation_avg_matrixreglstm499 -correlationvaravgpath movie_review_verify_correlation_var_avg_matrixreglstm499 -minibatchnum 58 -criticalvalue 2.00171749
