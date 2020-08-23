import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

#### need to have all epochs also ....
#### and different lambda beta combinations ...
if __name__ == '__main__':
    correlationavgpath_list = [
'movie_review_verify_correlation_avg_matrixmlp0',
'movie_review_verify_correlation_avg_matrixmlp124',
'movie_review_verify_correlation_avg_matrixmlp249',
'movie_review_verify_correlation_avg_matrixmlp374',
'movie_review_verify_correlation_avg_matrixmlp499',
'movie_review_verify_correlation_avg_matrix_1e-06_1.0_regmlp0',
'movie_review_verify_correlation_avg_matrix_1e-06_1.0_regmlp124',
'movie_review_verify_correlation_avg_matrix_1e-06_1.0_regmlp249',
'movie_review_verify_correlation_avg_matrix_1e-06_1.0_regmlp374',
'movie_review_verify_correlation_avg_matrix_1e-06_1.0_regmlp499',
'movie_review_verify_correlation_avg_matrix_0.0001_0.001_regmlp0',
'movie_review_verify_correlation_avg_matrix_0.0001_0.001_regmlp124',
'movie_review_verify_correlation_avg_matrix_0.0001_0.001_regmlp249',
'movie_review_verify_correlation_avg_matrix_0.0001_0.001_regmlp374',
'movie_review_verify_correlation_avg_matrix_0.0001_0.001_regmlp499',
'movie_review_verify_correlation_avg_matrix_0.0001_0.01_regmlp0',
'movie_review_verify_correlation_avg_matrix_0.0001_0.01_regmlp124',
'movie_review_verify_correlation_avg_matrix_0.0001_0.01_regmlp249',
'movie_review_verify_correlation_avg_matrix_0.0001_0.01_regmlp374',
'movie_review_verify_correlation_avg_matrix_0.0001_0.01_regmlp499'
    ]
    correlationvaravgpath_list = [
'movie_review_verify_correlation_var_avg_matrixmlp0',
'movie_review_verify_correlation_var_avg_matrixmlp124',
'movie_review_verify_correlation_var_avg_matrixmlp249',
'movie_review_verify_correlation_var_avg_matrixmlp374',
'movie_review_verify_correlation_var_avg_matrixmlp499',
'movie_review_verify_correlation_var_avg_matrix_1e-06_1.0_regmlp0',
'movie_review_verify_correlation_var_avg_matrix_1e-06_1.0_regmlp124',
'movie_review_verify_correlation_var_avg_matrix_1e-06_1.0_regmlp249',
'movie_review_verify_correlation_var_avg_matrix_1e-06_1.0_regmlp374',
'movie_review_verify_correlation_var_avg_matrix_1e-06_1.0_regmlp499',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.001_regmlp0',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.001_regmlp124',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.001_regmlp249',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.001_regmlp374',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.001_regmlp499',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.01_regmlp0',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.01_regmlp124',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.01_regmlp249',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.01_regmlp374',
'movie_review_verify_correlation_var_avg_matrix_0.0001_0.01_regmlp499'
    ]
    minibatchnum_list = [
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58,
58
    ]
    criticalvalue_list = [
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749,
2.00171749
    ]

    for i in range(len(correlationavgpath_list)):
        print ('i: ', i)
        print ('correlationavgpath_list[i]: ', correlationavgpath_list[i])
        avg_matrix = np.genfromtxt(correlationavgpath_list[i], delimiter=',')
        print ('correlationvaravgpath_list[i]: ', correlationvaravgpath_list[i])
        var_avg_matrix = np.genfromtxt(correlationvaravgpath_list[i], delimiter=',')

        t_matrix = avg_matrix/np.sqrt(var_avg_matrix/minibatchnum_list[i])
        print ('t_matrix shape: ')
        print (t_matrix.shape)
        print ('t_matrix size: ')
        print (t_matrix.size)
        print ('t_matrix: ')
        print (t_matrix)
        print ((np.abs(t_matrix)>criticalvalue_list[i]).sum())
        print ((np.abs(t_matrix)>criticalvalue_list[i]))
        print ('final ratio of %s: %f'% (correlationavgpath_list[i],float((np.abs(t_matrix)>criticalvalue_list[i]).sum()/t_matrix.size)))
    # plt.hist(t_matrix.reshape(-1), bins=50, normed=1, color='g', alpha=0.75)
    # plt.savefig(args.savepath)

# python plot_correlation_null_hypothesis.py -correlationavgpath mnist_verify_correlation_avg_matrixregmlp199 -correlationvaravgpath mnist_verify_correlation_var_avg_matrixregmlp199 -minibatchnum 937 -criticalvalue 1.96249899
