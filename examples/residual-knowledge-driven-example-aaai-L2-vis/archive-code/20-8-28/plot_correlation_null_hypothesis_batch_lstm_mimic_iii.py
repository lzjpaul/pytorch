import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

#### need to have all epochs also ....
#### and different lambda beta combinations ...
if __name__ == '__main__':
    correlationavgpath_list = [
'mimic-iii_verify_correlation_avg_matrixlstm0',
'mimic-iii_verify_correlation_avg_matrixlstm124',
'mimic-iii_verify_correlation_avg_matrixlstm249',
'mimic-iii_verify_correlation_avg_matrixlstm374',
'mimic-iii_verify_correlation_avg_matrixlstm499',
'mimic-iii_verify_correlation_avg_matrix_5.0_0.0001_reglstm0',
'mimic-iii_verify_correlation_avg_matrix_5.0_0.0001_reglstm124',
'mimic-iii_verify_correlation_avg_matrix_5.0_0.0001_reglstm249',
'mimic-iii_verify_correlation_avg_matrix_5.0_0.0001_reglstm374',
'mimic-iii_verify_correlation_avg_matrix_5.0_0.0001_reglstm499'
    ]
    correlationvaravgpath_list = [
'mimic-iii_verify_correlation_var_avg_matrixlstm0',
'mimic-iii_verify_correlation_var_avg_matrixlstm124',
'mimic-iii_verify_correlation_var_avg_matrixlstm249',
'mimic-iii_verify_correlation_var_avg_matrixlstm374',
'mimic-iii_verify_correlation_var_avg_matrixlstm499',
'mimic-iii_verify_correlation_var_avg_matrix_5.0_0.0001_reglstm0',
'mimic-iii_verify_correlation_var_avg_matrix_5.0_0.0001_reglstm124',
'mimic-iii_verify_correlation_var_avg_matrix_5.0_0.0001_reglstm249',
'mimic-iii_verify_correlation_var_avg_matrix_5.0_0.0001_reglstm374',
'mimic-iii_verify_correlation_var_avg_matrix_5.0_0.0001_reglstm499'
    ]
    minibatchnum_list = [
124,
124,
124,
124,
124,
124,
124,
124,
124,
124
    ]
    criticalvalue_list = [
1.97928011,
1.97928011,
1.97928011,
1.97928011,
1.97928011,
1.97928011,
1.97928011,
1.97928011,
1.97928011,
1.97928011
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
