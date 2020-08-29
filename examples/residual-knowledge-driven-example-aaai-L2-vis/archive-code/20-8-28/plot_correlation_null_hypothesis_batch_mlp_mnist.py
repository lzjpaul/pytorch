import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

#### need to have all epochs also ....
#### and different lambda beta combinations ...
if __name__ == '__main__':
    correlationavgpath_list = [
'mnist_verify_correlation_avg_matrixmlp0',
'mnist_verify_correlation_avg_matrixmlp49',
'mnist_verify_correlation_avg_matrixmlp99',
'mnist_verify_correlation_avg_matrixmlp149',
'mnist_verify_correlation_avg_matrixmlp199',
'mnist_verify_correlation_avg_matrix_1e-06_0.001_regmlp0',
'mnist_verify_correlation_avg_matrix_1e-06_0.001_regmlp49',
'mnist_verify_correlation_avg_matrix_1e-06_0.001_regmlp99',
'mnist_verify_correlation_avg_matrix_1e-06_0.001_regmlp149',
'mnist_verify_correlation_avg_matrix_1e-06_0.001_regmlp199'
    ]
    correlationvaravgpath_list = [
'mnist_verify_correlation_var_avg_matrixmlp0',
'mnist_verify_correlation_var_avg_matrixmlp49',
'mnist_verify_correlation_var_avg_matrixmlp99',
'mnist_verify_correlation_var_avg_matrixmlp149',
'mnist_verify_correlation_var_avg_matrixmlp199',
'mnist_verify_correlation_var_avg_matrix_1e-06_0.001_regmlp0',
'mnist_verify_correlation_var_avg_matrix_1e-06_0.001_regmlp49',
'mnist_verify_correlation_var_avg_matrix_1e-06_0.001_regmlp99',
'mnist_verify_correlation_var_avg_matrix_1e-06_0.001_regmlp149',
'mnist_verify_correlation_var_avg_matrix_1e-06_0.001_regmlp199'
    ]
    minibatchnum_list = [
938,
938,
938,
938,
938,
938,
938,
938,
938,
938
    ]
    criticalvalue_list = [
1.96249628,
1.96249628,
1.96249628,
1.96249628,
1.96249628,
1.96249628,
1.96249628,
1.96249628,
1.96249628,
1.96249628
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
