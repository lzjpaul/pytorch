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
'mnist_verify_correlation_avg_matrixregmlp0',
'mimic-iii_verify_correlation_avg_matrixmlp0',
'mimic-iii_verify_correlation_avg_matrixregmlp0',
'movie_review_verify_correlation_avg_matrixmlp0',
'movie_review_verify_correlation_avg_matrixregmlp0',
'mimic-iii_verify_correlation_avg_matrixlstm0',
'mimic-iii_verify_correlation_avg_matrixreglstm0',
'movie_review_verify_correlation_avg_matrixlstm0',
'movie_review_verify_correlation_avg_matrixreglstm0',
'autoencoder_verify_correlation_avg_matrix_0_autoenc_0',
'autoencoder_verify_correlation_avg_matrix__10.0_1.0_0_regautoenc_0',
'autoencoder_verify_correlation_avg_matrix_1_autoenc_0',
'autoencoder_verify_correlation_avg_matrix__10.0_1.0_1_regautoenc_0',
'autoencoder_verify_correlation_avg_matrix_2_autoenc_0',
'autoencoder_verify_correlation_avg_matrix__10.0_1.0_2_regautoenc_0',
'autoencoder_verify_correlation_avg_matrix_3_autoenc_0',
'autoencoder_verify_correlation_avg_matrix__10.0_1.0_3_regautoenc_0',
'autoencoder_verify_correlation_avg_matrix_4_autoenc_0',
'autoencoder_verify_correlation_avg_matrix__10.0_1.0_4_regautoenc_0',
'autoencoder_verify_correlation_avg_matrix_5_autoenc_0',
'autoencoder_verify_correlation_avg_matrix__10.0_1.0_5_regautoenc_0',
'autoencoder_verify_correlation_avg_matrix_6_autoenc_0',
'autoencoder_verify_correlation_avg_matrix__10.0_1.0_6_regautoenc_0',
'autoencoder_verify_correlation_avg_matrix_7_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_7_regautoenc_0',
'lenet_verify_correlation_avg_matrix_0_lenet_1'
'lenet_verify_correlation_avg_matrix__1e-08_100.0_0_reglenet_1',
'lenet_verify_correlation_avg_matrix_1_lenet_1',
'lenet_verify_correlation_avg_matrix__1e-08_100.0_1_reglenet_1',
'vgg_verify_correlation_avg_matrix_0_vgg16_bn_0',
'vgg_verify_correlation_avg_matrix__1.0_0.0001_0_regvgg16_bn_0',
'vgg_verify_correlation_avg_matrix_1_vgg16_bn_0',
'vgg_verify_correlation_avg_matrix__1.0_0.0001_1_regvgg16_bn_0'
    ]
    correlationvaravgpath_list = [
'mnist_verify_correlation_var_avg_matrixmlp0',
'mnist_verify_correlation_var_avg_matrixregmlp0',
'mimic-iii_verify_correlation_var_avg_matrixmlp0',
'mimic-iii_verify_correlation_var_avg_matrixregmlp0',
'movie_review_verify_correlation_var_avg_matrixmlp0',
'movie_review_verify_correlation_var_avg_matrixregmlp0',
'mimic-iii_verify_correlation_var_avg_matrixlstm0',
'mimic-iii_verify_correlation_var_avg_matrixreglstm0',
'movie_review_verify_correlation_var_avg_matrixlstm0',
'movie_review_verify_correlation_var_avg_matrixreglstm0',
'autoencoder_verify_correlation_var_avg_matrix_0_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_0_regautoenc_0',
'autoencoder_verify_correlation_var_avg_matrix_1_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_1_regautoenc_0',
'autoencoder_verify_correlation_var_avg_matrix_2_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_2_regautoenc_0',
'autoencoder_verify_correlation_var_avg_matrix_3_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_3_regautoenc_0',
'autoencoder_verify_correlation_var_avg_matrix_4_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_4_regautoenc_0',
'autoencoder_verify_correlation_var_avg_matrix_5_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_5_regautoenc_0',
'autoencoder_verify_correlation_var_avg_matrix_6_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_6_regautoenc_0',
'autoencoder_verify_correlation_var_avg_matrix_7_autoenc_0',
'autoencoder_verify_correlation_var_avg_matrix__10.0_1.0_7_regautoenc_0',
'lenet_verify_correlation_var_avg_matrix_0_lenet_1'
'lenet_verify_correlation_var_avg_matrix__1e-08_100.0_0_reglenet_1',
'lenet_verify_correlation_var_avg_matrix_1_lenet_1',
'lenet_verify_correlation_var_avg_matrix__1e-08_100.0_1_reglenet_1',
'vgg_verify_correlation_var_avg_matrix_0_vgg16_bn_0',
'vgg_verify_correlation_var_avg_matrix__1.0_0.0001_0_regvgg16_bn_0',
'vgg_verify_correlation_var_avg_matrix_1_vgg16_bn_0',
'vgg_verify_correlation_var_avg_matrix__1.0_0.0001_1_regvgg16_bn_0'
    ]
    minibatchnum_list = [
938,
938,
124,
124,
58,
58,
124,
124,
58,
58,
469,
469,
469,
469,
469,
469,
469,
469,
469,
469,
469,
469,
469,
469,
235,
235,
235,
235,
391,
391,
391,
391
    ]
    criticalvalue_list = [
1.96249628,
1.96249628,
1.97928011,
1.97928011,
2.00171749,
2.00171749,
1.97928011,
1.97928011,
2.00171749,
2.00171749,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.965035,
1.97011,
1.97011,
1.97011,
1.97011,
1.96605,
1.96605,
1.96605,
1.96605
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
