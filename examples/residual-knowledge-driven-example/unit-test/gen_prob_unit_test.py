import numpy as np

correlation_moving_average = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, -0.6], [-0.7, 0.8, -0.9]])

correlation_abs_matrix = np.abs(correlation_moving_average)
print ('correlation_abs_matrix: ', correlation_abs_matrix)
correlation_abs_matrix_sum = np.sum(correlation_abs_matrix, axis=1).reshape((correlation_abs_matrix.shape[0],1))
print ('correlation_abs_matrix_sum shape:', correlation_abs_matrix_sum.shape)
print ('correlation_abs_matrix_sum:', correlation_abs_matrix_sum)
correlation_abs_matrix_normalize = correlation_abs_matrix / correlation_abs_matrix_sum
print ('correlation_abs_matrix_normalize shape: ', correlation_abs_matrix_normalize.shape)
print ('correlation_abs_matrix_normalize: ', correlation_abs_matrix_normalize)
print ('np.sum(correlation_abs_matrix_normalize, axis=1)', np.sum(correlation_abs_matrix_normalize, axis=1))
correlation_abs_matrix_normalize_log = np.log(correlation_abs_matrix_normalize)
print ('correlation_abs_matrix_normalize_log: ', correlation_abs_matrix_normalize_log)
print ('np.min(correlation_abs_matrix_normalize) after log', np.min(correlation_abs_matrix_normalize_log))
