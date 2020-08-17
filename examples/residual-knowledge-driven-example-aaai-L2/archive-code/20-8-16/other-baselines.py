import numpy as np
import torch

### L1 norm
"""
param.grad.data.add_(float(lassostrenth), torch.sign(param.data))  # can not use inplace ...
"""

### maxnorm
### https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/constraints.py#L45-L78
### https://github.com/kevinzakka/pytorch-goodies

### need to constrain by max_val
def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))

param = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
norm = param.norm(2, dim=1, keepdim=True)
max_val = 3.8
desired = torch.clamp(norm, 0, max_val)
eps = 1e-8
param = param * (desired / (eps + norm))
print (param)

### normalize correlation
### both res_regularizer.py and res_regularizer_diff_dim.py!!!
import numpy as np
correlation_abs_matrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print ("correlation_abs_matrix.shape: \n", correlation_abs_matrix.shape)
print ("correlation_abs_matrix: \n", correlation_abs_matrix)
correlation_abs_matrix_sum = np.sum(correlation_abs_matrix, axis=1).reshape((-1,1))
print("correlation_abs_matrix_sum shape: \n", correlation_abs_matrix_sum.shape)
print("correlation_abs_matrix_sum: \n", correlation_abs_matrix_sum)
correlation_abs_matrix_sum = correlation_abs_matrix_sum.astype(float)
print("after float correlation_abs_matrix_sum: \n", correlation_abs_matrix_sum)
correlation_abs_matrix = correlation_abs_matrix / correlation_abs_matrix_sum
print ("correlation_abs_matrix: \n", correlation_abs_matrix)
print ("np.sum(correlation_abs_matrix, axis=1): \n", np.sum(correlation_abs_matrix, axis=1))
