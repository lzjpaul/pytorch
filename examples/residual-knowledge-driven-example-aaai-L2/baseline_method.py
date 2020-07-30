import numpy as np
import torch

class BaselineMethod():
    '''baseline methods
    '''
    def __init__(self):
        pass

    def lasso_regularization(self, param, lasso_strength):
    	param.grad.data.add_(float(lassostrenth), torch.sign(param.data))  # can not use inplace ...

    def max_norm(self, param, max_val, eps=1e-8):
    	norm = param.data.norm(2, dim=1, keepdim=True)
    	desired = torch.clamp(norm, 0, max_val)
    	param.data = param.data * (desired / (eps + norm))