import numpy as np
import torch
import logging


class ResRegularizer():
    '''Res regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''
    def __init__(self, reg_lambda=None):
        self.reg_lambda = reg_lambda
        print ("self.reg_lambda: ", self.reg_lambda)
    
    # calc the resposibilities for pj(wi)
    def calcCorrelation(self):
        logger = logging.getLogger('res_reg')
        self.feature_correlation = np.corrcoef(self.feature_matrix, rowvar=False)
        logger.info ("slef.feature_correlation.shape:")
        logger.info (self.feature_correlation.shape)

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGrad(self):
        logger = logging.getLogger('res_reg')
        correlation_abs_matrix = np.abs(self.feature_correlation)
        correlation_abs_avg = np.mean(correlation_abs_matrix)
        correlation_diff_matrix = correlation_abs_avg - correlation_abs_matrix
        reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * self.w_array
        logger.info ("reg_grad_w shape:")
        logger.info (reg_grad_w.shape)
        return reg_grad_w


    def apply(self, gpu_id, features, feature_idx, reg_lambda, epoch, param, name, step):
        logging.basicConfig(level=logging.DEBUG, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
        logger = logging.getLogger('res_reg')
        self.feature_matrix = features[feature_idx].data.cpu().numpy()
        logger.info ("feature_idx: %d", feature_idx)
        logger.info ("self.feature_matrix shape:")
        logger.info (self.feature_matrix.shape)
        logger.info ("self.feature_matrix norm: %f", np.linalg.norm(self.feature_matrix))
        self.reg_lambda = reg_lambda
        logger.info ("self.reg_lambda: %f", self.reg_lambda)
        self.w_array = param.data.cpu().numpy()
        logger.info ("self.w_array shape: ")
        logger.info (self.w_array.shape)
        self.calcCorrelation()
        self.reg_grad_w = self.calcRegGrad()
        reg_grad_w_dev = (torch.from_numpy(self.reg_grad_w)).float()
        logger.info ("step: %d", step)
        logger.info ("name: " +  name)
        logger.info ("data grad l2 norm: %f", np.linalg.norm(param.grad.data.cpu().numpy()))
        logger.info ("reg_grad_w_dev l2 norm: %f", np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        if gpu_id >= 0:
            param.grad.data.add_(1.0, reg_grad_w_dev.cuda(gpu_id)) # here3
        else:
            param.grad.data.add_(1.0, reg_grad_w_dev) # here3
        logger.info ("delta w (data grad + reg grad) norm: %f", np.linalg.norm(param.grad.data.cpu().numpy()))
        logger.info ("w norm: %f", np.linalg.norm(param.data.cpu().numpy()))
