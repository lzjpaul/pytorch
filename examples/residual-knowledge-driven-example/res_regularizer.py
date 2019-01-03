import numpy as np
import torch

class ResRegularizer():
    '''Res regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''
    def __init__(self, reg_lambda=None):
        self.reg_lambda = reg_lambda
        print ("self.reg_lambda: ", self.reg_lambda)
        # self.theta_alldoc = np.zeros((self.doc_num, self.topic_num))
        # for doc_idx in range(self.doc_num):
        #    self.theta_alldoc[doc_idx,:] = np.copy(theta)    
    
    # calc the resposibilities for pj(wi)
    def calcCorrelation(self):
        self.feature_correlation = np.corrcoef(self.feature_matrix, rowvar=False)
        # print ("slef.feature_correlation.shape: ", self.feature_correlation.shape)

    # singa: (word_num, doc_num)
    # pytorch: (doc_num, word_num)
    def calcRegGrad(self):
        correlation_abs_matrix = np.abs(self.feature_correlation)
        correlation_abs_avg = np.mean(correlation_abs_matrix)
        correlation_diff_matrix = correlation_abs_avg - correlation_abs_matrix
        reg_grad_w = 2 * self.reg_lambda * np.exp(correlation_diff_matrix) * self.w_array
        # print ("reg_grad_w shape: ", reg_grad_w.shape)
        return reg_grad_w


    def apply(self, gpu_id, features, feature_idx, reg_lambda, epoch, param, name, step):
        # print ("feature_idx: ", feature_idx)
        self.feature_matrix = features[feature_idx].data.cpu().numpy()
        # print ("self.feature_matrix shape: ", self.feature_matrix.shape)
        # print ("self.feature_matrix norm: ", np.linalg.norm(self.feature_matrix))
        self.reg_lambda = reg_lambda
        # print ("self.reg_lambda: ", self.reg_lambda)
        self.w_array = param.data.cpu().numpy()
        # print ("self.w_array shape: ", self.w_array.shape)
        self.calcCorrelation()
        self.reg_grad_w = self.calcRegGrad()
        reg_grad_w_dev = (torch.from_numpy(self.reg_grad_w)).float()
        # if (epoch == 0 and step < 50) or step % self.resuptfreq == 0:
        # print ("step: ", step)
        # print ("name: ", name)
        # print ("data grad l2 norm: ", np.linalg.norm(param.grad.data.cpu().numpy()))
        # print ("reg_grad_w_dev l2 norm: ", np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        if gpu_id >= 0:
            param.grad.data.add_(1.0, reg_grad_w_dev.cuda(gpu_id)) # here3
        else:
            param.grad.data.add_(1.0, reg_grad_w_dev) # here3
        # if (epoch == 0 and step < 50) or step % self.ldauptfreq == 0:
        # print ("delta w (data grad + reg grad) norm: ", np.linalg.norm(param.grad.data.cpu().numpy()))
        # print ("w norm: ", np.linalg.norm(param.data.cpu().numpy()))
        # if epoch < 2 or step % self.ldauptfreq == 0:
        #     if epoch >=2 and step % self.paramuptfreq != 0:
        #        self.calcResponsibility()
        #    self.update_LDA_EM(name, step)
        # return grad
